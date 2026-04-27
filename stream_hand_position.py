import socket
import sys
import os
import cv2
import time
import math
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

PROJECT_ROOT = r'C:\Users\adith\CARMA'
sys.path.append(PROJECT_ROOT)
from scripts.localization.localize_hand import HandLocalizer, OneEuroFilter

# ── Network config ────────────────────────────────────────────────────────────
VM_IP      = '192.168.100.133'
POS_PORT   = 5005
GRASP_PORT = 5006

pos_sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
grasp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ── Paths ─────────────────────────────────────────────────────────────────────
config_path    = os.path.join(PROJECT_ROOT, 'configs', 'camera_params.yaml')
yolo_path      = os.path.join(PROJECT_ROOT, 'models', 'train', 'weights', 'best.pt')
deformer_path  = os.path.join(PROJECT_ROOT, 'models', 'carma_deformer_epoch_20.pth')
mp_model_path  = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')

if not os.path.exists(yolo_path):
    print('best.pt not found, falling back to yolov8n-pose.pt')
    yolo_path = os.path.join(PROJECT_ROOT, 'yolov8n-pose.pt')

if not os.path.exists(mp_model_path):
    print('Downloading MediaPipe hand landmarker...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        mp_model_path
    )
    print('Downloaded.')

# ── CARMA Deformer constants ──────────────────────────────────────────────────
T_FRAMES = 4

BEND_TRIPLES = [
    (1,2,3),(2,3,4),(5,6,7),(6,7,8),(9,10,11),(10,11,12),
    (13,14,15),(14,15,16),(17,18,19),(18,19,20),
]
OPEN_FLEX_RAD   = math.radians(10.0)
CLOSED_FLEX_RAD = math.radians(90.0)
FLEX_RANGE      = CLOSED_FLEX_RAD - OPEN_FLEX_RAD

def openness_torch(kp):
    vals = []
    for a, b, c in BEND_TRIPLES:
        u    = kp[:, a] - kp[:, b]
        v    = kp[:, c] - kp[:, b]
        cos  = F.cosine_similarity(u, v, dim=-1, eps=1e-7).clamp(-1+1e-6, 1-1e-6)
        flex = math.pi - torch.acos(cos)
        vals.append((flex - OPEN_FLEX_RAD) / FLEX_RANGE)
    return torch.stack(vals, dim=1).mean(dim=1).clamp(0.0, 1.0)

def normalize_mp_landmarks(lm):
    lm = lm.copy()
    lm = lm - lm[0:1]
    scale = np.linalg.norm(lm[9])
    if scale > 1e-7:
        lm = lm / scale
    return lm

# ── Model architecture ────────────────────────────────────────────────────────
class JointEmbedding(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.proj     = nn.Linear(3, d)
        self.joint_pe = nn.Embedding(21, d)

    def forward(self, x):
        B   = x.size(0)
        ids = torch.arange(21, device=x.device).unsqueeze(0).expand(B, -1)
        return self.proj(x) + self.joint_pe(ids)

class SpatialTransformer(nn.Module):
    def __init__(self, d=256, nhead=8, n_enc=3, dropout=0.1):
        super().__init__()
        self.embed = JointEmbedding(d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_enc,
                                             enable_nested_tensor=False)
        self.norm  = nn.LayerNorm(d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        dec = nn.TransformerDecoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_layers=2)

    def forward(self, x):
        tokens  = self.embed(x)
        enc_out = self.encoder(tokens)
        q       = self.query.expand(x.size(0), -1, -1)
        f_plus  = self.decoder(q, enc_out)
        return self.norm(f_plus.squeeze(1))

class TemporalTransformer(nn.Module):
    def __init__(self, d=256, nhead=8, n_enc=3, dropout=0.1):
        super().__init__()
        self.pos_pe = nn.Embedding(T_FRAMES, d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_enc,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, T, _ = x.shape
        ids  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x    = x + self.pos_pe(ids)
        f_pp = self.encoder(x)
        return self.norm(f_pp)

class SimplifiedDFM(nn.Module):
    def __init__(self, d=256, pose_dim=63):
        super().__init__()
        self.pose_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, pose_dim))
        self.fw_head   = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, pose_dim))
        self.bw_head   = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, pose_dim))
        self.conf_head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, f_pp):
        B, T, d = f_pp.shape
        theta = self.pose_head(f_pp)
        fw    = self.fw_head(f_pp)
        bw    = self.bw_head(f_pp)
        conf  = self.conf_head(f_pp)
        warped = [theta[:, 0]]
        for t in range(T):
            candidates = [theta[:, t]]
            if t > 0:   candidates.append(theta[:, t-1] + fw[:, t-1])
            if t < T-1: candidates.append(theta[:, t+1] + bw[:, t+1])
            warped.append(torch.stack(candidates, dim=1).mean(dim=1))
        warped    = torch.stack(warped[1:], dim=1)
        conf_norm = F.softmax(conf, dim=1)
        fused     = (conf_norm * warped).sum(dim=1)
        return fused, theta, fw, bw

class CARMADeformer(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.spatial   = SpatialTransformer(d)
        self.temporal  = TemporalTransformer(d)
        self.dfm       = SimplifiedDFM(d)
        self.open_head = nn.Sequential(
            nn.Linear(63, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        B, T, J, C = x.shape
        x_flat    = x.view(B * T, J, C)
        f_plus    = self.spatial(x_flat)
        f_plus    = f_plus.view(B, T, -1)
        f_pp      = self.temporal(f_plus)
        fused, _, _, _ = self.dfm(f_pp)
        kp_pred   = fused.view(B, 21, 3)
        open_pred = self.open_head(fused).squeeze(-1)
        return {'kp_pred': kp_pred, 'open_pred': open_pred}

# ── Load CARMA Deformer ───────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

deformer = CARMADeformer().to(device)
ck = torch.load(deformer_path, map_location=device)
deformer.load_state_dict(ck['model'])
deformer.eval()

STRETCH_LO = ck.get('stretch_lo', 0.0)
STRETCH_HI = ck.get('stretch_hi', 1.0)
print(f'Deformer loaded. Stretch: lo={STRETCH_LO:.4f} hi={STRETCH_HI:.4f}')

# ── Load MediaPipe ────────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=mp_model_path)
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
)
mp_detector = mp_vision.HandLandmarker.create_from_options(hand_options)
print('MediaPipe ready.')

# ── Load hand position localizer ──────────────────────────────────────────────
localizer = HandLocalizer(
    config_path,
    yolo_path,
    filter=OneEuroFilter(time.time(), np.array([0,0,0]), min_cutoff=1.5, beta=0.01)
)
print('HandLocalizer ready.')

# ── Grasp inference function ──────────────────────────────────────────────────
def infer_grasp(frame_bgr):
    img_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result = mp_detector.detect(mp_img)

    if not result.hand_landmarks:
        return None

    lm_raw  = np.array([[l.x, l.y, l.z]
                         for l in result.hand_landmarks[0]], dtype=np.float32)
    lm_norm = normalize_mp_landmarks(lm_raw)
    lm_clip = np.stack([lm_norm] * T_FRAMES, axis=0)
    inp     = torch.tensor(lm_clip, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = deformer(inp)

    kp_pred   = out['kp_pred']
    open_pred = out['open_pred']

    direct   = float(open_pred[0].cpu())
    raw_geo  = float(openness_torch(kp_pred)[0].cpu())
    geo      = float(np.clip(
        (raw_geo - STRETCH_LO) / max(STRETCH_HI - STRETCH_LO, 1e-6), 0.0, 1.0))
    ensemble = (direct + geo) / 2.0

    return ensemble

# ── Initialize cameras ────────────────────────────────────────────────────────
cam_indices = [703, 701, 702]
cams = []
for idx in cam_indices:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cams.append(cap)
        print(f'Camera {idx} ready')
    else:
        print(f'Warning: Could not open camera {idx}')

if len(cams) < 3:
    print(f'Only {len(cams)}/3 cameras opened. Check indices.')

print(f'Streaming to {VM_IP} — pos:{POS_PORT}  grasp:{GRASP_PORT}')
print('Press Ctrl+C to stop')

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        frames = []
        for cam in cams:
            ret, frame = cam.read()
            frames.append(frame if ret else None)

        if any(f is None for f in frames):
            print('Warning: missing frame')
            continue

        # ── Hand position (all 3 cameras) ─────────────────────────────────
        pos = localizer.localize_3d(frames)
        if pos is not None:
            msg = f'{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}'
            pos_sock.sendto(msg.encode(), (VM_IP, POS_PORT))
            print(f'Pos: x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f}mm', end='   ')
        else:
            print('No hand detected', end='   ')

        # ── Grasp intensity (camera 703 = frames[0]) ──────────────────────
        intensity = infer_grasp(frames[0])
        if intensity is not None:
            grasp_sock.sendto(f'{intensity:.4f}'.encode(), (VM_IP, GRASP_PORT))
            state = 'GRASP' if intensity > 0.5 else 'OPEN'
            print(f'Grasp: {intensity:.3f} [{state}]')
        else:
            print('No grasp detected')

except KeyboardInterrupt:
    print('\nStopping...')
finally:
    for cam in cams:
        cam.release()
    mp_detector.close()
    pos_sock.close()
    grasp_sock.close()