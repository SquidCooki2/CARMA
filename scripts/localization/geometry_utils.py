import numpy as np

def get_look_at_projection_matrix(intrinsics, camera_pos, target_pos=[0, 0, 0]):
    """
    Creates a 3x4 projection matrix where the camera points at a target.
    
    Args:
        intrinsics: dict with fx, fy, cx, cy
        camera_pos: [x, y, z] position of the camera in world space
        target_pos: [x, y, z] target point to look at (default origin)
        
    Returns:
        P: 3x4 projection matrix
    """
    # 1. Intrinsic Matrix K
    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ], dtype=np.float64)

    # 2. Calculate Rotation Matrix R (Look-At)
    # Camera coordinate system (OpenCV): Z forward, X right, Y down
    C = np.array(camera_pos, dtype=np.float64)
    T = np.array(target_pos, dtype=np.float64)
    
    # Forward vector (Z_cam)
    z_cam = T - C
    z_cam /= np.linalg.norm(z_cam)
    
    # World Up (User said Z is up/down)
    world_up = np.array([0, 0, 1], dtype=np.float64)
    
    # Right vector (X_cam)
    x_cam = np.cross(world_up, z_cam)
    if np.linalg.norm(x_cam) < 1e-6: # Camera looking straight up/down
        x_cam = np.array([1, 0, 0], dtype=np.float64)
    else:
        x_cam /= np.linalg.norm(x_cam)
        
    # Down vector (Y_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)
    
    # R is the matrix that transforms world coordinates to camera coordinates
    # R = [x_cam, y_cam, z_cam].T
    R = np.vstack([x_cam, y_cam, z_cam])
    
    # 3. Translation vector t = -R * C
    t = -R @ C.reshape(3, 1)
    
    # 4. Projection Matrix P = K * [R | t]
    P = K @ np.hstack([R, t])
    return P

def triangulate_n_views(projection_matrices, points_2d):
    """
    Triangulates a 3D point from N views using the Direct Linear Transform (DLT).
    Returns (point_3d, error) where error is the geometric residual.
    """
    A = []
    for P, (u, v) in zip(projection_matrices, points_2d):
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
        
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    X_homogeneous = Vh[-1]
    point_3d = X_homogeneous[:3] / X_homogeneous[3]
    
    # Calculate Residual (Error): How far the point is from the rays
    # High error usually means one camera is seeing a false positive
    error = 0
    for P, (u, v) in zip(projection_matrices, points_2d):
        proj = P @ X_homogeneous
        u_p, v_p = proj[0]/proj[2], proj[1]/proj[2]
        error += np.sqrt((u - u_p)**2 + (v - v_p)**2)
    error /= len(points_2d)
    
    return point_3d, error

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=float)
        self.dx_prev = np.zeros_like(x0, dtype=float)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t = float(t)
        x = np.array(x, dtype=float)
        te = t - self.t_prev
        if te <= 0: return self.x_prev

        # Filter the derivative
        ad = self._alpha(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = ad * dx + (1 - ad) * self.dx_prev

        # Filter the signal
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(te, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

    def _alpha(self, te, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

def is_within_bounds(point, bounds):
    """
    point: [x, y, z]
    bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    for i in range(3):
        if point[i] < bounds[i][0] or point[i] > bounds[i][1]:
            return False
    return True

def get_hand_center(bbox):
    """
    bbox: [x_min, y_min, x_max, y_max]
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
