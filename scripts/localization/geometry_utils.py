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
    """
    A = []
    for P, (u, v) in zip(projection_matrices, points_2d):
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
        
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    X_homogeneous = Vh[-1]
    point_3d = X_homogeneous[:3] / X_homogeneous[3]
    return point_3d

def get_hand_center(bbox):
    """
    bbox: [x_min, y_min, x_max, y_max]
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
