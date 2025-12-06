
import numpy as np


def reprojection_error(pts3D, pts_img, R, t, K):
    """
    pts3D: (N,3) triangulated points (in cam1 frame or world frame
    consistent with R,t)
    pts_img: (N,2) original 2D points in that camera
    R, t, K: pose + intrinsics for that camera
    """
    pts3D = np.asarray(pts3D, dtype=np.float64)
    pts_img = np.asarray(pts_img, dtype=np.float64)

    P = K @ np.hstack((R, t))      # 3x4
    X_h = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))]).T  # 4xN

    proj = P @ X_h                 # 3xN
    proj = proj[:2] / proj[2:]     # 2xN
    proj = proj.T                  # Nx2

    err = np.linalg.norm(proj - pts_img, axis=1)  # per-point pixel error
    return err
