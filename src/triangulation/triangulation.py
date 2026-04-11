"""
Triangulate 3D points from in-memory data.

Parameters
----------
pts1, pts2 : (N, 2) float32
    Matched points in the two images (pixel coordinates).
R : (3, 3)
    Rotation from camera 1 to camera 2.
t : (3, 1)
    Translation direction from camera 1 to camera 2.
K : (3, 3)
    Camera intrinsics.
mask : (N, 1) or (N,), optional
    Inlier mask from recoverPose / findEssentialMat. If provided,
    only inliers are triangulated.
"""

import numpy as np
import cv2 as cv
from pathlib import Path
from tests.reprojection_err import reprojection_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
testing_dir = PROJECT_ROOT / "outputs" / "triangulation"


def triangulate_from_data(
    pts1,
    pts2,
    R,
    t,
    K,
    mask: np.ndarray | None = None,
    save_file: bool = False,
    out_name: str | None = None,
):
    """
    Triangulate 3D points from in-memory data.

    Parameters
    ----------
    pts1, pts2 : (N, 2) float-like
        Matched points in the two images (pixel coordinates).
    R : (3, 3)
        Rotation from camera 1 to camera 2.
    t : (3, 1) or (3,)
        Translation direction from camera 1 to camera 2.
    K : (3, 3)
        Intrinsic matrix.
    mask : (N, 1) or (N,), optional
        Inlier mask from recoverPose / findEssentialMat. If provided,
        we *prefer* inliers, but fall back to all points if it kills
        everything.
    """

    pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    N0 = pts1.shape[0]
    keep_idx = np.arange(N0)

    if pts1.shape != pts2.shape or pts1.shape[1] != 2:
        raise ValueError(f"[TRI] Expected pts1, pts2 of shape (N, 2), got {pts1.shape}, {pts2.shape}")

    print(f"[TRI] Initial correspondences: {pts1.shape[0]}")

    MIN_INLIERS = 25

    if mask is not None:
        m = mask.ravel().astype(bool)
        num_inliers = m.sum()
        print(f"[TRI] Using {num_inliers} inliers after mask")

        if num_inliers < MIN_INLIERS:
            print("[TRI] Too few inliers; skipping this pair entirely")
            return np.empty((0, 3), dtype=np.float32), float("inf"), np.empty((0,), dtype=int)

        pts1 = pts1[m]
        pts2 = pts2[m]
        keep_idx = keep_idx[m]

    if pts1.shape[0] < 2:
        print("[TRI] Not enough points to triangulate, returning empty array")
        return np.empty((0, 3), dtype=np.float32), float("inf"), np.empty((0,), dtype=int)

    print(f"[TRI] Loaded {pts1.shape[0]} matches for triangulation")

    # --- Projection matrices ---
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # K [I|0]
    P2 = K @ np.hstack((R, t))                         # K [R|t]

    # --- OpenCV expects 2xN float arrays ---
    pts1_T = pts1.T  # shape: 2 x N
    pts2_T = pts2.T

    pts4D = cv.triangulatePoints(P1, P2, pts1_T, pts2_T)  # (4, N)
    w = pts4D[3, :]

    # Handle near-zero w (points at infinity / numeric junk), CHECK ME
    mask_w = np.abs(w) > 1e-6
    if not np.any(mask_w):
        print("[TRI] All points had near-zero w; returning empty point cloud")
        return np.empty((0, 3), dtype=np.float32), float("inf"), np.empty((0,), dtype=int)

    pts4D = pts4D[:, mask_w]
    pts1 = pts1[mask_w]
    pts2 = pts2[mask_w]
    w = w[mask_w]
    keep_idx = keep_idx[mask_w]

    # Dehomogenise
    pts3D = (pts4D[:3] / w).T  # (N, 3)

    # --- 4) Simple geometric filtering ---
    z1 = pts3D[:, 2]
    pts3D_cam2 = (R @ pts3D.T + t).T
    z2 = pts3D_cam2[:, 2]
    mask_z = (z1 > 0) & (z2 > 0)
    # TUM depth is in metres -> 0–10 or 0–50 also ok

    pts3D = pts3D[mask_z]
    pts1 = pts1[mask_z]
    pts2 = pts2[mask_z]
    keep_idx = keep_idx[mask_z]

    if pts3D.shape[0] == 0:
        print("[TRI] WARNING: no points left after z filtering")
        return np.empty((0, 3), dtype=np.float32), float("inf"), np.empty((0,), dtype=int)

    # Mild outlier clipping by distance from origin
    d = np.linalg.norm(pts3D, axis=1)
    r98 = np.percentile(d, 98)
    mask_d = d < r98

    pts3D = pts3D[mask_d]
    pts1 = pts1[mask_d]
    pts2 = pts2[mask_d]
    keep_idx = keep_idx[mask_d]

    print(f"[TRI] Triangulated {pts3D.shape[0]} filtered 3D points")
    if True:
        print(f"[TRI] Raw triangulated points: {pts4D.shape[1]}")
        print(f"[TRI] After w-filter: {mask_w.sum()}")
        print(f"[TRI] After z>0 filter: {mask_z.sum()}")
        print(f"[TRI] After distance filter: {mask_d.sum()}")

    if save_file:
        out_name = out_name or "points.npy"
        out_path = testing_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pts3D)
        print(f"[TRI] Saved 3D points to {out_path}")

        # --- 5) Reprojection error on both cameras ---
    err1 = reprojection_error(pts3D, pts1, np.eye(3), np.zeros((3, 1)), K)
    err2 = reprojection_error(pts3D, pts2, R, t, K)

    MAX_REPROJ_ERR = 1.0  # pixels
    mask_reproj = (err1 < MAX_REPROJ_ERR) & (err2 < MAX_REPROJ_ERR)
    pts3D = pts3D[mask_reproj]
    keep_idx = keep_idx[mask_reproj]
    pts1 = pts1[mask_reproj]
    pts2 = pts2[mask_reproj]
    err1 = err1[mask_reproj]
    err2 = err2[mask_reproj]

    if pts3D.shape[0] == 0:
        print("[TRI] WARNING: no points left after reprojection filtering")
        return np.empty((0, 3), dtype=np.float32), float("inf"), np.empty((0,), dtype=int)

    mean1 = err1.mean() if err1.size > 0 else 0.0
    mean2 = err2.mean() if err2.size > 0 else 0.0
    if err1.size > 0:
        print(f"[TRI] Reproj err cam1: mean={err1.mean():.2f}px, median={np.median(err1):.2f}px")
    if err2.size > 0:
        print(f"[TRI] Reproj err cam2: mean={err2.mean():.2f}px, median={np.median(err2):.2f}px")
    if err1.size != 0 or err2.size != 0:
        err_mean = max(mean1, mean2)
    else:
        err_mean = float('inf')
    return pts3D, err_mean, keep_idx