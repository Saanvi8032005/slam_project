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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
testing_dir = PROJECT_ROOT / "outputs" / "triangulation"


def triangulate(
    pts1: np.ndarray,
    pts2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,     # notation confusing
    out_name: str | None = None
):
    # === 1. Load matches ===
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        raise ValueError("[TRI] No points provided")

    if pts1.shape != pts2.shape or pts1.shape[1] != 2:
        raise ValueError(f"[TRI] Expected pts1, pts2 of shape (N, 2), got {
            pts1.shape}, {pts2.shape}")

    #   when is mask none
    if mask is not None:
        mask = np.asarray(mask).ravel().astype(bool)
        pts1 = pts1[mask]
        pts2 = pts2[mask]
        print(f"[TRI] Using {pts1.shape[0]} inliers after pose mask")

    print(f"[TRI] Loaded {pts1.shape[0]} matches")

    # === 3. Build projection matrices ===
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # === 4. Triangulate ===
    pts4D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)

    w = pts4D[3, :]
    good_w = np.abs(w) > 1e-6               # oddly specific
    pts4D = pts4D[:, good_w]

    pts3D = (pts4D[:3] / pts4D[3]).T

    # --- Simple geometric filtering: keep points in front of camera ---
    z = pts3D[:, 2]
    mask_z = (z > 0) & (z < 5e4)   # tweak upper bound if needed
    pts3D = pts3D[mask_z]

    # Remove extreme distance outliers (e.g. top 2% by radius)
    d = np.linalg.norm(pts3D, axis=1)
    r98 = np.percentile(d, 98)
    pts3D = pts3D[d < r98]

    print(f"[TRI] Triangulated {pts3D.shape[0]} 3D points")

    # === 5. Save 3D points ===
    save_file = False
    if save_file:
        out_name = f"points_{out_name}.npy"
        out_path = testing_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pts3D)
        print(f"[TRI] Saved 3D points to {out_path}")

    return pts3D


def triangulate_from_data(
    pts1,
    pts2,
    R,
    t,
    K,
    mask=None,
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
        we *prefer* inliers, but fall back to all points if it kills everything.
    """

    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    # --- Basic sanity ---
    if pts1.shape != pts2.shape or pts1.shape[1] != 2:
        raise ValueError(f"[TRI] Expected pts1, pts2 of shape (N, 2), got {pts1.shape}, {pts2.shape}")

    print(f"[TRI] Initial correspondences: {pts1.shape[0]}")

    MIN_INLIERS = 30  # or 50, tune this

    if mask is not None:
        m = mask.ravel().astype(bool)
        num_inliers = m.sum()
        print(f"[TRI] Using {num_inliers} inliers after pose mask")

        if num_inliers < MIN_INLIERS:
            print("[TRI] Too few inliers; skipping this pair entirely")
            return np.empty((0, 3), dtype=np.float32)

        pts1 = pts1[m]
        pts2 = pts2[m]

    if pts1.shape[0] < 2:
        print("[TRI] Not enough points to triangulate, returning empty array")
        return np.empty((0, 3), dtype=np.float32)

    print(f"[TRI] Loaded {pts1.shape[0]} matches for triangulation")

    # --- Projection matrices ---
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # K [I|0]
    P2 = K @ np.hstack((R, t))                         # K [R|t]

    # --- OpenCV expects 2xN float arrays ---
    pts1_T = pts1.T  # shape: 2 x N
    pts2_T = pts2.T

    pts4D = cv.triangulatePoints(P1, P2, pts1_T, pts2_T)  # (4, N)

    w = pts4D[3, :]
    good_w = np.abs(w) > 1e-6
    if not np.any(good_w):
        print("[TRI] All points had near-zero w; returning empty point cloud")
        return np.empty((0, 3), dtype=np.float32)

    pts4D = pts4D[:, good_w]
    pts3D = (pts4D[:3] / pts4D[3]).T  # (N_good, 3)

    # Simple z filtering
    z = pts3D[:, 2]
    mask_z = (z > 0) & (z < 100)
    pts3D = pts3D[mask_z]

    if pts3D.shape[0] == 0:
        print("[TRI] WARNING: no points left after z filtering")
        return pts3D

    # Mild outlier clipping by distance
    d = np.linalg.norm(pts3D, axis=1)
    r98 = np.percentile(d, 98)
    pts3D = pts3D[d < r98]

    print(f"[TRI] Triangulated {pts3D.shape[0]} filtered 3D points")
    print(
        f"[TRI] Stats: "
        f"x[{pts3D[:,0].min():.3f}, {pts3D[:,0].max():.3f}], "
        f"y[{pts3D[:,1].min():.3f}, {pts3D[:,1].max():.3f}], "
        f"z[{pts3D[:,2].min():.3f}, {pts3D[:,2].max():.3f}]"
    )

    if save_file:
        out_name = out_name or "points.npy"
        out_path = testing_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pts3D)
        print(f"[TRI] Saved 3D points to {out_path}")

    return pts3D


if __name__ == "__main__":
    triangulate()
