"""
Estimate relative pose (R, t) between two views from matched points.

Parameters
----------
pts1, pts2 : (N, 2) float32
    Matched pixel coordinates (x, y) in image 1 and image 2.
K : (3, 3) float32, optional
    Camera intrinsic matrix. If None, loaded from CALIB_FILE.
dist : (5,) or (N,) float32, optional
    Distortion coefficients (currently not used explicitly; you should
    ideally undistort points before passing them here).

Returns
-------
R : (3, 3) float64
    Rotation from camera 1 to camera 2.
t : (3, 1) float64
    Unit translation direction (scale is unknown in monocular case).
K : (3, 3) float32
    Intrinsic matrix used.
maskPose : (N,) uint8
    Inlier mask returned by recoverPose (1 = inlier).
"""

import numpy as np
import cv2 as cv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIB_DIR = PROJECT_ROOT / "outputs" / "calibration"
CALIB_FILE = CALIB_DIR / "calibration_results.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pose_estimation"


def load_calibration():
    """
    Load camera intrinsic matrix and distortion coefficients.
    """
    f2 = False
    if f2:
        K = np.array([
            [520.9,   0.0, 325.1],
            [0.0, 521.0, 249.7],
            [0.0,   0.0,   1.0]
        ], dtype=np.float32)
        dist = np.array([
            0.2312,
            -0.7849,
            -0.0033,
            -0.0001,
            0.9172
        ], dtype=np.float32)
    else:
        K = np.array([
            [517.3,   0.0, 318.6],
            [0.0, 516.5, 255.3],
            [0.0,   0.0,   1.0]
        ], dtype=np.float32)
        dist = np.array([
            -0.29426946,
            0.12324915,
            0.00113851,
            -0.00013802,
            0.01020549
        ], dtype=np.float32)
    return K, dist


def pose_estimate(pts1, pts2, idx_i=None, idx_j=None, log_err: bool | None = None):
    """
    Estimate relative pose (R, t) between two views using matched points.
    Includes parallax gating, Essential matrix rank check, and cheirality check.

    Parameters
    ----------
    pts1, pts2 : (N, 2) float32
        Matched pixel coordinates in image 1 and image 2.

    Returns
    -------
    R : (3, 3) float64
        Rotation from camera 1 to camera 2.
    t : (3, 1) float64
        Unit translation direction (scale is unknown in monocular case).
    K : (3, 3) float32
        Intrinsic matrix used.
    maskPose : (N,) uint8
        Inlier mask returned by recoverPose (1 = inlier).
    """
    # Ensure proper type
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)

    print(f"[POSE] Loaded {pts1.shape[0]} matches")
    K, dist = load_calibration()

    # Compute parallax (flow)
    flow = np.linalg.norm(pts2 - pts1, axis=1)
    median_flow = np.median(flow)
    PARALLAX_PX_THRESH = 1.0  # Threshold for parallax in pixels

    print(f"[POSE] Median flow: {median_flow:.2f} px")
    if median_flow < PARALLAX_PX_THRESH:
        print(f"[POSE] Low parallax: median flow {median_flow:.2f} px -> skipping pair")
        return None, None, K, None, 0, 0.0, None, None, None

    # Undistort points
    #   pts1 = cv.undistortPoints(pts1.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    #   pts2 = cv.undistortPoints(pts2.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        raise ValueError("[POSE] No points provided for pose estimation")
    if pts1.shape != pts2.shape or pts1.shape[1] != 2:
        raise ValueError(f"[POSE] Expected pts1, pts2 of shape (N, 2), got {pts1.shape}, {pts2.shape}")

    # Estimate essential matrix
    E, maskE = cv.findEssentialMat(
        pts1, pts2, K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=0.75  # Increased threshold for RANSAC
    )
    if E is None:
        raise RuntimeError("[POSE] Essential matrix estimation failed")

    # Check if E is of rank 2
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.matrix_rank(E) != 2:
        print("[POSE][WARN] Essential matrix is not rank 2, enforcing rank 2 constraint")
        S[2] = 0  # Set the smallest singular value to 0
        E = U @ np.diag(S) @ Vt

    # Use inliers from RANSAC for pose recovery
    pts1_inliers = pts1[maskE.ravel() == 1]
    pts2_inliers = pts2[maskE.ravel() == 1]

    # Recover pose
    n_inliers, R, t, maskPose = cv.recoverPose(E, pts1_inliers, pts2_inliers, K)
    t = t.reshape(3, 1)
    t /= (np.linalg.norm(t) + 1e-12)

    # Check if R is a valid rotation matrix
    if not (np.allclose(np.dot(R.T, R), np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0)):
        print("[POSE][ERROR] Recovered R is not a valid rotation matrix")
        return None, None, K, None, 0, 0.0, None, None, None

    # Debugging: Check inliers after recoverPose
    maskPose_bool = maskPose.ravel().astype(bool)
    num_inliers_pose = maskPose_bool.sum()
    num_considered = len(maskPose_bool)
    ratio_pose = num_inliers_pose / max(num_considered, 1)

    # Essential matrix inliers (count, not sum of values)
    E_inliers = (maskE.ravel() > 0).sum()
    total_matches = len(pts1)

    if num_inliers_pose < 50 or ratio_pose < 0.1:
        print(f"[POSE][WARN] Pose invalid: inliers {num_inliers_pose}/{num_considered} ({ratio_pose:.2f}) -> skipping")
        return None, None, K, None, num_inliers_pose, ratio_pose, None, None, None

    # Cheirality (physical validity) gate
    cheirality_ratio = num_inliers_pose / max(E_inliers, 1)
    if cheirality_ratio < 0.2:
        print("[POSE][WARN] Weak cheirality; skipping pose")
        return None, None, K, None, num_inliers_pose, ratio_pose, None, None, None
    
    print("[POSE] Rotation R:\n", R)
    print("[POSE] Translation t^T\n", t.T)
    print(f"[POSE] E inliers: {E_inliers}/{total_matches} ({E_inliers/max(total_matches,1):.2f})")
    print(f"[POSE] Pose inliers: {num_inliers_pose}/{E_inliers} ({num_inliers_pose/max(E_inliers,1):.2f})")


    pts1_final = pts1_inliers[maskPose_bool]
    pts2_final = pts2_inliers[maskPose_bool]   
    idx_i_inl = idx_i[maskE.ravel() == 1][maskPose_bool] if idx_i is not None else None
    idx_j_inl = idx_j[maskE.ravel() == 1][maskPose_bool] if idx_j is not None else None

    return R, t, K, num_inliers_pose, ratio_pose, pts1_final, pts2_final, idx_i_inl, idx_j_inl

if __name__ == "__main__":
    print('Run from pipeline.py')
