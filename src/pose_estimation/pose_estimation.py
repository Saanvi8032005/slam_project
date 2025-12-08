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
#   import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIB_DIR = PROJECT_ROOT / "outputs" / "calibration"
CALIB_FILE = CALIB_DIR / "calibration_results.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pose_estimation"


#   check notation of () below pls
"""
    def load_calibration(calib_path: Path = CALIB_FILE):
    Reads camera matrix and distortion coefficients from a text file.
    with open(calib_path, "r") as f:
        text = f.read()

    # --- Extract the 3x3 camera matrix ---
    mtx_str = re.findall(r"\[\[(.*?)\]\]", text, re.S)[0]
    if not mtx_str:
        raise ValueError("Could not find camera matrix in calibration file")

    rows = mtx_str.strip().split("\n")
    K = np.array([
        [float(n) for n in row.replace("[", "").replace("]", "").split()]
        for row in rows
    ], dtype=np.float32)

    # --- Extract distortion ---
    dist_str = re.findall(
        r"Distortion coefficients \(dist\):\s*\[(.*?)\]", text, re.S
    )[0]
    dist = np.array([float(x) for x in dist_str.split()], dtype=np.float32)
    if not dist:
        raise ValueError("Could not find distortion
        coefficients in calibration file")

    return K, dist
"""


def load_calibration():
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


def pose_estimate(pts1,
                  pts2,
                  log_err: bool | None = None
                  ):
    # Ensure proper type
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)

    print(f"[POSE] Loaded {pts1.shape[0]} matches")
    K, dist = load_calibration()

    #   pts1 = cv.undistortPoints(pts1.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    #   pts2 = cv.undistortPoints(pts2.reshape(-1, 1, 2), K, dist).reshape(-1, 2)

    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        raise ValueError("[POSE] No points provided for pose estimation")
    if pts1.shape != pts2.shape or pts1.shape[1] != 2:
        raise ValueError(f"[POSE] Expected pts1, pts2 of shape (N, 2), got {pts1.shape}, {pts2.shape}")

    if K is None:
        K, dist_loaded = load_calibration()
        if dist is None:
            dist = dist_loaded

    E, maskE = cv.findEssentialMat(
        pts1, pts2, K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if E is None:
        raise RuntimeError("Essential matrix estimation failed")

    maskE_bool = maskE.ravel().astype(bool)
    num_inliers_E = maskE_bool.sum()
    ratio_E = num_inliers_E / len(maskE_bool)
    print(f"[POSE] RANSAC inliers (findEssentialMat): "
    f"{num_inliers_E}/{len(maskE_bool)} ({ratio_E:.2f})")

    n_inliers, R, t, maskPose = cv.recoverPose(E, pts1, pts2, K, mask=maskE)
    t = t.reshape(3, 1)
    t /= (np.linalg.norm(t) + 1e-12)

    # maskPose is defined over the same subset where maskE == 1
    maskPose_bool = maskPose.ravel().astype(bool)
    num_inliers_pose = maskPose_bool.sum()
    # number of points actually considered by recoverPose:
    num_considered = len(maskPose_bool)
    ratio_pose = num_inliers_pose / max(num_considered, 1)

    save_file = False
    if save_file:
        #   NEW: save pose + intrinsics for triangulation stage
        pose_path = OUTPUT_DIR / "pose.npz"
        np.savez(pose_path, R=R, t=t, K=K)
        print(f"[POSE] Saved pose to {pose_path}")


    # Flagging bad poses, was 0.05
    if num_inliers_pose < 50 or ratio_pose < 0.1:
        print("[POSE][WARN] Very few inliers – pose may be unreliable")

    print("[POSE] Rotation R:\n", R)
    print("[POSE] Translation t^T\n", t.T)
    print(
        f"[POSE] Inliers after recoverPose: "
        f"{num_inliers_pose}/{num_considered} ({ratio_pose:.2f})"
    )
    return R, t, K, maskPose, num_inliers_pose, ratio_pose


if __name__ == "__main__":
    print('Run from pipeline.py')
