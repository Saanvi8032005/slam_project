"""
E, mask = cv.findEssentialMat(pts1, pts2, K)
_, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

INPUTS:
Camera intrinsics K (3×3 matrix from calibration).
Matched 2D point pairs between the two images:
-   pts1: Nx2 array of points in image 1
-   pts2: Nx2 array of corresponding points in image 2

In terms of your tracking code, those come from:
-   kp1, kp2 (keypoints)
-   good (list of inlier cv.DMatch after histogram/RANSAC)
"""
import numpy as np
import cv2 as cv
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pose_estimation"
CALIB_DIR = PROJECT_ROOT / "outputs" / "calibration"
CALIB_FILE = CALIB_DIR / "calibration_results.txt"


def load_calibration(calib_path):
    """Reads camera matrix and distortion coefficients from a text file."""
    with open(calib_path, "r") as f:
        text = f.read()

    # --- Extract the 3x3 camera matrix ---
    # Find the block [[ ... ]]
    mtx_str = re.findall(r"\[\[(.*?)\]\]", text, re.S)[0]

    # Convert to list of rows
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

    return K, dist


def pose_estimate():
    # === 1. Load matches ===
    data = np.load(OUTPUT_DIR / "matches_left03_left04.npz")
    pts1 = data["pts1"]
    pts2 = data["pts2"]
    print(f"[POSE] Loaded {pts1.shape[0]} matches")

    # === 2. LOAD CALIBRATION FROM TEXT FILE ===
    K, dist = load_calibration(CALIB_FILE)

    #   print("[CALIB] Camera matrix K:\n", K)
    #   print("[CALIB] Distortion coefficients:\n", dist)

    # === 3. Pose estimation ===
    E, maskE = cv.findEssentialMat(
        pts1, pts2, K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        raise RuntimeError("Essential matrix estimation failed")

    _, R, t, maskPose = cv.recoverPose(E, pts1, pts2, K)

    print("[POSE] Rotation R:\n", R)
    print("[POSE] Translation t^\n", t.T)

    # === 4. Triangulate ===
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    pts4D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = (pts4D[:3] / pts4D[3]).T

    print(f"[POSE] Triangulated {pts3D.shape[0]} 3D points")


if __name__ == "__main__":
    pose_estimate()
