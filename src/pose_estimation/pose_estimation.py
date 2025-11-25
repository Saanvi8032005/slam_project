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
    mtx_str = re.findall(r"\[\[(.*?)\]\]", text, re.S)[0]
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


def pose_estimate(matches_file="matches.npz"):
    data = np.load(OUTPUT_DIR / matches_file)
    pts1 = data["pts1"]
    pts2 = data["pts2"]
    print(f"[POSE] Loaded {pts1.shape[0]} matches")

    K, dist = load_calibration(CALIB_FILE)

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
    print("[POSE] Translation t^T\n", t.T)

    #   NEW: save pose + intrinsics for triangulation stage
    pose_path = OUTPUT_DIR / "pose.npz"
    np.savez(pose_path, R=R, t=t, K=K)
    print(f"[POSE] Saved pose to {pose_path}")

    return R, t, K


if __name__ == "__main__":
    pose_estimate("matches_left03_left04.npz")
