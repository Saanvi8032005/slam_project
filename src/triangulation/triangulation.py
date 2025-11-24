# triangulation.py

import numpy as np
import cv2 as cv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pose_estimation"


def triangulate_from_files(
    matches_file="matches_left03_left04.npz",
    pose_file="pose_left03_left04.npz"
):
    """
    Stage 3: Triangulate 3D points using:
    - 2D matches (pts1, pts2) from matches_file
    - Pose (R, t) and intrinsics K from pose_file
    """
    # === 1. Load matches ===
    match_path = OUTPUT_DIR / matches_file
    data = np.load(match_path)
    pts1 = data["pts1"]
    pts2 = data["pts2"]
    print(f"[TRI] Loaded {pts1.shape[0]} matches from {match_path}")

    # === 2. Load pose & intrinsics ===
    pose_path = OUTPUT_DIR / pose_file
    pose_data = np.load(pose_path)
    R = pose_data["R"]
    t = pose_data["t"]
    K = pose_data["K"]

    print("[TRI] Using pose:")
    print("R =\n", R)
    print("t^T =", t.T)
    print("K =\n", K)

    # === 3. Build projection matrices ===
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # === 4. Triangulate ===
    pts4D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = (pts4D[:3] / pts4D[3]).T

    print(f"[TRI] Triangulated {pts3D.shape[0]} 3D points")

    # === 5. Save 3D points ===

    pts_path = OUTPUT_DIR / "points_left03_left04.npy"
    np.save(pts_path, pts3D)
    print(f"[TRI] Saved 3D points to {pts_path}")

    return pts3D


if __name__ == "__main__":
    triangulate_from_files()
