# triangulation.py

import numpy as np
import cv2 as cv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pose_estimation"
TEMP_DIR = PROJECT_ROOT / "outputs" / "temp"


def triangulate_from_files(
    matches_file="matches.npz",
    pose_file="pose.npz",
    out_file=None,
):
    """
    Stage 3: Triangulate 3D points using:
    - 2D matches (pts1, pts2) from matches_file
    - Pose (R, t) and intrinsics K from pose_file
    """
    match_path = Path(matches_file)
    pose_path = Path(pose_file)
    if out_file is None:
        # default: save points next to matches file
        out_path = match_path.with_name("points.npy")
    else:
        out_path = Path(out_file)

    # === 1. Load matches ===
    data = np.load(match_path)
    pts1 = data["pts1"]
    pts2 = data["pts2"]
    print(f"[TRI] Loaded {pts1.shape[0]} matches from {match_path}")

    # === 2. Load pose & intrinsics ===
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

    w = pts4D[3, :]
    good_w = np.abs(w) > 1e-6
    pts4D = pts4D[:, good_w]

    pts3D = (pts4D[:3] / pts4D[3]).T

    print(f"[TRI] Triangulated {pts3D.shape[0]} 3D points")

    # === 5. Save 3D points ===

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pts3D)
    print(f"[TRI] Saved 3D points to {out_path}")

    return pts3D


if __name__ == "__main__":
    triangulate_from_files()
