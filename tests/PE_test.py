import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.tracking.tracking import matching
from src.tracking.lsd_tracking import lsd
from src.pose_estimation.pose_estimation import pose_estimate


GT_PATH = PROJECT_ROOT / "data" / "ground_truth" / "groundtruth.txt"
DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset" / "rgb"


def quat_to_R(qx, qy, qz, qw):
    """Convert unit quaternion (x,y,z,w) to a 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    # normalise just in case
    n = (x*x + y*y + z*z + w*w) ** 0.5
    x, y, z, w = x/n, y/n, z/n, w/n

    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R


def load_tum_poses(gt_path):
    """Load TUM groundtruth file -> list of (timestamp, R_wc, t_wc)."""
    poses = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = line.split()
            if len(vals) != 8:
                continue
            ts = float(vals[0])
            tx, ty, tz = map(float, vals[1:4])
            qx, qy, qz, qw = map(float, vals[4:8])
            R_wc = quat_to_R(qx, qy, qz, qw)
            t_wc = np.array([tx, ty, tz], dtype=float).reshape(3, 1)
            poses.append((ts, R_wc, t_wc))
    return poses


def find_pose_for_timestamp(poses, ts_target, max_diff=0.02):
    """
    Find pose whose timestamp is closest to ts_target.
    Fail if difference > max_diff (seconds).
    """
    best = None
    best_dt = None
    for ts, R, t in poses:
        dt = abs(ts - ts_target)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = (ts, R, t)
    if best_dt is None or best_dt > max_diff:
        raise ValueError(f"No pose within {max_diff}s for timestamp {ts_target}, best dt={best_dt}")
    #   print(best)
    return best


def relative_pose_from_gt(gt_path, ts1, ts2, max_diff=0.02):
    """
    Compute relative pose R_21, t_21 from camera1->camera2 using TUM GT.
    ts1, ts2 are timestamps (floats) of the two RGB images.
    """
    poses = load_tum_poses(gt_path)
    _, R_wc1, t_wc1 = find_pose_for_timestamp(poses, ts1, max_diff)
    _, R_wc2, t_wc2 = find_pose_for_timestamp(poses, ts2, max_diff)

    # R_21 and t_21 as derived above:
    R_21 = R_wc2.T @ R_wc1
    t_21 = R_wc2.T @ (t_wc1 - t_wc2)  # 3x1

    return R_21, t_21


def rotation_error_deg(R_est, R_gt):
    R_err = R_est.T @ R_gt
    trace = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    return np.degrees(angle)


def direction_error_deg(t_est, t_gt):
    te = t_est.reshape(3)
    tg = t_gt.reshape(3)
    te /= np.linalg.norm(te) + 1e-12
    tg /= np.linalg.norm(tg) + 1e-12
    dot = np.clip(np.dot(te, tg), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def load_image_files():
    """
    Load image file paths from the dataset directory.
    Adjust this function based on your dataset structure.
    """
    #   image_files = list(DATA_DIR.glob("*.png"))
    image_files = sorted(DATA_DIR.glob("*.png"))   # ← sorted is crucial
    print(f"[PIPE] Found {len(image_files)} images")
    #   print("FIRST 5 FILES:", image_files[:5])
    return image_files


if __name__ == "__main__":
    image_files = load_image_files()
    error_dict = {}
    for i in range(25):
        print(f"\n[PIPE] Processing image pair {i} and {i + 1}...")
        print(f"[PIPE] Image 1: {image_files[i]}")
        print(f"[PIPE] Image 2: {image_files[i + 1]}")

        img1_path = image_files[i]
        img2_path = image_files[i + 1]

        ts1 = float(img1_path.stem)
        ts2 = float(img2_path.stem)

        print("[TEST] img1_path:", img1_path)
        print("[TEST] img2_path:", img2_path)

        MATCHER = "flann"
        FILTER = "hist"
        pts1, pts2, kp1, kp2, matches = matching(
                matcher=MATCHER,
                filter_method=FILTER,
                img1_path=img1_path,
                img2_path=img2_path,
                save_npz=False,
                unit_test=False,
                return_data=True,
                out_name=None,
                )
        pts1_line, pts2_line = lsd(img1_path, img2_path)
        pts1 = np.vstack([
            np.float32(pts1).reshape(-1, 2),
            np.float32(pts1_line).reshape(-1, 2)
        ])
        pts2 = np.vstack([
            np.float32(pts2).reshape(-1, 2),
            np.float32(pts2_line).reshape(-1, 2)
        ])
        R_est, t_est, K, maskPose = pose_estimate(pts1, pts2)

        R_gt, t_gt = relative_pose_from_gt(GT_PATH, ts1, ts2)

        rot_err = rotation_error_deg(R_est, R_gt)
        dir_err = direction_error_deg(t_est, t_gt)

        # --- store result in dictionary ---
        key = f"{img1_path.name} -> {img2_path.name}"
        error_dict[key] = {
            "rotation_error_deg": rot_err,
            "translation_dir_error_deg": dir_err,
        }

    print("\n================ POSE ERROR SUMMARY ================\n")
    for pair, metrics in error_dict.items():
        print(f"Image Pair: {pair}")
        print(f"   Rotation error:            {metrics['rotation_error_deg']:.3f} deg")
        print(f"   Translation direction err: {metrics['translation_dir_error_deg']:.3f} deg\n")
