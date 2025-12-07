import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.tracking.tracking import matching
#   from src.tracking.lsd_tracking import lsd
from src.pose_estimation.pose_estimation import pose_estimate

# Import TUM's associate utils, get rid of . if running from here
from .associate import read_file_list, associate

GT_PATH = PROJECT_ROOT / "data" / "ground_truth" / "groundtruth.txt"
DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset" / "rgb"


def quat_to_R(qx, qy, qz, qw):
    """Convert unit quaternion (x,y,z,w) to a 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    n = (x*x + y*y + z*z + w*w) ** 0.5
    x, y, z, w = x/n, y/n, z/n, w/n

    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=float)
    return R


def rotation_error_deg(R_est, R_gt):
    """
    Smallest angle between two rotation matrices in degrees.
    """
    R_err = R_est.T @ R_gt
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    return np.degrees(angle)


def translation_direction_error_deg(t_est, t_gt):
    """
    Compute translation direction error in degrees.
    Handles t vs -t sign ambiguity by taking the smaller angle.
    """
    t_est = t_est.reshape(3)
    t_gt = t_gt.reshape(3)

    t_est = t_est / np.linalg.norm(t_est)
    t_gt = t_gt / np.linalg.norm(t_gt)

    # angle between t_est and +t_gt
    dot1 = np.clip(np.dot(t_est, t_gt), -1.0, 1.0)
    angle1 = np.degrees(np.arccos(dot1))

    # angle between t_est and -t_gt
    dot2 = np.clip(np.dot(t_est, -t_gt), -1.0, 1.0)
    angle2 = np.degrees(np.arccos(dot2))

    return min(angle1, angle2)


def load_image_files():
    """
    Load image file paths from the dataset directory.
    """
    image_files = sorted(DATA_DIR.glob("*.png"))
    print(f"[PIPE] Found {len(image_files)} images")
    return image_files


# -------------------------------------------------------------------
# Build RGB timestamp → (R_wc, t_wc) map using TUM associate.py
# -------------------------------------------------------------------
def build_rgb_to_gt_pose_map(gt_path, image_files, max_diff=0.02):
    """
    Build a map from RGB timestamps (from image filenames) to ground-truth
    poses using TUM's associate() to align timestamps.

    rgb_list:  ts -> ["filename.png"]
    gt_list:   ts -> ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    """
    # 1) Build rgb_list from image filenames
    rgb_list = {}
    for img_path in image_files:
        ts = float(img_path.stem)         # e.g. "1305031452.859642"
        rgb_list[ts] = [img_path.name]    # dummy data, associate() only uses keys

    # 2) Load ground truth with read_file_list
    gt_list = read_file_list(str(gt_path))

    # 3) Associate using TUM's function
    matches = associate(rgb_list, gt_list, offset=0.0, max_difference=max_diff)
    print(f"[GT] Associated {len(matches)} rgb timestamps with ground truth")

    rgb_to_gt_pose = {}

    for rgb_ts, gt_ts in matches:
        gt_data = gt_list[gt_ts]
        if len(gt_data) != 7:
            continue

        tx, ty, tz, qx, qy, qz, qw = map(float, gt_data)
        R_wc = quat_to_R(qx, qy, qz, qw)
        t_wc = np.array([tx, ty, tz], dtype=float).reshape(3, 1)

        rgb_to_gt_pose[float(rgb_ts)] = (R_wc, t_wc)

    return rgb_to_gt_pose


def relative_pose_from_rgb_ts(rgb_to_gt_pose, ts1, ts2):
    """
    Compute relative pose R_21, t_21 from camera1->camera2
    using the rgb_ts -> (R_wc, t_wc) map produced above.
    """
    if ts1 not in rgb_to_gt_pose or ts2 not in rgb_to_gt_pose:
        raise KeyError(f"Missing GT pose for ts1={ts1} or ts2={ts2}")

    R_wc1, t_wc1 = rgb_to_gt_pose[ts1]
    R_wc2, t_wc2 = rgb_to_gt_pose[ts2]

    # X2 = R_21 X1 + t_21, with world-frame poses R_wc, t_wc:
    # R_21 = R_wc2^T R_wc1
    # t_21 = R_wc2^T (t_wc1 - t_wc2)
    R_21 = R_wc2.T @ R_wc1
    t_21 = R_wc2.T @ (t_wc1 - t_wc2)

    return R_21, t_21


def run():
    # Build GT map once using associate.py
    image_files = load_image_files()
    rgb_to_gt_pose = build_rgb_to_gt_pose_map(GT_PATH, image_files, max_diff=0.02)
    error_dict = {}

    # Evaluate first N pairs
    N_PAIRS = 10
    for i in range(N_PAIRS):
        print(f"\n[PIPE] Processing image pair {i} and {i + 1}...")
        img1_path = image_files[i]
        img2_path = image_files[i + 1]

        print(f"[PIPE] Image 1: {img1_path}")
        print(f"[PIPE] Image 2: {img2_path}")

        # timestamps from filenames: "1305031452.859642.png"
        ts1 = float(img1_path.stem)
        ts2 = float(img2_path.stem)

        # --- matching ---
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

        # If you want line segments as well, uncomment this:
        """
        pts1_line, pts2_line = lsd(img1_path, img2_path)
        pts1 = np.vstack([
            np.float32(pts1).reshape(-1, 2),
            np.float32(pts1_line).reshape(-1, 2),
        ])
        pts2 = np.vstack([
            np.float32(pts2).reshape(-1, 2),
            np.float32(pts2_line).reshape(-1, 2),
        ])
        """

        # --- estimate pose ---
        R_est, t_est, K, maskPose = pose_estimate(pts1, pts2)

        # --- ground truth relative pose (via associate-based map) ---
        R_gt, t_gt = relative_pose_from_rgb_ts(rgb_to_gt_pose, ts1, ts2)

        # --- compute errors ---
        rot_err = rotation_error_deg(R_est, R_gt)
        dir_err = translation_direction_error_deg(t_est, t_gt)

        print(f"[RESULT] Rotation error:            {rot_err:.3f} deg")
        print(f"[RESULT] Translation direction err: {dir_err:.3f} deg")

        # store in dict
        key = f"{img1_path.name} -> {img2_path.name}"
        error_dict[key] = {
            "rotation_error_deg": rot_err,
            "translation_direction_error_deg": dir_err,
        }
    return error_dict


if __name__ == "__main__":
    error_dict = run()
    # ----------------------------------------------------------------
    # Pretty summary
    # ----------------------------------------------------------------
    print("\n================ POSE ERROR SUMMARY ================\n")
    for pair, metrics in error_dict.items():
        print(f"Image Pair: {pair}")
        print(f"   Rotation error:            {metrics['rotation_error_deg']:.3f} deg")
        print(f"   Translation direction err: {metrics['translation_direction_error_deg']:.3f} deg\n")
    print("====================================================\n")
