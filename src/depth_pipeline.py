"""
pipeline.py

Third script to run the individual stages in order:
1) Tracking / matching  (combined.py)
2) Pose estimation      (pose_estimation.py)
"""

from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tracking.tracking import matching
#   from tracking.lsd_tracking import lsd
from pose_estimation.pose_estimation import pose_estimate
from triangulation.triangulation import triangulate_from_data
from visualising.visualising import visualize_points
from aligning_pc.aligning_pc import align_point_clouds
from tests.pose_estimation_eval import (
        build_rgb_to_gt_pose_map,
        relative_pose_from_rgb_ts,
        rotation_error_deg,
        translation_direction_error_deg,
        GT_PATH,
    )
from keyframe_selection.keyframe_selec import Map, Edge, print_map, Keyframe
from keyframe_selection.keyframe_helpers import (
    initialize_map,  # Fixed function name
    add_map_edge,
)
from pose_graph_optimization.pose_graph_optimization import optimise_pose_graph
from utils.trajectory_utils import save_estimated_trajectory
from depth.util import (
    depth_to_meters,
    backproject_keypoint,
    backproject_keypoints,
    matched_pixels_to_3d2d,
    stage_pose_rgbd,
    generate_3d_points_from_depth,
    keypoints_with_depth,
    load_txt_entries,
)

DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset" / "rgb"
TEMP_DIR = PROJECT_ROOT / "outputs" / "temp"
RGB_DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset"
RGB_TXT = PROJECT_ROOT / "data" / "rgb_dataset" / "rgb.txt"
DEPTH_DIR = PROJECT_ROOT / "data" / "rgb_dataset" / "depth"
DEPTH_TXT = PROJECT_ROOT / "data" / "rgb_dataset" / "depth.txt"
FX = 525.0
FY = 525.0
CX = 319.5
CY = 239.5
K_RGBD = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0,  0,  1]
], dtype=np.float64)


def load_rgb_entries(rgb_txt_path, dataset_root):
    timestamps = []
    image_files = []

    with open(rgb_txt_path, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            ts, rel_path = line.split()
            timestamps.append(float(ts))
            image_files.append(dataset_root / rel_path)

    print(f"[PIPE] Found {len(image_files)} images from rgb.txt")
    return timestamps, image_files


def stage_tracking(img1, img2, pair_id, tracking_results, pts_3d_1=None, pts_3d_2=None):
    """
    Stage 1: run feature detection + matching + filtering.
    Writes pts1, pts2 to an .npz file for pose_estimation to use.
    """
    MATCHER = "flann"
    FILTER = "hist"

    print(f"\n=== STAGE 1: TRACKING / MATCHING ({pair_id}) ===")

    out_file = f"matches_{pair_id}.npz"
    pts1, pts2, kp1, kp2, matches, _, _, _, _ = matching(
             matcher=MATCHER,
             filter_method=FILTER,
             img1_path=img1,
             img2_path=img2,
             save_npz=False,
             unit_test=False,
             return_data=True,
             out_name=out_file,
             )

    pts1_all = np.vstack([
        np.float32(pts1).reshape(-1, 2),
        #   np.float32(pts1_line).reshape(-1, 2)
    ])

    pts2_all = np.vstack([
        np.float32(pts2).reshape(-1, 2),
        #   np.float32(pts2_line).reshape(-1, 2)
    ])
    entry = {
        "pair_id": pair_id,
        "img1": img1,
        "img2": img2,
        "pts1": pts1_all,
        "pts2": pts2_all,
        "kp1": kp1,
        "kp2": kp2,
        "matches": matches,
    }
    tracking_results[pair_id] = entry

    return entry


def stage_pose(
        tracking_entry,
        pose_store=None,
        img1=None,
        img2=None,
        ts1=None,
        ts2=None
    ):
    pair_id = tracking_entry["pair_id"]
    pts1 = tracking_entry["pts1"]
    pts2 = tracking_entry["pts2"]

    print(f"\n=== STAGE 2: POSE ESTIMATION ({pair_id}) ===")
    result = pose_estimate(pts1, pts2)

    # Handle skipped pairs
    if result[0] is None:
        print(f"[PIPE] Skipping pair {pair_id} due to low parallax or pose estimation failure")
        return None, None, None, 0, 0.0, None, None, None, None

    R, t, K, num_inliers, ratio, pts1, pts2, _, _ = result
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-12:
        t = t / t_norm
    STEP_SCALE = 0.05
    t = t * STEP_SCALE

    if pose_store is not None:
        pose_store[pair_id] = {
            "pair_id": pair_id,
            "R": R,
            "t": t,
            "K": K,
            "pts1": pts1,
            "pts2": pts2,
        }

    error_print = True
    if error_print:
        rgb_to_gt_pose = build_rgb_to_gt_pose_map(
                            GT_PATH,
                            image_files,
                            max_diff=0.02
                        )
        R_gt, t_gt = relative_pose_from_rgb_ts(rgb_to_gt_pose, ts1, ts2)

        # Errors
        rot_err = rotation_error_deg(R, R_gt)
        dir_err = translation_direction_error_deg(t, t_gt)
        print(f"[POSE][GT] Rotation error:            {rot_err:.3f} deg")
        print(f"[POSE][GT] Translation direction err: {dir_err:.3f} deg")

    # should just return pose_store tbh
    return R, t, K, num_inliers, ratio, rot_err, dir_err, pts1, pts2


def stage_triangulate(tracking_entry, pose_entry, points_store=None):
    """
    Stage 3: Triangulation from in-memory pts + pose.

    tracking_entry: dict from tracking_results[pair_id]
    pose_entry:     dict from pose_results[pair_id]
    points_store:   optional dict to accumulate 3D points per pair
    """
    print("\n=== STAGE 3: TRIANGULATION ===")

    pair_id = tracking_entry["pair_id"]

    # Extract matched points and pose data
    pts1 = pose_entry["pts1"]
    pts2 = pose_entry["pts2"]
    R = pose_entry["R"]
    t = pose_entry["t"]
    K = pose_entry["K"]

    # Call triangulation function
    pts3D, err_mean, _ = triangulate_from_data(
        pts1,
        pts2,
        R,
        t,
        K,
        mask=None,
        out_name=f"points_{pair_id}.npy",
    )

    if pts3D.shape[0] == 0:
        print(f"[PIPE] Skipping pair {pair_id} due to empty triangulation")
    if points_store is not None:
        points_store[pair_id] = {
            "pair_id": pair_id,
            "points": pts3D,
        }
    return pts3D, err_mean


def stage_align_pc(pose_results, points_results):
    print("\n=== STAGE 4: ALIGNING POINT CLOUDS ===")

    global_points = align_point_clouds(
        pose_results,
        points_results,
        output_name="global_points.npy",
        save=False,
    )
    return global_points


def stage_visualise(points_file):
    print("\n=== STAGE 5: VISUALISATION ===")
    visualize_points(points_file=str(points_file))


def is_good_keyframe(num_inliers, inlier_ratio, reproj_mean):
    if num_inliers < 80: return False
    #   if inlier_ratio < 0.4: return False
    if reproj_mean > 1.5: return False
    #   if num_3d_points < 25 : return False     # alr checking for in triangulation
    return True


def save_global_points(global_points):
    if global_points is None or global_points.size == 0:
        print("[PIPE] No global points to save.")
        print("\n[PIPE] Done processing all image pairs.")
    else:
        # Remove infinities / NaNs
        mask = np.isfinite(global_points).all(axis=1)
        pts = global_points[mask]

        # Option 1: save in project root
        # np.savetxt("global_points.xyz", pts)

        # Option 2 (nicer): save into outputs/aligning_pc
        out_path = PROJECT_ROOT / "outputs" / "aligning_pc" / "final_cloud.xyz"
        np.savetxt(out_path, pts)


def print_stats(name, arr, want_min=True):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        print(f"[STATS] {name}: no data")
        return
    mean = arr.mean()
    median = np.median(arr)
    minv = arr.min()
    maxv = arr.max()
    if want_min:
        print(f"[STATS] {name}: mean={mean:.3f}, median={median:.3f}, "
                f"min={minv:.3f}, max={maxv:.3f}")
    else:
        print(f"[STATS] {name}: mean={mean:.3f}, median={median:.3f}, "
                f"max={maxv:.3f}")


def histogram(rot_err_arr):
    rot_err_arr = np.array(rot_err_arr, dtype=float)

    # Crop small rotation errors (<10°)
    small = rot_err_arr[rot_err_arr < 0.8]

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(small, bins=20, edgecolor='black')
    plt.xlabel("Reprojection Error (degrees, cropped)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Reprojection Errors ")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print catastrophic failures
    num_big = np.sum(rot_err_arr >= 10)
    print(f"Number of catastrophic failures (≥10°): {num_big}")


def associate_rgb_depth(rgb_entries, depth_entries, max_diff=0.02):
    """
    rgb_entries:   list of (rgb_ts, rgb_path)
    depth_entries: list of (depth_ts, depth_path)

    Returns:
        frames = [
            {
                "rgb_ts": ...,
                "rgb_path": ...,
                "depth_ts": ...,
                "depth_path": ...
            },
            ...
        ]
    """
    frames = []
    j = 0

    for rgb_ts, rgb_path in rgb_entries:
        best_j = None
        best_diff = float("inf")

        while j < len(depth_entries) and depth_entries[j][0] < rgb_ts - max_diff:
            j += 1

        for k in [j - 1, j, j + 1]:
            if 0 <= k < len(depth_entries):
                depth_ts, depth_path = depth_entries[k]
                diff = abs(depth_ts - rgb_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_j = k

        if best_j is not None and best_diff <= max_diff:
            depth_ts, depth_path = depth_entries[best_j]
            frames.append({
                "rgb_ts": rgb_ts,
                "rgb_path": rgb_path,
                "depth_ts": depth_ts,
                "depth_path": depth_path,
            })

    print(f"[RGBD] Associated {len(frames)} RGB frames with depth")
    return frames


def load_rgbd_frames(rgb_txt_path, depth_txt_path, dataset_root, max_diff=0.02):
    rgb_entries = load_txt_entries(rgb_txt_path, dataset_root)
    depth_entries = load_txt_entries(depth_txt_path, dataset_root)
    frames = associate_rgb_depth(rgb_entries, depth_entries, max_diff=max_diff)
    print(f"[RGBD] Loaded {len(frames)} RGB-D frames")
    return frames


def save_rgbd_trajectory(global_poses, frames, output_file):
    """
    Save estimated trajectory in TUM format:
    timestamp tx ty tz qx qy qz qw

    Assumes global_poses stores T_cw.
    Converts each pose to T_wc before saving.
    """
    with open(output_file, "w") as f:
        for frame, T_cw in zip(frames[:len(global_poses)], global_poses):
            ts = frame["rgb_ts"]

            T_wc = np.linalg.inv(T_cw)
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]

            qx, qy, qz, qw = Rotation.from_matrix(R_wc).as_quat()
            f.write(f"{ts} {t_wc[0]} {t_wc[1]} {t_wc[2]} {qx} {qy} {qz} {qw}\n")


if __name__ == "__main__":

    timestamps, image_files = load_rgb_entries(RGB_TXT, RGB_DATA_DIR)
    tracking_results = {}
    global_poses = [np.eye(4)]
    frames = load_rgbd_frames(RGB_TXT, DEPTH_TXT, RGB_DATA_DIR)

    for i in range(len(frames) - 1):
        print("\n" + "="*200)
        print(f"\n[PIPE] Processing image pair {i} and {i + 1}...")

        f1 = frames[i]
        f2 = frames[i + 1]

        img1 = cv.imread(str(f1["rgb_path"]), cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(str(f2["rgb_path"]), cv.IMREAD_GRAYSCALE)

        depth1 = cv.imread(str(f1["depth_path"]), cv.IMREAD_UNCHANGED)
        depth2 = cv.imread(str(f2["depth_path"]), cv.IMREAD_UNCHANGED)

        ts1 = f1["rgb_ts"]
        ts2 = f2["rgb_ts"]

        depth1_m = depth_to_meters(depth1)
        depth2_m = depth_to_meters(depth2)

        tracking_entry = stage_tracking(
            str(f1["rgb_path"]),
            str(f2["rgb_path"]),
            pair_id=i,
            tracking_results=tracking_results,
        )

        pose_entry = stage_pose_rgbd(tracking_entry, depth1_m, K_RGBD)

        if pose_entry is None:
            continue

        R = pose_entry["R"]
        t = pose_entry["t"]
        num_inliers = pose_entry["num_inliers"]
        ratio = pose_entry["inlier_ratio"]

        if num_inliers < 30 or ratio < 0.7:
            print("[WARN] Weak RGB-D pose estimate, skipping pose accumulation")
            continue

        print("[RGBD] Relative pose:")
        print("R =\n", R)
        print("t =\n", t.ravel())

        # Check convention
        T_21 = pose_entry["T_21"]

        T_cw_prev = global_poses[-1]
        T_cw_new = T_21 @ T_cw_prev

        global_poses.append(T_cw_new)

        print(f"[PIPE] Pose count = {len(global_poses)}")
        print(f"[RGBD] PnP inliers: {num_inliers}")
        print(f"[RGBD] Inlier ratio: {ratio:.3f}")
        print(f"[RGBD] Translation norm: {np.linalg.norm(t):.3f}")

        rgb_to_gt_pose = build_rgb_to_gt_pose_map(GT_PATH, image_files, max_diff=0.02)
        R_gt, t_gt = relative_pose_from_rgb_ts(rgb_to_gt_pose, ts1, ts2)

        rot_err = rotation_error_deg(R, R_gt)
        dir_err = translation_direction_error_deg(t, t_gt)

        print(f"[RGBD][GT] Rotation error: {rot_err:.3f} deg")
        print(f"[RGBD][GT] Translation direction error: {dir_err:.3f} deg")

        out_file = PROJECT_ROOT / "outputs" / "tests" / "rgbd_vo_traj.txt"
        save_rgbd_trajectory(global_poses, frames, out_file)
        print(f"[PIPE] Saved trajectory to {out_file}")