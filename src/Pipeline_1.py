from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tracking.tracking import matching
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
from keyframe_selection.keyframe_selec import Map, print_map,
from keyframe_selection.keyframe_helpers import (
    initialize_map,  # Fixed function name
    add_map_edge,
)
from utils.trajectory_utils import save_estimated_trajectory

DATA_DIR = PROJECT_ROOT / "data" / "rgbd_dataset_room" / "rgb"
RBG_DATA_DIR = PROJECT_ROOT / "data" / "rgbd_dataset_room"
RGB_TXT = PROJECT_ROOT / "data" / "rgbd_dataset_room" / "rgb.txt"

estimated_trajectory_file = PROJECT_ROOT / "report_results" / "pipeline1" / "room" / "tests.txt"
out_path = PROJECT_ROOT / "report_results" / "pipeline1" / "room" / "final_cloud.xyz"


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


def stage_tracking(img1, img2, pair_id, tracking_results):
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
        ts2=None,
        missing_pair=0,
    ):
    pair_id = tracking_entry["pair_id"]
    pts1 = tracking_entry["pts1"]
    pts2 = tracking_entry["pts2"]

    print(f"\n=== STAGE 2: POSE ESTIMATION ({pair_id}) ===")
    result = pose_estimate(pts1, pts2)

    # Handle skipped pairs
    if result[0] is None:
        print(f"[PIPE] Skipping pair {pair_id} due to low parallax or pose estimation failure")
        return None, None, None, 0, 0.0, None, None, None, None, missing_pair

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
        R_gt, t_gt = None, None
        try:
            R_gt, t_gt = relative_pose_from_rgb_ts(rgb_to_gt_pose, ts1, ts2)
        except KeyError as e:
            print(f"[WARNING] {e}")
            missing_pair += 1

        if R_gt is not None and t_gt is not None:
            rot_err = rotation_error_deg(R, R_gt)
            dir_err = translation_direction_error_deg(t, t_gt)
            print(f"[POSE][GT] Rotation error:            {rot_err:.3f} deg")
            print(f"[POSE][GT] Translation direction err: {dir_err:.3f} deg")
        else:
            rot_err = None
            dir_err = None
            print(f"[POSE][GT] Ground truth pose not found for pair {pair_id}")

    # should just return pose_store tbh
    return R, t, K, num_inliers, ratio, rot_err, dir_err, pts1, pts2, missing_pair


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


if __name__ == "__main__":

    timestamps, image_files = load_rgb_entries(RGB_TXT, RBG_DATA_DIR)
    tracking_results = {}
    pose_results = {}
    points_results = {}
    tracking_acceptance = 0
    ransac_ratios = []          # for report
    rot_errors = []             # for report
    trans_dir_errors = []       # for report
    tri_counts = []             # for report
    tri_errors = []             # for report
    match_counts = []           # for report
    missing_pair = 0

    slam_map = Map()
    is_initialized = False
    init_kf0_id = None
    init_kf1_id = None
    last_kf_id = None

    for i in range(len(image_files) - 1):    # len(image_files) - 1
        print("\n" + "="*200)
        print(f"\n[PIPE] Processing image pair {i} and {i + 1}...")
        print(f"[PIPE] Image 1: {image_files[i]}")
        print(f"[PIPE] Image 2: {image_files[i + 1]}")

        pair_id = f"{i:03d}"
        img1 = image_files[i]
        img2 = image_files[i + 1]

        tracking_entry = stage_tracking(img1, img2, pair_id, tracking_results)
        R, t, K, num_inliers, inlier_ratio, rot_err, dir_err, _, _ , missing_pair= stage_pose(
            tracking_entry,
            pose_results,
            img1,
            img2,
            ts1=timestamps[i],
            ts2=timestamps[i + 1],
            missing_pair=missing_pair,
        )
        matches_len = len(tracking_entry["matches"])
        match_counts.append(matches_len)

        if R is None:  # Skip pairs with low parallax
            continue

        pose_entry = pose_results[pair_id]

        ransac_ratios.append(inlier_ratio)  # for report
        if rot_err is not None:
            rot_errors.append(rot_err)
        if dir_err is not None:
            trans_dir_errors.append(dir_err)

        pts3D, err_mean = stage_triangulate(tracking_entry,
                                            pose_entry,
                                            points_store=points_results)
        tri_counts.append(pts3D.shape[0])
        tri_errors.append(err_mean)

        if is_good_keyframe(num_inliers, inlier_ratio, err_mean) and pts3D.shape[0] > 0:
            print(f"[KF] Accepting keyframe {pair_id}")
            tracking_acceptance += 1
            if not is_initialized:
                # Initialize map with the first good pair
                init_kf0_id, init_kf1_id = initialize_map(
                    slam_map=slam_map,
                    frame_id0=i,
                    frame_id1=i + 1,
                    K=K,
                    R=R,
                    t=t,
                    kp1=tracking_entry["kp1"],
                    kp2=tracking_entry["kp2"],
                    des1=tracking_entry.get("des1", None),
                    des2=tracking_entry.get("des2", None),
                )
                is_initialized = True
                last_kf_id = init_kf1_id
            else:
                # Add odometry edge and insert KF_{i+1} with chained pose
                kf_j_id = add_map_edge(
                    slam_map=slam_map,
                    kf_i_id=last_kf_id,
                    frame_j=i + 1,
                    R=R,
                    t=t,
                    kp_j=tracking_entry["kp2"],
                    des_j=tracking_entry.get("des2", None),
                    K=K,
                )
                if kf_j_id is not None:
                    last_kf_id = kf_j_id
                """
                ransac_ratios.append(inlier_ratio)  # for report
                if rot_err is not None:
                    rot_errors.append(rot_err)
                if dir_err is not None:
                    trans_dir_errors.append(dir_err)
                """
        else:
            print(f"[KF] Rejecting keyframe {pair_id} (low quality)")
            pose_results.pop(pair_id, None)
            points_results.pop(pair_id, None)

    print(tracking_acceptance, "keyframes accepted out of", len(tracking_results))

    #   print_map(slam_map)
    optimise_pose_graph(slam_map, max_nfev=50, robust=True, verbose=2)  # Set verbosity level
    print_map(slam_map)

    global_points = stage_align_pc(pose_results, points_results)
    if False:
        visualize_points(points_file="global_points.npy")
    else:
        save_global_points(global_points)
    print("\n[PIPE] Done processing all image pairs.")

    # Save the estimated trajectory
    save_estimated_trajectory(slam_map, image_files, estimated_trajectory_file)

    print("\n================ GLOBAL POSE STATS ================\n")
    print_stats("RANSAC inlier ratio", ransac_ratios, want_min=True)
    print_stats("Rotation error [deg]", rot_errors, want_min=False)
    print_stats("Translation-direction error [deg]", trans_dir_errors, want_min=False)
    print("\n===================================================\n")

    histogram(tri_errors)
    print("\n================ TRIANGULATION STATS ==============\n")
    print_stats("Num 3D points per pair", tri_counts, want_min=True)
    print_stats("Mean reprojection error [px]", tri_errors, want_min=True)
    print("\n===================================================\n")

    print("Accepted keyframe frame_ids:")
    for kf_id in sorted(slam_map.keyframes.keys()):
        print(slam_map.keyframes[kf_id].frame_id, end=" ")
    print()

    results = {
        "matcher": "MATCHER",
        "filter": "FILTER",

        "mean_inlier_ratio": np.mean(ransac_ratios),
        "median_inlier_ratio": np.median(ransac_ratios),

        "mean_rot_error": np.mean(rot_errors),
        "median_rot_error": np.median(rot_errors),

        "mean_trans_error": np.mean(trans_dir_errors),
        "median_trans_error": np.median(trans_dir_errors),

        "mean_reproj_error": np.mean(tri_errors),
        "median_reproj_error": np.median(tri_errors),

        "mean_3d_points": np.mean(tri_counts),
        "median_3d_points": np.median(tri_counts),

        "accepted_keyframes": tracking_acceptance,
        "mean_matches": np.mean(match_counts),
    }
    print(results)
    print(f"No. of missing pairs (used for GT error analysis): {missing_pair}")