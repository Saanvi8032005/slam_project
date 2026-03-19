"""
pipeline.py

"""

from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2 as cv


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
    run_pnp_for_frame,
    create_mappoints_from_triangulation,
    insert_keyframe_if_needed,
    kps_to_xy,
)
from pose_graph_optimization.pose_graph_optimization import optimise_pose_graph
from utils.trajectory_utils import save_estimated_trajectory
from pose_estimation.PnP import run_pnp_for_frame

DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset" / "rgb"
TEMP_DIR = PROJECT_ROOT / "outputs" / "temp"


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


def stage_tracking(img1, img2, pair_id, tracking_results):
    """
    Stage 1: run feature detection + matching + filtering.
    Writes pts1, pts2 to an .npz file for pose_estimation to use.
    """
    MATCHER = "flann"
    FILTER = "hist"

    print(f"\n=== STAGE 1: TRACKING / MATCHING ({pair_id}) ===")

    out_file = f"matches_{pair_id}.npz"
    pts1, pts2, kp1, kp2, matches, idx_i, idx_j, des1, des2 = matching(
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
        "idx_i": idx_i,
        "idx_j": idx_j,
        "des1": des1,
        "des2": des2,
    }
    tracking_results[pair_id] = entry

    return entry


def stage_pose(tracking_entry, pose_store=None, img1=None, img2=None):
    pair_id = tracking_entry["pair_id"]
    pts1 = tracking_entry["pts1"]
    pts2 = tracking_entry["pts2"]
    idx_i = tracking_entry["idx_i"]
    idx_j = tracking_entry["idx_j"]

    print(f"\n=== STAGE 2: POSE ESTIMATION ({pair_id}) ===")
    result = pose_estimate(pts1, pts2, idx_i, idx_j)

    # Handle skipped pairs
    if result[0] is None:
        print(f"[PIPE] Skipping pair {pair_id} due to low parallax or pose estimation failure")
        return None, None, None

    R, t, K, num_inliers, ratio, pts1, pts2, idx_i_inl, idx_j_inl = result

    if pose_store is not None:
        pose_store[pair_id] = {
            "pair_id": pair_id,
            "R": R,
            "t": t,
            "K": K,
            "pts1": pts1,
            "pts2": pts2,
            "num_inliers": num_inliers,
            "inlier_ratio": ratio,
            "idx_i_inl": idx_i_inl,
            'idx_j_inl': idx_j_inl,
        }

    error_print = True
    if error_print:
        ts1 = float(img1.stem)
        ts2 = float(img2.stem)
        rgb_to_gt_pose = build_rgb_to_gt_pose_map(
                            GT_PATH,
                            image_files,
                            max_diff=0.02
                        )
        R_gt, t_gt = relative_pose_from_rgb_ts(rgb_to_gt_pose, ts1, ts2)

        rot_err = rotation_error_deg(R, R_gt)
        dir_err = translation_direction_error_deg(t, t_gt)
        print(f"[POSE][GT] Rotation error:            {rot_err:.3f} deg")
        print(f"[POSE][GT] Translation direction err: {dir_err:.3f} deg")
    return pose_store[pair_id], rot_err, dir_err


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
    pts3D, err_mean, triang_filter = triangulate_from_data(
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
            "triang_filter": triang_filter,
            "err_mean": err_mean,
            "pts3D": pts3D,
        }
    return points_store[pair_id]


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
    if inlier_ratio < 0.4: return False
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


if __name__ == "__main__":

    image_files = load_image_files()
    tracking_results = {}
    pose_results = {}
    points_results = {}
    tracking_acceptance = 0

    slam_map = Map()
    is_initialized = False
    init_kf0_id = None
    init_kf1_id = None
    last_kf_id = None

    for i in range(10 - 1):
        print("\n" + "="*200)
        print(f"\n[PIPE] Processing image pair {i} and {i + 1}...")
        print(f"[PIPE] Image 1: {image_files[i]}")
        print(f"[PIPE] Image 2: {image_files[i + 1]}")

        pair_id = f"{i:03d}"
        img1 = image_files[i]
        img2 = image_files[i + 1]

        if not is_initialized:
            tracking_entry = stage_tracking(
                img1,
                img2,
                pair_id,
                tracking_results)
            pose_entry, rot_err, dir_err = stage_pose(tracking_entry, pose_results, img1, img2)
            if pose_entry is None:
                print(f"[PIPE] Skipping pair {pair_id} due to pose estimation failure")
                continue
            points_entry = stage_triangulate(tracking_entry,
                                                pose_entry,
                                                points_store=points_results)
            pts3D = points_entry["pts3D"]
            triang_filter = points_entry["triang_filter"]

            if is_good_keyframe(
                pose_entry['num_inliers'],
                pose_entry['inlier_ratio'],
                points_entry['err_mean'],
            ) and pts3D.shape[0] > 0:
                tracking_acceptance += 1
                init_kf0_id, init_kf1_id = initialize_map(
                    slam_map=slam_map,
                    frame_id0=i,
                    frame_id1=i + 1,
                    K=pose_entry["K"],
                    R=pose_entry["R"],
                    t=pose_entry["t"],
                    kp1=tracking_entry["kp1"],
                    kp2=tracking_entry["kp2"],
                    des1=tracking_entry.get("des1"),
                    des2=tracking_entry.get("des2"),
                )
                is_initialized = True
                last_kf_id = init_kf1_id
                print(f"[PIPE] Initialized map with keyframes {init_kf0_id} and {init_kf1_id}")
                create_mappoints_from_triangulation(
                    slam_map,
                    kf_i_id=init_kf0_id,
                    kf_j_id=init_kf1_id,
                    pts3D=pts3D,
                    idx_i=pose_entry["idx_i_inl"][triang_filter],
                    idx_j=pose_entry["idx_j_inl"][triang_filter],
                )                  
        else:
            tracking_entry = stage_tracking(
                img1,
                img2,
                pair_id,
                tracking_results
            )

        T_cw_cur, ninliers, inlier_kp_to_mp = run_pnp_for_frame(
            slam_map=slam_map,
            last_kf_id=last_kf_id,
            keypoints=tracking_entry["kp2"],
            descriptors=tracking_entry["des2"],
            K=slam_map.keyframes[last_kf_id].K,
        )

        if T_cw_cur is None:
            print("[PIPE] PnP failed")
            continue
        if ninliers < 30:
            print("[PIPE] PnP too weak for KF insertion")
            continue

        print("[PIPE] PnP pose estimated")
        print(T_cw_cur)

        kf_j_id = insert_keyframe_if_needed(
            slam_map=slam_map,
            frame_id=i + 1,
            T_cw=T_cw_cur,
            K=slam_map.keyframes[last_kf_id].K,
            keypoints_xy=kps_to_xy(tracking_entry["kp2"]),
            descriptors=tracking_entry["des2"],
        )
        kf_j = slam_map.keyframes[kf_j_id]
        for kp_idx, mp_id in inlier_kp_to_mp.items():
            kf_j.kp_to_mp[kp_idx] = mp_id
            slam_map.mappoints[mp_id].observations[kf_j_id] = int(kp_idx)

        last_kf = slam_map.keyframes[last_kf_id]
        Tcw_last = last_kf.T_cw
        T_rel = T_cw_cur @ np.linalg.inv(Tcw_last)

        slam_map.add_edge(
            Edge(
                kf_i=last_kf_id,
                kf_j=kf_j_id,
                T_ij=T_rel,
                edge_type="odometry",
            )
        )
        #   matches = match(last_kf.descriptors, kf_j.descriptors, method="flann")
