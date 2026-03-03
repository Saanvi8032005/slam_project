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
    run_pnp_for_frame,
    create_mappoints_from_triangulation,
    run_pnp_for_frame2
)
from pose_graph_optimization.pose_graph_optimization import optimise_pose_graph
from utils.trajectory_utils import save_estimated_trajectory

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
    pts1, pts2, kp1, kp2, matches, des1, des2 = matching(
             matcher=MATCHER,
             filter_method=FILTER,
             img1_path=img1,
             img2_path=img2,
             save_npz=False,
             unit_test=False,
             return_data=True,
             out_name=out_file,
             )

    pts1_all = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    pts2_all = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
    idx_i = np.array([m.queryIdx for m in matches], dtype=int)
    idx_j = np.array([m.trainIdx for m in matches], dtype=int)
    entry = {
        "pair_id": pair_id,
        "img1": img1,
        "img2": img2,
        "pts1": pts1_all,
        "pts2": pts2_all,
        "kp1": kp1,
        "kp2": kp2,
        "matches": matches,
        "des1": des1,
        "des2": des2,
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    tracking_results[pair_id] = entry

    return entry


def stage_pose(tracking_entry, pose_store=None, img1=None, img2=None):
    pair_id = tracking_entry["pair_id"]
    pts1 = tracking_entry["pts1"]
    pts2 = tracking_entry["pts2"]

    print(f"\n=== STAGE 2: POSE ESTIMATION ({pair_id}) ===")
    result = pose_estimate(pts1, pts2)

    # Handle skipped pairs
    if result[0] is None:
        print(f"[PIPE] Skipping pair {pair_id} due to low parallax or pose estimation failure")
        return None, None, None, 0, 0.0, None, None, None, None, None

    R, t, K, num_inliers, ratio, pts1, pts2, mask = result

    if pose_store is not None:
        pose_store[pair_id] = {
            "pair_id": pair_id,
            "R": R,
            "t": t,
            "K": K,
            "pts1": pts1,
            "pts2": pts2,
            "mask": mask,
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

        # Errors
        rot_err = rotation_error_deg(R, R_gt)
        dir_err = translation_direction_error_deg(t, t_gt)
        print(f"[POSE][GT] Rotation error:            {rot_err:.3f} deg")
        print(f"[POSE][GT] Translation direction err: {dir_err:.3f} deg")

    # should just return pose_store tbh
    return R, t, K, num_inliers, ratio, rot_err, dir_err, pts1, pts2, mask


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
    pts3D, err_mean = triangulate_from_data(
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
        out_path = PROJECT_ROOT / "outputs" / "aligning_pc" / "global_points_pose.xyz"
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
    ransac_ratios = []          # for report
    rot_errors = []             # for report
    trans_dir_errors = []       # for report
    tri_counts = []             # for report
    tri_errors = []             # for report

    slam_map = Map()
    is_initialized = False
    init_kf0_id = None
    init_kf1_id = None
    last_kf_id = None
    fail_count = 0

    for i in range(20 - 1):    # len(image_files) - 1
        print("\n" + "="*200)
        print(f"\n[PIPE] Processing image pair {i} and {i + 1}...")
        print(f"[PIPE] Image 1: {image_files[i]}")
        print(f"[PIPE] Image 2: {image_files[i + 1]}")

        pair_id = f"{i:03d}"
        img1 = image_files[i]
        img2 = image_files[i + 1]

        tracking_entry = stage_tracking(img1, img2, pair_id, tracking_results)

        if not is_initialized:
            # --- essential init just to bootstrap map ---
            R, t, K, num_inliers, inlier_ratio, rot_err, dir_err, pts1_f, pts2_f, inlier_mask = stage_pose(
                tracking_entry, pose_results, img1, img2
            )
            if R is None:
                continue

            # Decide if good init pair
            if num_inliers < 80 or inlier_ratio < 0.4:
                print("[INIT] Not enough inliers for init")
                continue

            # --- Align inlier_mask with idx_i and idx_j ---
            idx_i = tracking_entry["idx_i"]
            idx_j = tracking_entry["idx_j"]

            # Ensure inlier_mask is aligned with idx_i and idx_j
            inlier_mask = inlier_mask[:len(idx_i)]

            # Filter idx_i and idx_j using inlier_mask
            idx_i_inl = idx_i[inlier_mask]
            idx_j_inl = idx_j[inlier_mask]

            # --- Use inlier pixel coords directly (aligned to same mask) ---
            pts1_inl = tracking_entry["pts1"][inlier_mask]
            pts2_inl = tracking_entry["pts2"][inlier_mask]

            # --- Triangulate ONCE (from inliers only) ---
            pts3D, reproj_mean = triangulate_from_data(
                pts1_inl, pts2_inl, R, t, K, mask=None, out_name=None
            )

            if pts3D is None or pts3D.shape[0] == 0:
                print("[INIT] Triangulation empty, skip init")
                continue

            if reproj_mean > 1.5:
                print(f"[INIT] Reproj too high ({reproj_mean:.2f}), skip init")
                continue

            # --- Insert the two initial keyframes (poses chained from R,t) ---
            init_kf0_id, init_kf1_id = initialize_map(
                slam_map=slam_map,
                frame_id0=i,
                frame_id1=i + 1,
                K=K,
                R=R,
                t=t,
                kp1=tracking_entry["kp1"],
                kp2=tracking_entry["kp2"],
                des1=tracking_entry["des1"],
                des2=tracking_entry["des2"],
            )

            # --- Seed MapPoints from triangulation ---
            if pts3D.shape[0] != idx_i_inl.shape[0]:
                print("[INIT][WARN] Triangulation output count != inlier count.")
                print("-> Update triangulate_from_data to also return an 'active' mask.")
                n = min(pts3D.shape[0], idx_i_inl.shape[0])
                pts3D = pts3D[:n]
                idx_i_inl = idx_i_inl[:n]
                idx_j_inl = idx_j_inl[:n]

            created = create_mappoints_from_triangulation(
                slam_map=slam_map,
                kf_i_id=init_kf0_id,
                kf_j_id=init_kf1_id,
                pts3D=pts3D,
                idx_i=idx_i_inl,
                idx_j=idx_j_inl,
            )

            print(f"[INIT] Seeded {created} MapPoints (reproj_mean={reproj_mean:.3f})")
            print(f"[INIT] MapPoints after init: {len(slam_map.mappoints)}")

            is_initialized = True
            last_kf_id = init_kf1_id
            tracking_acceptance += 1
            continue

        # ============================================================
        # (B) TRACKING MODE: PnP instead of Essential
        # ============================================================
        # 1) Absolute pose from PnP (world->camera)
        Tcw_cur, ninl, inlier_kp_to_mp = run_pnp_for_frame2(
            slam_map=slam_map,
            keypoints=tracking_entry["kp2"],
            descriptors=tracking_entry["des2"],
            K=K
        )

        if Tcw_cur is None:
            print("[TRACK][PnP] Failed (not enough inliers / correspondences)")
            # optional fallback to stage_pose() if wanted
            fail_count += 1
            continue
        print(f"[TRACK][PnP] Inliers: {ninl}")

        # 2) Keyframe decision (simple version)
        #    can make this more sophisticated later (motion, time, parallax)
        if ninl < 30:
            # Track-only: pose exists but we don't insert a KF
            continue

        # 3) Convert absolute pose into relative edge from last keyframe
        last_kf = slam_map.keyframes[last_kf_id]
        Tcw_last = last_kf.T_cw

        T_rel = Tcw_cur @ np.linalg.inv(Tcw_last)  # cam_last -> cam_cur
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        # 4) Insert keyframe + edge (uses your existing helper)
        kf_j_id = add_map_edge(
            slam_map=slam_map,
            kf_i_id=last_kf_id,
            frame_j=i + 1,
            R=R_rel,
            t=t_rel,
            kp_j=tracking_entry["kp2"],
            des_j=tracking_entry["des2"],
            K=K,
        )
        if kf_j_id is None:
            continue

        # 5) Update kp_to_mp for the new KF using PnP inlier associations
        kf_j = slam_map.keyframes[kf_j_id]
        for kp_idx, mp_id in inlier_kp_to_mp.items():
            kf_j.kp_to_mp[kp_idx] = mp_id
            slam_map.mappoints[mp_id].observations[kf_j_id] = int(kp_idx)

        # 6) TRIANGULATE NEW POINTS between last_kf and new_kf to grow map
        #    Use matches from last_kf descriptors to current descriptors
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        kf_matches = bf.match(last_kf.descriptors, kf_j.descriptors)

        idx_i = np.array([m.queryIdx for m in kf_matches], dtype=int)
        idx_j = np.array([m.trainIdx for m in kf_matches], dtype=int)

        pts1 = np.array([last_kf.keypoints_xy[q] for q in idx_i], dtype=np.float32)
        pts2 = np.array([kf_j.keypoints_xy[t_] for t_ in idx_j], dtype=np.float32)

        # Relative pose for triangulation should be last->current
        pts3D_new, reproj_mean = triangulate_from_data(
            pts1, pts2, R_rel, t_rel, K, mask=None, out_name=None
        )

        if pts3D_new is not None and pts3D_new.shape[0] > 0:
            created = create_mappoints_from_triangulation(
                slam_map=slam_map,
                kf_i_id=last_kf_id,
                kf_j_id=kf_j_id,
                pts3D=pts3D_new,
                idx_i=idx_i,
                idx_j=idx_j,
            )
            print(f"[MAP] Added {created} MapPoints (map now {len(slam_map.mappoints)})")

        # 7) Advance last keyframe pointer
        last_kf_id = kf_j_id




    print(tracking_acceptance, "keyframes accepted out of", len(tracking_results))
    print(fail_count, "pairs failed in PnP tracking stage")

    global_points = stage_align_pc(pose_results, points_results)
    if False:
        visualize_points(points_file="global_points.npy")
    else:
        save_global_points(global_points)
    print("\n[PIPE] Done processing all image pairs.")

    #   print_map(slam_map)
    optimise_pose_graph(slam_map, max_nfev=50, robust=True, verbose=2)  # Set verbosity level
    print_map(slam_map)

    # Save the estimated trajectory
    estimated_trajectory_file = PROJECT_ROOT / "outputs" / "tests" / "tests.txt"
    save_estimated_trajectory(slam_map, image_files, estimated_trajectory_file)

    ransac_ratios.append(inlier_ratio)  # for report
    if rot_err is not None:
        rot_errors.append(rot_err)
    if dir_err is not None:
        trans_dir_errors.append(dir_err)

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
