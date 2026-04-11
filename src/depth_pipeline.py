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
    create_mappoints_from_triangulation,
    insert_keyframe_if_needed,
    kps_to_xy,
)
from pose_graph_optimization.pose_graph_optimization import optimise_pose_graph
from utils.trajectory_utils import save_estimated_trajectory
from utils.util_clean import (
    depth_to_meters,
    load_txt_entries,
    create_new_mappoints_from_depth_for_keyframe,
    initialize_map_rgbd,
    cull_weak_mappoints,
)
from depth.MiDaS_monocular import (
    load_midas,
    estimate_depth,
    midas_to_pseudo_depth,
)
from pose_estimation.PnP import run_pnp_for_frame
from tests.reprojection_err import reprojection_error

DATA_DIR = PROJECT_ROOT / "data" / "rgbd_dataset_room" / "rgb"
#   TEMP_DIR = PROJECT_ROOT / "outputs" / "temp"
RGB_DATA_DIR = PROJECT_ROOT / "data" / "rgbd_dataset_room"
RGB_TXT = PROJECT_ROOT / "data" / "rgbd_dataset_room" / "rgb.txt"
DEPTH_DIR = PROJECT_ROOT / "data" / "rgbd_dataset_room" / "depth"
DEPTH_TXT = PROJECT_ROOT / "data" / "rgbd_dataset_room" / "depth.txt"

out_pc = PROJECT_ROOT / "report_results" / "pipeline4" / "room" / "final_cloud.xyz"
out_file = PROJECT_ROOT / "report_results" / "pipeline4" / "room" / "tests.txt"
out_pics = PROJECT_ROOT / "report_results" / "pipeline4" / "room" / "pics"

f2 = False
if f2:
    K_RGBD = np.array([
        [520.9,   0.0, 325.1],
        [0.0, 521.0, 249.7],
        [0.0,   0.0,   1.0]
    ], dtype=np.float32)
else:
    K_RGBD = np.array([
        [517.3,   0.0, 318.6],
        [0.0, 516.5, 255.3],
        [0.0,   0.0,   1.0]
    ], dtype=np.float32)


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
    MATCHER = "bf"
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


def is_good_keyframe(
    slam_map,
    last_kf_id,
    T_cw_cur,
    ninliers,
    inlier_kp_to_mp,
    frame_idx,
    last_kf_frame_idx,
):
    return True, 'overriding for testing'
    last_kf = slam_map.keyframes[last_kf_id]

    # relative motion
    T_rel = T_cw_cur @ np.linalg.inv(last_kf.T_cw)
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]

    # rotation (deg)
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    rot_deg = np.degrees(np.arccos(trace))

    trans = np.linalg.norm(t_rel)
    tracked = len(inlier_kp_to_mp)
    frames_since = frame_idx - last_kf_frame_idx

    # ---- conditions ----

    if ninliers < 50:
        return False, "low_inliers"

    if tracked < 80:
        return True, "low_tracking"

    if rot_deg > 8:
        return True, f"rotation {rot_deg:.1f}"

    if trans > 0.08:
        return True, f"translation {trans:.2f}"

    if frames_since > 8:
        return True, "frame_gap"

    return False, "too_close"


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
    arr = arr[np.isfinite(arr)]
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


def save_quick_pointcloud_xyz(frames, global_poses, K, output_file, max_frames=590, stride=4):
    """
    Quick sanity-check point cloud from RGB-D frames.
    Uses depth images + estimated poses and saves an .xyz file.

    Assumes global_poses stores T_cw.
    """
    all_points = []

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    n = min(max_frames, len(global_poses), len(frames))

    for i in range(n):
        depth_raw = cv.imread(str(frames[i]["depth_path"]), cv.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue

        depth = depth_to_meters(depth_raw)
        T_cw = global_poses[i]

        h, w = depth.shape
        for v in range(0, h, stride):
            for u in range(0, w, stride):
                z = depth[v, u]
                if not np.isfinite(z) or z <= 0:
                    continue

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                p_cam = np.array([x, y, z, 1.0], dtype=np.float64)
                T_wc = np.linalg.inv(T_cw)
                p_world = (T_wc @ p_cam)[:3]
                all_points.append(p_world)

    if len(all_points) == 0:
        print("[PIPE] No valid points to save.")
        return

    all_points = np.asarray(all_points, dtype=np.float64)
    np.savetxt(output_file, all_points, fmt="%.6f %.6f %.6f")
    print(f"[PIPE] Saved quick point cloud to {output_file}")


def save_quick_pointcloud_xyzrgb(frames, global_poses, K, output_file, max_frames=600, stride=8):
    """
    Quick sanity-check coloured point cloud from RGB-D frames.
    Uses RGB + depth + estimated poses and saves an XYZRGB file.

    Assumes global_poses stores T_cw.
    Output columns: X Y Z R G B
    """
    all_points = []

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    n = min(max_frames, len(global_poses), len(frames))

    for i in range(n):
        depth_raw = cv.imread(str(frames[i]["depth_path"]), cv.IMREAD_UNCHANGED)
        rgb = cv.imread(str(frames[i]["rgb_path"]), cv.IMREAD_COLOR)

        if depth_raw is None or rgb is None:
            continue

        depth = depth_to_meters(depth_raw)
        T_cw = global_poses[i]
        T_wc = np.linalg.inv(T_cw)

        h, w = depth.shape
        for v in range(0, h, stride):
            for u in range(0, w, stride):
                z = depth[v, u]
                if not np.isfinite(z) or z <= 0:
                    continue

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                p_cam = np.array([x, y, z, 1.0], dtype=np.float64)
                p_world = (T_wc @ p_cam)[:3]

                # OpenCV loads colour as BGR
                b, g, r = rgb[v, u]

                all_points.append([p_world[0], p_world[1], p_world[2], int(r), int(g), int(b)])

    if len(all_points) == 0:
        print("[PIPE] No valid points to save.")
        return

    all_points = np.asarray(all_points, dtype=np.float64)
    np.savetxt(output_file, all_points, fmt="%.6f %.6f %.6f %d %d %d")
    print(f"[PIPE] Saved quick coloured point cloud to {output_file}")


def invert_pose(R_wc, t_wc):
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc
    return R_cw, t_cw


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def relative_transform(T_curr, T_prev):
    return T_curr @ np.linalg.inv(T_prev)


def build_Tcw_gt_map(rgb_to_gt_pose):
    gt_Tcw = {}
    for ts, (R_wc, t_wc) in rgb_to_gt_pose.items():
        R_cw, t_cw = invert_pose(R_wc, np.asarray(t_wc).reshape(3))
        gt_Tcw[ts] = make_T(R_cw, t_cw)
    return gt_Tcw


def cumulative_plot(errors, label):
    errors = np.array(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    errors = np.sort(errors)
    y = np.arange(1, len(errors) + 1) / len(errors)
    plt.plot(errors, y, label=label)
    plt.xlabel("Relative Rotation Error (deg)")
    plt.ylabel("Fraction of Frames")
    plt.title("PnP Map: Cumulative Rotation Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def histogram(rot_err_arr, bins=20, title="Histogram of Reprojection Errors"):
    rot_err_arr = np.array(rot_err_arr, dtype=float)

    if len(rot_err_arr) == 0:
        print("[ERROR] No data to plot in the histogram.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(rot_err_arr, bins=bins, edgecolor='black')
    plt.xlabel("Reprojection Error (degrees, cropped)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = out_pics / title
    plt.savefig(save_path)
    print(f"[PIPE] Histogram saved to {save_path}")

    plt.show()


if __name__ == "__main__":

    frames = load_rgbd_frames(RGB_TXT, DEPTH_TXT, RGB_DATA_DIR)

    image_files = [f["rgb_path"] for f in frames]
    rgb_to_gt_pose = build_rgb_to_gt_pose_map(GT_PATH, image_files, max_diff=0.02)
    gt_Tcw_map = build_Tcw_gt_map(rgb_to_gt_pose)

    slam_map = Map()
    is_initialized = False
    last_kf_id = None
    last_kf_frame_idx = None
    USE_MIDAS = False
    if USE_MIDAS:
        midas, midas_transform, midas_device = load_midas(model_type="MiDaS_small")

    depth_pnp_stats = []
    pnp_success = 0
    pnp_failed = 0
    pnp_weak = 0
    kf_inserted = 0
    kf_reused = 0

    prev_est_Tcw = None
    prev_gt_Tcw = None

    empty_stats = {
        "inlier_ratio": np.nan,
        "reprojection_error_px": np.nan,
        "relative_rotation_error_deg": np.nan,
        "relative_translation_error_m": np.nan,
        "relative_translation_direction_error_deg": np.nan,
    }

    for i in range(len(frames)):
        print("\n" + "=" * 120)
        print(f"[PIPE] Processing frame {i}")

        f1 = frames[i]
        ts_curr = f1["rgb_ts"]

        img1 = cv.imread(str(f1["rgb_path"]), cv.IMREAD_GRAYSCALE)

        if img1 is None:
            print("[PIPE] Missing RGB or depth image")
            depth_pnp_stats.append(empty_stats.copy())
            prev_est_Tcw = None
            prev_gt_Tcw = None
            continue

        if USE_MIDAS:
            depth_pred = estimate_depth(
                image_path=str(f1["rgb_path"]),
                midas=midas,
                transform=midas_transform,
                device=midas_device,
                save_vis=False,
            )
            depth1_m = midas_to_pseudo_depth(depth_pred)
        else:
            depth1 = cv.imread(str(f1["depth_path"]), cv.IMREAD_UNCHANGED)
            if depth1 is None:
                print("[PIPE] Missing RGB or depth image")
                depth_pnp_stats.append(empty_stats.copy())
                prev_est_Tcw = None
                prev_gt_Tcw = None
                continue
            depth1_m = depth_to_meters(depth1)

        if not is_initialized:
            try:
                last_kf_id = initialize_map_rgbd(
                    slam_map=slam_map,
                    frame_id=i,
                    img=img1,
                    depth_m=depth1_m,
                    K=K_RGBD,
                )
                is_initialized = True
                last_kf_frame_idx = i
            except Exception as e:
                print(f"[PIPE] RGB-D init failed: {e}")
            continue

        orb = cv.ORB_create(4000)
        keypoints, descriptors = orb.detectAndCompute(img1, None)

        if keypoints is None or descriptors is None or len(keypoints) == 0:
            print("[PIPE] No features in current frame")
            depth_pnp_stats.append(empty_stats.copy())
            prev_est_Tcw = None
            prev_gt_Tcw = None
            continue

        T_cw_cur, ninliers, inlier_kp_to_mp, stats = run_pnp_for_frame(
            slam_map=slam_map,
            last_kf_id=last_kf_id,
            keypoints=keypoints,
            descriptors=descriptors,
            K=slam_map.keyframes[last_kf_id].K,
            kf_window=5,
            min_observations=1,
        )

        if T_cw_cur is None or not stats:
            depth_pnp_stats.append(empty_stats.copy())
            print("[PIPE] PnP failed")
            pnp_failed += 1
            prev_est_Tcw = None
            prev_gt_Tcw = None
            continue

        pnp_success += 1
        print(f"[PIPE] PnP succeeded with {ninliers} inliers")

        cur_stats = {
            "inlier_ratio": stats.get("inlier_ratio", np.nan),
            "reprojection_error_px": stats.get("reprojection_error", np.nan),
            "relative_rotation_error_deg": np.nan,
            "relative_translation_error_m": np.nan,
            "relative_translation_direction_error_deg": np.nan,
        }

        if ts_curr in gt_Tcw_map:
            gt_Tcw_cur = gt_Tcw_map[ts_curr]

            if prev_est_Tcw is not None and prev_gt_Tcw is not None:
                T_rel_est = relative_transform(T_cw_cur, prev_est_Tcw)
                T_rel_gt = relative_transform(gt_Tcw_cur, prev_gt_Tcw)

                R_rel_est = T_rel_est[:3, :3]
                t_rel_est = T_rel_est[:3, 3]

                R_rel_gt = T_rel_gt[:3, :3]
                t_rel_gt = T_rel_gt[:3, 3]

                cur_stats["relative_rotation_error_deg"] = rotation_error_deg(R_rel_est, R_rel_gt)
                cur_stats["relative_translation_error_m"] = float(np.linalg.norm(t_rel_est - t_rel_gt))
                cur_stats["relative_translation_direction_error_deg"] = (
                    translation_direction_error_deg(t_rel_est, t_rel_gt)
                )

            prev_est_Tcw = T_cw_cur.copy()
            prev_gt_Tcw = gt_Tcw_cur.copy()
        else:
            print(f"[PIPE] No GT pose found for timestamp {ts_curr:.6f}")
            prev_est_Tcw = None
            prev_gt_Tcw = None

        depth_pnp_stats.append(cur_stats)

        if ninliers < 30:
            pnp_weak += 1
            print("[PIPE] PnP too weak for KF insertion")
            continue

        is_kf, reason = is_good_keyframe(
            slam_map=slam_map,
            last_kf_id=last_kf_id,
            T_cw_cur=T_cw_cur,
            ninliers=ninliers,
            inlier_kp_to_mp=inlier_kp_to_mp,
            frame_idx=i,
            last_kf_frame_idx=last_kf_frame_idx,
        )

        print(
            f"[KF CHECK] frame={i}, insert={is_kf}, reason={reason}, "
            f"ninliers={ninliers}, tracked_mps={len(inlier_kp_to_mp)}"
        )

        if not is_kf:
            kf_reused += 1
            continue

        kf_j_id = insert_keyframe_if_needed(
            slam_map=slam_map,
            frame_id=i,
            T_cw=T_cw_cur,
            K=slam_map.keyframes[last_kf_id].K,
            keypoints_xy=kps_to_xy(keypoints),
            descriptors=descriptors,
            tracking_acceptance={"tracking_acceptance:": 0},
        )

        if kf_j_id == last_kf_id:
            print("[PIPE] Same keyframe returned, skipping")
            kf_reused += 1
            continue

        kf_inserted += 1
        kf_j = slam_map.keyframes[kf_j_id]

        for kp_idx, mp_id in inlier_kp_to_mp.items():
            kf_j.kp_to_mp[kp_idx] = mp_id
            slam_map.mappoints[mp_id].observations[kf_j_id] = int(kp_idx)

        created = create_new_mappoints_from_depth_for_keyframe(
            slam_map=slam_map,
            kf_id=kf_j_id,
            depth_m=depth1_m,
            max_new_points=5000,
        )
        print(f"[PIPE] Added {created} new depth MapPoints in KF{kf_j_id}")

        last_kf = slam_map.keyframes[last_kf_id]
        T_rel = T_cw_cur @ np.linalg.inv(last_kf.T_cw)

        slam_map.add_edge(
            Edge(
                kf_i=last_kf_id,
                kf_j=kf_j_id,
                T_ij=T_rel,
                edge_type="odometry",
            )
        )

        if kf_j_id % 5 == 0:
            cull_weak_mappoints(slam_map, min_observations=2)

        last_kf_id = kf_j_id
        last_kf_frame_idx = i

    print_map(slam_map)

    save_estimated_trajectory(slam_map, [f["rgb_path"] for f in frames], out_file)

    global_poses = [kf.T_cw for kf in slam_map.keyframes.values()]
    save_quick_pointcloud_xyzrgb(
        frames=frames,
        global_poses=global_poses,
        K=K_RGBD,
        output_file=out_pc
    )

    print(f"[PIPE] Estimated trajectory saved to {out_file}")
    print(f"[MAP] Number of MapPoints: {len(slam_map.mappoints)}")
    print("PnP success:", pnp_success)
    print("PnP failed:", pnp_failed)
    print("PnP weak (<30 inliers):", pnp_weak)
    print("Keyframes inserted:", kf_inserted)
    print("Frames reused existing keyframe:", kf_reused)

    print_stats("Inlier ratio", [d["inlier_ratio"] for d in depth_pnp_stats])
    print_stats("Reprojection error (px)", [d["reprojection_error_px"] for d in depth_pnp_stats])
    print_stats("Relative rotation error (deg)", [d["relative_rotation_error_deg"] for d in depth_pnp_stats])
    print_stats("Relative translation error (m)", [d["relative_translation_error_m"] for d in depth_pnp_stats])
    print_stats(
        "Relative translation-direction error (deg)",
        [d["relative_translation_direction_error_deg"] for d in depth_pnp_stats]
    )

    trans_err_map = [d["relative_translation_error_m"] for d in depth_pnp_stats]
    rot_err_map = [d["relative_rotation_error_deg"] for d in depth_pnp_stats]
    reproj_err_map = [d["reprojection_error_px"] for d in depth_pnp_stats]

    plt.figure(figsize=(7,5))
    #   cumulative_plot(trans_err_map, "PnP Map")
    histogram(rot_err_map, bins=20, title="Map-Based Histogram of PnP Relative Rotation Errors Room")
    histogram(reproj_err_map, bins=20, title="Map-Based Histogram of PnP Reprojection Errors (px) Room")
    histogram(trans_err_map, bins=20, title="Map-Based Histogram of PnP Relative Translation Errors (m) Room")

