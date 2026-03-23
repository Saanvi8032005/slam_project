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
        out_path = PROJECT_ROOT / "outputs" / "aligning_pc" / "post_pnp_cloud.xyz"
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


def geometric_filter_matches(kf_i, kf_j, matches, K):
    """
    Use Essential matrix RANSAC to keep only geometrically consistent matches.

    Returns:
        pts1_inl, pts2_inl, idx_i_inl, idx_j_inl
    """
    if len(matches) < 8:
        return None, None, None, None

    idx_i = np.array([m.queryIdx for m in matches], dtype=int)
    idx_j = np.array([m.trainIdx for m in matches], dtype=int)

    pts1 = np.array([kf_i.keypoints_xy[q] for q in idx_i], dtype=np.float32)
    pts2 = np.array([kf_j.keypoints_xy[t] for t in idx_j], dtype=np.float32)

    E, maskE = cv.findEssentialMat(
        pts1, pts2, K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None or maskE is None:
        return None, None, None, None

    maskE = maskE.ravel().astype(bool)

    pts1_inl = pts1[maskE]
    pts2_inl = pts2[maskE]
    idx_i_inl = idx_i[maskE]
    idx_j_inl = idx_j[maskE]

    return pts1_inl, pts2_inl, idx_i_inl, idx_j_inl


def filter_new_feature_matches(kf_i, kf_j, matches):
    """
    Keep only matches where neither keypoint is already assigned to a MapPoint.
    """
    filtered = []

    for m in matches:
        kp_i = m.queryIdx
        kp_j = m.trainIdx

        # Skip if this keypoint already has a MapPoint
        if kf_i.kp_to_mp[kp_i] is not None:
            continue
        if kf_j.kp_to_mp[kp_j] is not None:
            continue

        filtered.append(m)

    return filtered


def match_keyframes_for_triangulation(kf_i, kf_j, ratio=0.75):
    """
    Match descriptors between two keyframes using KNN + ratio test.

    Returns:
        good_matches: list[cv.DMatch]
    """
    if kf_i.descriptors is None or kf_j.descriptors is None:
        return []

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(kf_i.descriptors, kf_j.descriptors, k=2)

    good_matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def _skew(v):
    x, y, z = v.reshape(3)
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ], dtype=float)


def _triangulate_points_pose_consistent(pts1, pts2, R, t, K):
    """
    Triangulate points using the provided relative pose:
        cam1: P1 = K [I | 0]
        cam2: P2 = K [R | t]

    Returns:
        pts3D : (N, 3)
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    pts4D = cv.triangulatePoints(
        P1, P2,
        pts1.T.astype(np.float64),
        pts2.T.astype(np.float64)
    )
    pts3D = (pts4D[:3] / pts4D[3]).T
    return pts3D


def _compute_depths_in_both_cameras(pts3D, R, t):
    """
    cam1 coordinates: X1 = X
    cam2 coordinates: X2 = R X + t
    """
    z1 = pts3D[:, 2]
    X2 = (R @ pts3D.T + t.reshape(3, 1)).T
    z2 = X2[:, 2]
    return z1, z2


def _reproject_points(pts3D, K, R=None, t=None):
    """
    Reproject 3D points into a camera.
    If R,t are None -> first camera [I|0]
    Else -> second camera [R|t]
    """
    if R is None:
        Xc = pts3D
    else:
        Xc = (R @ pts3D.T + t.reshape(3, 1)).T

    proj = (K @ Xc.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv


def _parallax_angles_deg(pts3D, R, t):
    """
    Compute parallax angle between the two viewing rays.

    Camera 1 center: C1 = [0,0,0]
    Camera 2 center in cam1 frame: C2 = -R^T t
    """
    C1 = np.zeros(3)
    C2 = -R.T @ t.reshape(3)

    v1 = pts3D - C1
    v2 = pts3D - C2

    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)

    valid = (n1[:, 0] > 1e-12) & (n2[:, 0] > 1e-12)

    cosang = np.full(len(pts3D), np.nan, dtype=float)
    cosang[valid] = np.sum(v1[valid] * v2[valid], axis=1) / (
        n1[valid, 0] * n2[valid, 0]
    )
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return ang


def triangulate_new_features_between_keyframes(
    slam_map,
    kf_i_id,
    kf_j_id,
    R_rel,
    t_rel,
    K,
    ratio=0.75,
    max_reproj_err=2.0,
    min_parallax_deg=1.0,
    min_num_created=8,
    max_new_points=1000,  # Cap new MapPoints per pair
):
    """
    Triangulate new MapPoints between two keyframes using the KNOWN relative pose.

    Pipeline:
      1. Match descriptors
      2. Remove keypoints already assigned to MapPoints
      3. Triangulate with provided R_rel, t_rel
      4. Keep only points that satisfy:
           - finite coordinates
           - positive depth in both cameras
           - low reprojection error in both images
           - sufficient parallax
      5. Create new MapPoints (capped to max_new_points)

    Returns:
        created : int
    """
    kf_i = slam_map.keyframes[kf_i_id]
    kf_j = slam_map.keyframes[kf_j_id]

    # 1. descriptor matching
    matches = match_keyframes_for_triangulation(kf_i, kf_j, ratio=ratio)
    print(f"[MAP] KF match candidates: {len(matches)}")
    if len(matches) < 8:
        print("[MAP] Too few descriptor matches")
        return 0

    # 2. keep only not-yet-mapped features
    matches = filter_new_feature_matches(kf_i, kf_j, matches)
    print(f"[MAP] Unmapped feature matches: {len(matches)}")
    if len(matches) < 8:
        print("[MAP] Too few unmapped matches")
        return 0

    idx_i = np.array([m.queryIdx for m in matches], dtype=int)
    idx_j = np.array([m.trainIdx for m in matches], dtype=int)

    pts1 = np.array([kf_i.keypoints_xy[q] for q in idx_i], dtype=np.float64)
    pts2 = np.array([kf_j.keypoints_xy[t] for t in idx_j], dtype=np.float64)

    flow = np.linalg.norm(pts2 - pts1, axis=1)
    median_flow = np.median(flow)
    print(f"[MAP] Median pixel flow: {median_flow:.2f}px")

    if median_flow < 5.0:
        print("[MAP] Too little image motion for reliable triangulation")
        return 0

    # 3. triangulate directly with your known pose
    pts3D = _triangulate_points_pose_consistent(pts1, pts2, R_rel, t_rel, K)

    if pts3D is None or len(pts3D) == 0:
        print("[MAP] No points triangulated")
        return 0

    # 4a. finite check
    finite_mask = np.isfinite(pts3D).all(axis=1)

    # 4b. cheirality / positive depth
    z1, z2 = _compute_depths_in_both_cameras(pts3D, R_rel, t_rel)
    depth_mask = (z1 > 1e-6) & (z2 > 1e-6)

    # 4c. reprojection error in both views
    uv1 = _reproject_points(pts3D, K)
    uv2 = _reproject_points(pts3D, K, R_rel, t_rel)

    err1 = np.linalg.norm(uv1 - pts1, axis=1)
    err2 = np.linalg.norm(uv2 - pts2, axis=1)
    reproj_mask = (err1 < max_reproj_err) & (err2 < max_reproj_err)

    # 4d. parallax
    parallax_deg = _parallax_angles_deg(pts3D, R_rel, t_rel)
    parallax_mask = parallax_deg > min_parallax_deg

    keep_mask = finite_mask & depth_mask & reproj_mask & parallax_mask

    n_keep = int(np.sum(keep_mask))
    print(f"[MAP] finite:      {np.sum(finite_mask)} / {len(matches)}")
    print(f"[MAP] positive z:  {np.sum(depth_mask)} / {len(matches)}")
    print(f"[MAP] reproj ok:   {np.sum(reproj_mask)} / {len(matches)}")
    print(f"[MAP] parallax ok: {np.sum(parallax_mask)} / {len(matches)}")
    print(f"[MAP] final kept:  {n_keep} / {len(matches)}")

    if n_keep < min_num_created:
        print("[MAP] Too few valid triangulated points after filtering")
        return 0

    if n_keep > max_new_points:
        keep_indices = np.where(keep_mask)[0]
        # score: smaller reprojection error is better
        total_err = err1 + err2
        ranked = keep_indices[np.argsort(total_err[keep_indices])]
        keep_mask[:] = False
        keep_mask[ranked[:max_new_points]] = True

    pts3D_keep = pts3D[keep_mask]
    idx_i_keep = idx_i[keep_mask]
    idx_j_keep = idx_j[keep_mask]

    created = create_mappoints_from_triangulation(
        slam_map=slam_map,
        kf_i_id=kf_i_id,
        kf_j_id=kf_j_id,
        pts3D=pts3D_keep,
        idx_i=idx_i_keep,
        idx_j=idx_j_keep,
    )

    #   Debugging
    baseline = np.linalg.norm(t_rel)
    print(f"[MAP] Relative baseline norm: {baseline:.4f}")

    print(f"[MAP] Added {created} new MapPoints")
    return created


def cull_weak_mappoints(slam_map, min_observations=2):
    """
    Remove MapPoints with too few observations.
    Also clears references from keyframes.
    """
    to_remove = []

    for mp_id, mp in slam_map.mappoints.items():
        if len(mp.observations) < min_observations:
            to_remove.append(mp_id)

    if len(to_remove) == 0:
        print("[MAP] No weak MapPoints to cull")
        return 0

    for mp_id in to_remove:
        mp = slam_map.mappoints[mp_id]

        for kf_id, kp_idx in list(mp.observations.items()):
            if kf_id in slam_map.keyframes:
                kf = slam_map.keyframes[kf_id]
                if kf.kp_to_mp is not None and 0 <= kp_idx < len(kf.kp_to_mp):
                    if kf.kp_to_mp[kp_idx] == mp_id:
                        kf.kp_to_mp[kp_idx] = None

        del slam_map.mappoints[mp_id]

    print(f"[MAP] Culled {len(to_remove)} weak MapPoints")
    return len(to_remove)


def save_slam_map_points(slam_map):
    pts = []
    for mp in slam_map.mappoints.values():
        if mp.xyz is not None:
            pts.append(mp.xyz)

    if len(pts) == 0:
        print("[PIPE] No SLAM map points to save.")
        return

    pts = np.array(pts, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]

    out_path = PROJECT_ROOT / "outputs" / "aligning_pc" / "post_pnp_cloud.xyz"
    np.savetxt(out_path, pts)
    print(f"[PIPE] Saved {len(pts)} SLAM map points to {out_path}")


if __name__ == "__main__":

    image_files = load_image_files()
    tracking_results = {}
    pose_results = {}
    points_results = {}
    tracking_acceptance = {"tracking_acceptance:": 0}

    slam_map = Map()
    is_initialized = False
    init_kf0_id = None
    init_kf1_id = None
    last_kf_id = None

    pnp_success = 0
    pnp_failed = 0
    pnp_weak = 0
    kf_inserted = 0
    kf_reused = 0

    for i in range(35 - 1):
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
                tracking_acceptance["tracking_acceptance:"] += 1
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
                kf_window=5,
                min_observations=2,
            )

            if T_cw_cur is None:
                print("[PIPE] PnP failed")
                pnp_failed += 1
                continue
            pnp_success += 1
            print(f"[PIPE] PnP succedded with {ninliers} inliers")
            if ninliers < 30:
                pnp_weak += 1
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
                tracking_acceptance=tracking_acceptance,
            )

            # avoid self-edge / duplicate work
            if kf_j_id == last_kf_id:
                print("[PIPE] Same keyframe returned, skipping")
                kf_reused += 1
                continue
            kf_inserted += 1

            kf_j = slam_map.keyframes[kf_j_id]

            # attach existing MapPoints observed by PnP
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

            R_rel = T_rel[:3, :3]
            t_rel = T_rel[:3, 3]

            #   debugging
            print(f"[PIPE] KF{last_kf_id} -> KF{kf_j_id}")
            print("det(R_rel):", np.linalg.det(R_rel))
            print("t_rel:", t_rel)
            print("t_rel norm:", np.linalg.norm(t_rel))

            created = triangulate_new_features_between_keyframes(
                slam_map=slam_map,
                kf_i_id=last_kf_id,
                kf_j_id=kf_j_id,
                R_rel=R_rel,
                t_rel=t_rel,
                K=slam_map.keyframes[last_kf_id].K,
            )

            print(f"[PIPE] Added {created} new MapPoints between KF{last_kf_id} and KF{kf_j_id}")
            if kf_j_id % 5 == 0:
                cull_weak_mappoints(slam_map, min_observations=2)
            last_kf_id = kf_j_id

    # changed nothing 
    optimise_pose_graph(slam_map, max_nfev=50, robust=True, verbose=2) 
    print_map(slam_map)

    if True:
        # Save the estimated trajectory
        estimated_trajectory_file = PROJECT_ROOT / "outputs" / "tests" / "pnp_tests.txt"
        save_estimated_trajectory(slam_map, image_files, estimated_trajectory_file)

        print(f"[PIPE] Estimated trajectory saved to {estimated_trajectory_file}")

    print("tracking_acceptance", tracking_acceptance, "out of", len(tracking_results))
    print(f"[MAP] Number of MapPoints: {len(slam_map.mappoints)}")

    if True:
        save_slam_map_points(slam_map)

    print("PnP success:", pnp_success)
    print("PnP failed:", pnp_failed)
    print("PnP weak (<30 inliers):", pnp_weak)
    print("Keyframes inserted:", kf_inserted)
    print("Frames reused existing keyframe:", kf_reused)