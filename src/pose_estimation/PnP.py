"""
PnP Tracking Module

This module provides functionality for PnP tracking, which estimates the
absolute pose of a camera given 2D-3D correspondences.
"""

import numpy as np
import cv2 as cv
from keyframe_selection.keyframe_selec import Map, Keyframe
from tests.reprojection_err import reprojection_error


def pnp_tracking(
        slam_map: Map,
        current_kf: Keyframe,
        image_points: np.ndarray
        ):
    """
    Perform PnP tracking to estimate the absolute pose of the current keyframe.

    Args:
        slam_map (Map): The SLAM map containing keyframes and MapPoints.
        current_kf (Keyframe): The current keyframe to track.
        image_points (np.ndarray): 2D keypoints detected in the current frame.

    Returns:
        np.ndarray: Updated camera pose (T_cw) for the current keyframe.
    """
    print("\n=== PnP TRACKING ===")

    # Find 2D-3D correspondences
    object_points = []
    image_points_filtered = []
    for i, mp_id in enumerate(current_kf.kp_to_mp):
        if mp_id is not None and mp_id in slam_map.mappoints:
            object_points.append(slam_map.mappoints[mp_id].Xw)
            image_points_filtered.append(image_points[i])

    if len(object_points) < 4:
        print("[PnP] Not enough 2D-3D correspondences for PnP.")
        return None

    object_points = np.array(object_points, dtype=np.float32)
    image_points_filtered = np.array(image_points_filtered, dtype=np.float32)

    # Solve PnP
    success, rvec, tvec, inliers = cv.solvePnPRansac(
        object_points,
        image_points_filtered,
        current_kf.K,
        None,
        flags=cv.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None or len(inliers) < 4:
        print("[PnP] PnP failed or insufficient inliers.")
        return None

    # Convert rvec and tvec to T_cw
    R, _ = cv.Rodrigues(rvec)
    T_cw = np.eye(4)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = tvec.ravel()

    # Update the keyframe pose in the map
    current_kf.T_cw = T_cw
    slam_map.keyframes[current_kf.kf_id] = current_kf

    print(f"[PnP] Updated pose for KF{current_kf.kf_id}:\n{T_cw}")
    return T_cw


def run_pnp_for_frame(
    slam_map,
    last_kf_id: int,
    keypoints,
    descriptors,
    K,
    min_matches: int = 20,
    reproj_error: float = 4.0,
    kf_window: int = 5,
    min_observations: int = 2,
):
    """
    Estimate current frame pose using 2D-3D correspondences from MapPoints
    observed in a local window of recent keyframes.
    """
    import numpy as np
    import cv2 as cv

    if descriptors is None or len(descriptors) == 0:
        print("[PnP] No descriptors in current frame")
        return None, 0, {}, {}

    if last_kf_id not in slam_map.keyframes:
        print(f"[PnP] last_kf_id {last_kf_id} not in slam_map.keyframes")
        return None, 0, {}, {}

    # -------------------------------------------------
    # 1. Gather candidate MapPoints from recent keyframes
    # -------------------------------------------------
    start_kf_id = max(0, last_kf_id - kf_window + 1)
    candidate_kf_ids = list(range(start_kf_id, last_kf_id + 1))

    mp_ids = []
    mp_desc = []
    mp_xyz = []
    seen_mp_ids = set()

    for kf_id in candidate_kf_ids:
        if kf_id not in slam_map.keyframes:
            continue

        kf = slam_map.keyframes[kf_id]
        if kf.kp_to_mp is None:
            continue

        for mp_id in kf.kp_to_mp:
            if mp_id is None:
                continue
            if mp_id in seen_mp_ids:
                continue
            if mp_id not in slam_map.mappoints:
                continue

            mp = slam_map.mappoints[mp_id]

            if mp.xyz is None:
                continue
            if not np.all(np.isfinite(mp.xyz)):
                continue
            if mp.descriptor is None:
                continue
            if len(mp.observations) < min_observations:
                continue

            seen_mp_ids.add(mp_id)
            mp_ids.append(mp_id)
            mp_desc.append(mp.descriptor)
            mp_xyz.append(mp.xyz)

    if len(mp_ids) < min_matches:
        print(f"[PnP] Not enough candidate MapPoints: {len(mp_ids)}")
        return None, 0, {}, {}

    mp_desc = np.asarray(mp_desc, dtype=np.uint8)
    mp_xyz = np.asarray(mp_xyz, dtype=np.float32).reshape(-1, 3)

    print(f"[PnP] Candidate keyframes: {candidate_kf_ids}")
    print(f"[PnP] Candidate MapPoints: {len(mp_ids)}")

    # -------------------------------------------------
    # 2. Match current descriptors to MapPoint descriptors
    # -------------------------------------------------
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(descriptors, mp_desc, k=2)

    good_matches = []
    used_mp_local_idx = set()

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < 0.75 * n.distance:
            if m.trainIdx in used_mp_local_idx:
                continue
            used_mp_local_idx.add(m.trainIdx)
            good_matches.append(m)

    if len(good_matches) < min_matches:
        print(f"[PnP] Not enough good matches: {len(good_matches)}")
        return None, 0, {}, {}

    print(f"[PnP] Good descriptor matches: {len(good_matches)}")

    # -------------------------------------------------
    # 3. Build 2D-3D correspondences
    # -------------------------------------------------
    image_points = []
    object_points = []
    match_kp_idxs = []
    match_mp_ids = []

    for m in good_matches:
        kp_idx = m.queryIdx
        mp_local_idx = m.trainIdx

        x, y = keypoints[kp_idx].pt
        Xw = mp_xyz[mp_local_idx]

        image_points.append([x, y])
        object_points.append(Xw)
        match_kp_idxs.append(kp_idx)
        match_mp_ids.append(mp_ids[mp_local_idx])

    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)

    # -------------------------------------------------
    # 4. Solve PnP with RANSAC
    # -------------------------------------------------
    ok, rvec, tvec, inliers = cv.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=K,
        distCoeffs=None,
        iterationsCount=200,
        reprojectionError=reproj_error,
        confidence=0.99,
        flags=cv.SOLVEPNP_EPNP,
    )

    if not ok or inliers is None or len(inliers) < 6:
        print("[PnP] solvePnPRansac failed")
        return None, 0, {}, {}

    inliers = inliers.ravel()
    ninliers = len(inliers)

    print(f"[PnP] Inliers: {ninliers}")

    # -------------------------------------------------
    # 5. Refine
    # -------------------------------------------------
    ok_refine, rvec, tvec = cv.solvePnP(
        object_points[inliers],
        image_points[inliers],
        K,
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv.SOLVEPNP_ITERATIVE,
    )

    if not ok_refine:
        print("[PnP] refinement failed")
        return None, 0, {}, {}

    R, _ = cv.Rodrigues(rvec)

    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = tvec.ravel()

    inlier_kp_to_mp = {}
    for idx in inliers:
        kp_idx = match_kp_idxs[idx]
        mp_id = match_mp_ids[idx]
        inlier_kp_to_mp[int(kp_idx)] = int(mp_id)

    reproj_error = reprojection_error(
        object_points[inliers],
        image_points[inliers],
        R,
        tvec.reshape(3, 1),
        K)
    mean_reproj_error = np.mean(reproj_error) if len(reproj_error) > 0 else 0.0
    n_matches = len(good_matches)
    inlier_ratio = ninliers / n_matches if n_matches > 0 else 0.0
    stats = {
        "n_matches": n_matches,
        "ninliers": ninliers,
        "inlier_ratio": inlier_ratio,
        "reprojection_error": mean_reproj_error,
        "R": R,
        "t": tvec.reshape(3, 1),
    }
    return T_cw, ninliers, inlier_kp_to_mp, stats
