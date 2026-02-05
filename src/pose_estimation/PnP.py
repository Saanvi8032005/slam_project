"""
PnP Tracking Module

This module provides functionality for PnP tracking, which estimates the
absolute pose of a camera given 2D-3D correspondences.
"""

import numpy as np
import cv2 as cv
from keyframe_selection.keyframe_selec import Map, Keyframe


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
