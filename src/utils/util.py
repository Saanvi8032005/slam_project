import numpy as np
import cv2 as cv
from keyframe_selection.keyframe_selec import Map, Keyframe


def backproject_keypoint(kp, depth_img, fx, fy, cx, cy):
    u, v = kp.pt
    u = int(round(u))
    v = int(round(v))

    if u < 0 or v < 0 or u >= depth_img.shape[1] or v >= depth_img.shape[0]:
        return None

    z = depth_img[v, u]
    if not np.isfinite(z) or z <= 0:
        return None

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def depth_to_meters(depth_raw, scale=5000.0):
    depth_m = depth_raw.astype(np.float32) / scale
    depth_m[depth_raw == 0] = np.nan
    return depth_m


def backproject_keypoints(kpts, descs, depth_img, fx, fy, cx, cy):
    pts3d = []
    valid_kpts = []
    valid_descs = []

    for kp, desc in zip(kpts, descs):
        p3d = backproject_keypoint(kp, depth_img, fx, fy, cx, cy)
        if p3d is None:
            continue
        pts3d.append(p3d)
        valid_kpts.append(kp)
        valid_descs.append(desc)

    if len(pts3d) == 0:
        return [], None, np.empty((0, 3), dtype=np.float32)

    return valid_kpts, np.array(valid_descs), np.array(pts3d, dtype=np.float32)


def stage_pose_rgbd(tracking_entry, depth1_m, K):
    """
    Estimate relative pose from frame1 -> frame2 using:
      - 3D points from frame1 depth
      - 2D observations in frame2
    """
    pair_id = tracking_entry["pair_id"]
    pts1 = tracking_entry["pts1"]
    pts2 = tracking_entry["pts2"]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    print(f"\n=== STAGE 2: RGB-D PnP POSE ({pair_id}) ===")

    obj_pts, img_pts, valid_idx = matched_pixels_to_3d2d(
        pts1, pts2, depth1_m, fx, fy, cx, cy
    )

    print(f"[RGBD] Valid 3D-2D correspondences: {len(obj_pts)}")

    if len(obj_pts) < 6:
        print("[RGBD] Not enough 3D-2D correspondences for PnP")
        return None

    success, rvec, tvec, inliers = cv.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=None,
        iterationsCount=200,
        reprojectionError=3.0,
        confidence=0.99,
        flags=cv.SOLVEPNP_EPNP,
    )

    if not success or inliers is None or len(inliers) < 6:
        print("[RGBD] PnP failed")
        return None

    inliers = inliers[:, 0]
    obj_in = obj_pts[inliers]
    img_in = img_pts[inliers]

    ok_refine, rvec, tvec = cv.solvePnP(
        objectPoints=obj_in,
        imagePoints=img_in,
        cameraMatrix=K,
        distCoeffs=None,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv.SOLVEPNP_ITERATIVE,
    )

    if not ok_refine:
        print("[RGBD] Iterative PnP refinement failed")
        return None

    R, _ = cv.Rodrigues(rvec)

    T_21 = np.eye(4, dtype=np.float64)
    T_21[:3, :3] = R
    T_21[:3, 3] = tvec.reshape(3)

    print(f"[RGBD] PnP inliers: {len(inliers)} / {len(obj_pts)}")
    return {
        "pair_id": pair_id,
        "R": R,
        "t": tvec.reshape(3, 1),
        "T_21": T_21,
        "num_inliers": len(inliers),
        "inlier_ratio": len(inliers) / len(obj_pts),
        "obj_pts": obj_pts,
        "img_pts": img_pts,
        "obj_pts_in": obj_in,
        "img_pts_in": img_in,
        "valid_idx": valid_idx,
        "inliers_idx": inliers,
    }


def generate_3d_points_from_depth(depth_image, K):
    """
    Generate 3D points from a depth image.

    Args:
        depth_image (np.ndarray): Depth image.
        K (np.ndarray): Camera intrinsic matrix.

    Returns:
        np.ndarray: 3D points (N x 3).
    """
    h, w = depth_image.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Generate pixel grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Back-project to 3D
    z = depth_image.astype(np.float32) / 5000.0  # Convert depth to meters if needed
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    points_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points_3d


def keypoints_with_depth(kp, desc, depth_m, fx, fy, cx, cy):
    pts2d = []
    pts3d = []
    desc_valid = []

    h, w = depth_m.shape

    for k, d in zip(kp, desc):
        u, v = k.pt
        ui = int(round(u))
        vi = int(round(v))

        if ui < 0 or vi < 0 or ui >= w or vi >= h:
            continue

        z = depth_m[vi, ui]
        if not np.isfinite(z) or z <= 0:
            continue

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts2d.append([u, v])
        pts3d.append([x, y, z])
        desc_valid.append(d)

    return (
        np.array(pts2d, dtype=np.float32),
        np.array(desc_valid, dtype=np.uint8) if len(desc_valid) > 0 else np.empty((0, 32), dtype=np.uint8),
        np.array(pts3d, dtype=np.float32),
    )


def load_txt_entries(txt_path, dataset_root):
    entries = []
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            ts, rel_path = line.split()
            entries.append((float(ts), dataset_root / rel_path))
    return entries


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


def create_new_mappoints_from_depth_for_keyframe(
    slam_map,
    kf_id,
    depth_m,
    max_depth=5.0,
    max_new_points=80,
):
    kf = slam_map.keyframes[kf_id]
    return create_mappoints_from_depth(
        slam_map=slam_map,
        kf_id=kf_id,
        depth_m=depth_m,
        keypoints_xy=kf.keypoints_xy,
        descriptors=kf.descriptors,
        max_depth=max_depth,
        max_new_points=max_new_points,
    )


def detect_orb_features(img, nfeatures=4000):
    orb = cv.ORB_create(nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if keypoints is None:
        keypoints = []
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)

    keypoints_xy = np.array([kp.pt for kp in keypoints], dtype=np.float32) if len(keypoints) > 0 else np.empty((0, 2), dtype=np.float32)
    return keypoints, keypoints_xy, descriptors


def initialize_map_rgbd(
    slam_map: Map,
    frame_id: int,
    img: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    max_depth: float = 5.0,
):
    """
    Initialise the SLAM map from a single RGB-D frame.
    Creates:
      - one keyframe at identity pose
      - depth-born map points from valid ORB keypoints
    """

    keypoints, keypoints_xy, descriptors = detect_orb_features(img)

    if keypoints_xy.shape[0] == 0:
        raise ValueError("No ORB features found in initial frame")

    T_cw0 = np.eye(4, dtype=np.float64)

    kf0 = Keyframe(
        kf_id=None,
        frame_id=frame_id,
        T_cw=T_cw0,
        K=K,
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
        kp_to_mp=[None] * len(keypoints_xy),
    )

    kf0_id = slam_map.add_keyframe(kf0)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    created = 0

    for kp_idx, (u_f, v_f) in enumerate(keypoints_xy):
        u = int(round(u_f))
        v = int(round(v_f))

        if not (0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1]):
            continue

        z = depth_m[v, u]
        if not np.isfinite(z) or z <= 0 or z > max_depth:
            continue

        x = (u_f - cx) * z / fx
        y = (v_f - cy) * z / fy

        # since first pose is identity and world = camera0, this is already a world point
        Pw = np.array([x, y, z], dtype=np.float64)

        mp_id = slam_map.add_mappoint(
            xyz=Pw,
            descriptor=descriptors[kp_idx],
            observations={kf0_id: kp_idx},
        )

        slam_map.keyframes[kf0_id].kp_to_mp[kp_idx] = mp_id
        created += 1

    print(f"[MAP] RGB-D initialised with KF{kf0_id}")
    print(f"[MAP] Created {created} initial MapPoints")

    return kf0_id


def matched_pixels_to_3d2d(pts1, pts2, depth1_m, fx, fy, cx, cy):
    obj_pts = []
    img_pts = []
    valid_idx = []

    h, w = depth1_m.shape

    for i, ((u1, v1), (u2, v2)) in enumerate(zip(pts1, pts2)):
        u = int(round(u1))
        v = int(round(v1))

        if u < 0 or v < 0 or u >= w or v >= h:
            continue

        z = depth1_m[v, u]
        if not np.isfinite(z) or z <= 0:
            continue

        x = (u1 - cx) * z / fx
        y = (v1 - cy) * z / fy

        obj_pts.append([x, y, z])   # 3D in camera-1 frame
        img_pts.append([u2, v2])    # 2D in frame 2
        valid_idx.append(i)

    if len(obj_pts) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.array([], dtype=int),
        )

    return (
        np.array(obj_pts, dtype=np.float32),
        np.array(img_pts, dtype=np.float32),
        np.array(valid_idx, dtype=int),
    )


def create_mappoints_from_depth(
    slam_map,
    kf_id,
    depth_m,
    keypoints_xy,
    descriptors,
    max_depth=5.0,
    max_new_points=80,
):
    """
    Ceate MapPoints directly from valid-depth keypoints in a keyframe.
    Assumes keyframe pose is T_cw and keypoints_xy is (N, 2).
    """
    kf = slam_map.keyframes[kf_id]
    T_cw = kf.T_cw
    T_wc = np.linalg.inv(T_cw)
    K = kf.K

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    created = 0

    for kp_idx, (u_f, v_f) in enumerate(keypoints_xy):
        if created >= max_new_points:
            break

        # skip if already linked
        if kf.kp_to_mp[kp_idx] is not None:
            continue


        u = int(round(u_f))
        v = int(round(v_f))

        if not (0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1]):
            continue

        z = depth_m[v, u]
        if not np.isfinite(z) or z <= 0 or z > max_depth:
            continue

        x = (u_f - cx) * z / fx
        y = (v_f - cy) * z / fy

        p_cam = np.array([x, y, z, 1.0], dtype=np.float64)
        p_world = (T_wc @ p_cam)[:3]

        mp_id = slam_map.add_mappoint(
            xyz=p_world,
            descriptor=descriptors[kp_idx],
            observations={kf_id: int(kp_idx)},
        )

        kf.kp_to_mp[kp_idx] = mp_id
        created += 1

    print(f"[MAP] Created {created} depth-born MapPoints in KF{kf_id}")
    return created


def add_new_mappoints_from_depth_for_keyframe(
    slam_map: Map,
    kf_id: int,
    depth_m: np.ndarray,
    max_depth: float = 5.0,
):
    """
    For a keyframe that already exists in the map:
    create new MapPoints from keypoints that are not yet linked to any MapPoint.
    """
    kf = slam_map.keyframes[kf_id]
    T_cw = kf.T_cw
    T_wc = np.linalg.inv(T_cw)

    fx, fy = kf.K[0, 0], kf.K[1, 1]
    cx, cy = kf.K[0, 2], kf.K[1, 2]

    created = 0

    for kp_idx, (u_f, v_f) in enumerate(kf.keypoints_xy):
        if kf.kp_to_mp[kp_idx] is not None:
            continue

        u = int(round(u_f))
        v = int(round(v_f))

        if not (0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1]):
            continue

        z = depth_m[v, u]
        if not np.isfinite(z) or z <= 0 or z > max_depth:
            continue

        x = (u_f - cx) * z / fx
        y = (v_f - cy) * z / fy

        p_cam = np.array([x, y, z, 1.0], dtype=np.float64)
        p_world = (T_wc @ p_cam)[:3]

        mp_id = slam_map.add_mappoint(
            xyz=p_world,
            descriptor=kf.descriptors[kp_idx],
            observations={kf_id: kp_idx},
        )

        kf.kp_to_mp[kp_idx] = mp_id
        created += 1

    print(f"[MAP] Added {created} new depth MapPoints in KF{kf_id}")
    return created
