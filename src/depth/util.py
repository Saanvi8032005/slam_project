import numpy as np
import cv2 as cv  # Add this import for OpenCV


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
        flags=cv.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < 6:
        print("[RGBD] PnP failed")
        return None

    inliers = inliers[:, 0]
    obj_in = obj_pts[inliers]
    img_in = img_pts[inliers]

    rvec, tvec = cv.solvePnPRefineLM(
        objectPoints=obj_in,
        imagePoints=img_in,
        cameraMatrix=K,
        distCoeffs=None,
        rvec=rvec,
        tvec=tvec,
    )

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
