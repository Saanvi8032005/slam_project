import numpy as np


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