import numpy as np
from scipy.spatial.transform import Rotation as R


def save_estimated_trajectory(slam_map, image_files, output_file):
    """
    Save the estimated trajectory in TUM format:
    timestamp tx ty tz qx qy qz qw

    Assumes keyframes store T_cw (world-to-camera), so we invert to get T_wc.
    """
    keyframe_timestamps = extract_keyframe_timestamps(slam_map, image_files)

    # Sort keyframes by frame_id for consistent trajectory order
    ordered_items = sorted(
        slam_map.keyframes.items(),
        key=lambda item: item[1].frame_id
    )

    with open(output_file, "w") as f:
        for kf_id, kf in ordered_items:
            T_cw = kf.T_cw
            T_wc = np.linalg.inv(T_cw)

            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]

            quat = R.from_matrix(R_wc).as_quat()   # returns [qx, qy, qz, qw]
            qx, qy, qz, qw = quat

            timestamp = keyframe_timestamps[kf_id]

            f.write(
                f"{timestamp} "
                f"{t_wc[0]} {t_wc[1]} {t_wc[2]} "
                f"{qx} {qy} {qz} {qw}\n"
            )

    print(f"[PIPE] Saved estimated trajectory to {output_file}")


def extract_keyframe_timestamps(slam_map, image_files):
    """
    Extract timestamps for all keyframes in the SLAM map.

    Parameters
    ----------
    slam_map : Map
        The SLAM map containing keyframes.
    image_files : List[Path]
        List of image file paths, sorted by timestamp.

    Returns
    -------
    keyframe_timestamps : Dict[int, float]
        Dictionary mapping keyframe IDs to their timestamps.
    """
    keyframe_timestamps = {}
    for kf_id, kf in slam_map.keyframes.items():
        frame_id = kf.frame_id
        timestamp = float(image_files[frame_id].stem)  # Extract timestamp from filename
        keyframe_timestamps[kf_id] = timestamp
    return keyframe_timestamps


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to quaternion (qx, qy, qz, qw).
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q[0], q[1], q[2], q[3]

