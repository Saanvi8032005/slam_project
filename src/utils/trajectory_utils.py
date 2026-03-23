import numpy as np


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


def save_estimated_trajectory(slam_map, image_files, output_file):
    """
    Save the estimated trajectory from the SLAM map to a file.

    Parameters
    ----------
    slam_map : Map
        The SLAM map containing keyframes and edges.
    image_files : List[Path]
        List of image file paths, sorted by timestamp.
    output_file : str or Path
        Path to the output file where the trajectory will be saved.
    """
    keyframe_timestamps = extract_keyframe_timestamps(slam_map, image_files)

    with open(output_file, "w") as f:
        for kf_id, kf in slam_map.keyframes.items():
            T_cw = kf.T_cw
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]

            # Convert rotation matrix to quaternion
            qw = np.sqrt(1.0 + R_cw[0, 0] + R_cw[1, 1] + R_cw[2, 2]) / 2.0
            qx = (R_cw[2, 1] - R_cw[1, 2]) / (4.0 * qw)
            qy = (R_cw[0, 2] - R_cw[2, 0]) / (4.0 * qw)
            qz = (R_cw[1, 0] - R_cw[0, 1]) / (4.0 * qw)

            # Get timestamp for the keyframe
            timestamp = keyframe_timestamps[kf_id]

            # Write to file
            f.write(f"{timestamp} {t_cw[0]} {t_cw[1]} {t_cw[2]} {qx} {qy} {qz} {qw}\n")

    print(f"[PIPE] Saved estimated trajectory to {output_file}")


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


def save_estimated_trajectory2(slam_map, image_files, output_file):
    """
    Save estimated trajectory in TUM format:
        timestamp tx ty tz qx qy qz qw

    IMPORTANT:
    - Internal poses are stored as T_cw (world -> camera)
    - TUM format expects camera pose in world coordinates
    - Therefore we must save T_wc = inv(T_cw)
    """
    keyframe_timestamps = extract_keyframe_timestamps(slam_map, image_files)

    with open(output_file, "w") as f:
        for kf_id in sorted(slam_map.keyframes.keys()):
            kf = slam_map.keyframes[kf_id]

            T_cw = kf.T_cw
            T_wc = np.linalg.inv(T_cw)

            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]

            qx, qy, qz, qw = rotation_matrix_to_quaternion(R_wc)

            timestamp = keyframe_timestamps[kf_id]

            f.write(
                f"{timestamp:.6f} "
                f"{t_wc[0]:.9f} {t_wc[1]:.9f} {t_wc[2]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )

    print(f"[PIPE] Saved estimated trajectory to {output_file}")