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
