import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSE_DIR = PROJECT_ROOT / "outputs" / "pose_estimation"
SAVE_HERE = PROJECT_ROOT / "outputs" / "visualising"


def load_pose(pose_file):
    """Load R, t, K from a saved pose file."""
    data = np.load(POSE_DIR / pose_file)
    return data["R"], data["t"], data["K"]


def load_points(points_file):
    """Load triangulated 3D points"""
    return np.load(POSE_DIR / points_file)


def align_point_clouds(pose_files,
                       point_files,
                       output_name="global_points.npy",
                       save=True
                       ):
    """
    Build a global point cloud by chaining poses and transforming each cloud.
    pose_files: list of pose file names (pose_000.npz, pose_001.npz, ...)
    point_files: list of corresponding points files (points_000.npy, ...)
    """

    global_points = []

    T_world = np.eye(4)  # start world frame = first camera
    T_prev = T_world

    for pose_file, points_file in zip(pose_files, point_files):
        R, t, K = load_pose(pose_file)
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)

        # Build 4x4 transform
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t

        # Chain transforms
        T_world = T_prev @ T
        T_prev = T_world

        # Load 3D points
        P = load_points(points_file)
        P_h = np.hstack([P, np.ones((P.shape[0], 1))])  # homogeneous

        # Transform into world frame
        P_w = (T_world @ P_h.T).T[:, :3]

        global_points.append(P_w)

    # Merge into one cloud
    global_points = np.vstack(global_points)

    # Save
    if save:
        out_path = SAVE_HERE / output_name
        np.save(out_path, global_points)
        print(f"[ALIGN] Saved global point cloud: {out_path}")

    # ---------------------------------------------------------
    # TEMPORARY DEBUG: Visualise each cloud in a different colour
    # ---------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # reload each individual cloud for plotting
        colours = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

        for idx, (pose_file, points_file) in enumerate(zip(
                pose_files, point_files)
                ):
            P = load_points(points_file)
            P_h = np.hstack([P, np.ones((P.shape[0], 1))])
            # Apply the same chaining as before for correct position
            R, t, _ = load_pose(pose_file)
            R = R.reshape(3, 3)
            t = t.reshape(3, 1)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3:] = t

            # chain transform again (debug mode only)
            if idx == 0:
                T_world_debug = np.eye(4)
            else:
                T_world_debug = T_world_debug @ T

            P_w_debug = (T_world_debug @ P_h.T).T[:, :3]

            ax.scatter(
                P_w_debug[:, 0],
                P_w_debug[:, 1],
                P_w_debug[:, 2],
                s=4,
                c=colours[idx % len(colours)],
                label=f"Cloud {idx}"
            )

        ax.legend()
        ax.set_title("DEBUG: Point clouds per frame (different colours)")
        plt.show()

    except Exception as e:
        print("[DEBUG VIS ERROR]", e)
    # ---------------------------------------------------------
    # END DEBUG
    # ---------------------------------------------------------

    return global_points
