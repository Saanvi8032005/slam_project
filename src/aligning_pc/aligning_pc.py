"""
Build a global point cloud by chaining camera poses and transforming
each per-pair point cloud.

pose_results: dict
    e.g. {
        "000": {"R": R0, "t": t0, "K": K, "mask": mask0},
        "001": {"R": R1, "t": t1, "K": K, "mask": mask1},
        ...
    }
points_results: dict
    e.g. {
        "000": {"points": pts3D_0, ...},
        "001": {"points": pts3D_1, ...},
        ...
    }
"""
import numpy as np
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
testing_dir = PROJECT_ROOT / "outputs" / "aligning_pc"
os.makedirs(testing_dir, exist_ok=True)


def align_point_clouds(pose_results: dict,
                       points_results: dict,
                       output_name="global_points.npy",
                       save=None
                       ):

    pair_ids = sorted(pose_results.keys())    
    #   check again

    # 2) Compute camera poses C_k in world frame
    #    C_0 = I, then C_{k+1} = C_k * T_k
    camera_poses = []  # list of 4x4 matrices, one per camera
    C = np.eye(4)      # world = camera 0 frame
    camera_poses.append(C)

    for pid in pair_ids:
        R = pose_results[pid]["R"].reshape(3, 3)
        t = pose_results[pid]["t"].reshape(3, 1)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t

        C = C @ T          # pose of next camera
        camera_poses.append(C)

    # camera_poses[k] = pose of camera k in world frame
    # pair "000" uses camera 0, pair "001" uses camera 1, etc.

    # 3) Transform each per-pair point cloud into world frame
    global_points_list = []

    for idx, pid in enumerate(pair_ids):
        pts_local = points_results[pid]["points"]  # (N, 3)
        if pts_local.size == 0:
            continue

        # pair i → points are in camera i frame
        C_i = camera_poses[idx]  # 4x4

        P_h = np.hstack([pts_local, np.ones((pts_local.shape[0], 1))])  # (N, 4)
        P_w = (C_i @ P_h.T).T[:, :3]  # (N, 3)
        global_points_list.append(P_w)

    if not global_points_list:
        print("[ALIGN] No points to merge")
        return np.empty((0, 3))

    global_points = np.vstack(global_points_list)
        # --- OPTIONAL: normalise global scale for nicer visualisation ---
    # target radius in "world units" (e.g. 5 m)

    # Save
    if save:
        out_path = testing_dir / output_name
        np.save(out_path, global_points)
        print(f"[ALIGN] Saved global point cloud: {out_path}")
    debug = True
    if debug:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            colours = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

            # Global clipping radius based on all points
            d_all = np.linalg.norm(global_points, axis=1)
            r = np.percentile(d_all, 98)
            print(f"[ALIGN] Global debug radius (98th percentile): {r:.2f}")

            # Plot each cloud separately with same clipping
            for idx, pid in enumerate(pair_ids):
                pts_local = points_results[pid]["points"]
                if pts_local.size == 0:
                    continue

                C_i = camera_poses[idx]
                P_h = np.hstack([pts_local, np.ones((pts_local.shape[0], 1))])
                P_w = (C_i @ P_h.T).T[:, :3]

                d = np.linalg.norm(P_w, axis=1)
                P_plot = P_w[d < r]

                ax.scatter(
                    P_plot[:, 0],
                    P_plot[:, 1],
                    P_plot[:, 2],
                    s=4,
                    c=colours[idx % len(colours)],
                    label=f"Cloud {pid}",
                )

            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(-r, r)
            ax.set_title("DEBUG: Point clouds per frame (different colours)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("[DEBUG VIS ERROR]", e)

    return global_points
