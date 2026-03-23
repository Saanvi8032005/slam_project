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
from aligning_pc.icp import icp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
testing_dir = PROJECT_ROOT / "outputs" / "aligning_pc"
os.makedirs(testing_dir, exist_ok=True)


def align_point_clouds(pose_results: dict,
                       points_results: dict,
                       output_name="global_points.npy",
                       save=None
                       ):

    pair_ids = sorted(pose_results.keys())
    if not pair_ids:
        print("[ALIGN] No common pose/point pairs to align")
        return np.empty((0, 3))

    camera_poses = []  # list of 4x4 matrices, one per camera
    C = np.eye(4)      # world = camera 0 frame
    camera_poses.append(C)

    prev_pts = None     # previous pair's local cloud

    valid_pairs = []  # Track valid pairs (pid, valid_idx)

    for idx, pid in enumerate(pair_ids):
        if pid not in points_results:
            print(f"[ALIGN] Skipping pair {pid}: No points found in points_results")
            continue

        R = pose_results[pid]["R"].reshape(3, 3)
        t = pose_results[pid]["t"].reshape(3, 1)

        # --- Optional ICP refinement for local mapping ---
        curr_pts = points_results[pid].get("points", np.empty((0, 3)))

        if prev_pts is not None and curr_pts.size > 0 and prev_pts.size > 0:
            try:
                # only run if we have enough points
                if prev_pts.shape[0] >= 50 and curr_pts.shape[0] >= 50:
                    R_icp, t_icp = icp(
                        A=prev_pts,      # frame i-1 points
                        B=curr_pts,      # frame i points
                        init_R=R,
                        init_t=t,
                    )
                    R = R_icp
                    t = t_icp
            except Exception as e:
                print(f"[ALIGN][ICP] Skipping ICP for {pid}: {e}")

        prev_pts = curr_pts

        # --- Build transform with (possibly) refined R, t ---
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)

        C = C @ T          # pose of next camera in world
        camera_poses.append(C)
        valid_pairs.append((pid, len(camera_poses) - 1))  # Track valid pair and index

    # camera_poses[k] = pose of camera k in world frame
    # valid_pairs contains (pid, valid_idx) for valid pairs

    # 3) Transform each per-pair point cloud into world frame
    global_points_list = []

    for pid, valid_idx in valid_pairs:
        pts_local = points_results.get(pid, {}).get("points", np.empty((0, 3)))
        if pts_local.size == 0:
            print(f"[ALIGN] Skipping pair {pid}: Empty point cloud")
            continue

        # pair i → points are in camera i frame
        C_i = camera_poses[valid_idx]  # Use valid index for camera_poses

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

    debug = False 
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
            for pid, valid_idx in valid_pairs:
                pts_local = points_results[pid]["points"]
                if pts_local.size == 0:
                    continue

                C_i = camera_poses[valid_idx]
                P_h = np.hstack([pts_local, np.ones((pts_local.shape[0], 1))])
                P_w = (C_i @ P_h.T).T[:, :3]

                d = np.linalg.norm(P_w, axis=1)
                P_plot = P_w[d < r]

                ax.scatter(
                    P_plot[:, 0],
                    P_plot[:, 1],
                    P_plot[:, 2],
                    s=4,
                    c=colours[valid_idx % len(colours)],
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
