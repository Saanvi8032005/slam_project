import open3d as o3d
import numpy as np
import os

dir_path = "/Users/saanvibajaj/Y3_SLAM_Project/report_results/pipeline4/room/final_cloud.xyz"

data = np.loadtxt(dir_path)
xyz = data[:, :3]
rgb = data[:, 3:6] / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

center = xyz.mean(axis=0)

# --- 1) set a good initial "top-looking" pose by rotating the cloud itself ---
# tilt slightly downward so you keep seeing the desk top, not underneath
Rx = pcd.get_rotation_matrix_from_xyz((np.deg2rad(-25), 0, 0))
pcd.rotate(Rx, center=center)
# flip upside down view
Rz = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
pcd.rotate(Rz, center=center)

# optional: small left/right correction if needed
# Ry0 = pcd.get_rotation_matrix_from_xyz((0, np.deg2rad(10), 0))
# pcd.rotate(Ry0, center=center)

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=1280, height=720)
vis.add_geometry(pcd)

render_option = vis.get_render_option()
render_option.point_size = 1.0
render_option.background_color = np.array([0, 0, 0])
render_option.light_on = False

ctr = vis.get_view_control()
ctr.set_lookat(center)
ctr.set_front([0, 0, -1])
ctr.set_up([0, 1, 0])
ctr.set_zoom(0.45)

save_dir = "/Users/saanvibajaj/Y3_SLAM_Project/gifs/frames"
os.makedirs(save_dir, exist_ok=True)

# --- 2) pure turntable rotation around vertical axis only ---
for i in range(120):
    Ry = pcd.get_rotation_matrix_from_xyz((0, np.deg2rad(3), 0))
    pcd.rotate(Ry, center=center)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{save_dir}/frame_{i:03d}.png")

vis.destroy_window()