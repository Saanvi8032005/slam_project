import numpy as np
import open3d as o3d

dir_path = "/Users/saanvibajaj/Y3_SLAM_Project/report_results/pipeline4/room/final_cloud.xyz"
data = np.loadtxt(dir_path)

points = data[:, :3]
colors = data[:, 3:6] / 255.0  # normalize to 0–1

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

save_dir = "/Users/saanvibajaj/Y3_SLAM_Project/gifs/final_cloud.ply"

o3d.io.write_point_cloud(save_dir, pcd)

