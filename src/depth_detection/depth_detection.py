import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
depth_image_path = PROJECT_ROOT / "data" / "depth_dataset" / "depth.png"

fx, fy = 517.3, 516.5
cx, cy = 318.6, 255.3

def depth_to_pointcloud(depth):
    depth = depth.astype(np.float32) / 5000.0  # mm → metres
    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # filter out zeros
    pts = pts[pts[:,2] > 0]
    return pts


# Load depth
depth = cv.imread("/path/to/depth.png", cv.IMREAD_UNCHANGED)
pts = depth_to_pointcloud(depth)

# Plot
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
plt.show()
