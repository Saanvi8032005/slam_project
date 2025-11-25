# visualise_points.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualising"


def visualize_points(points_file="points.npy"):
    pts_path = OUTPUT_DIR / points_file
    pts3D = np.load(pts_path)
    print(f"[VIS] Loaded {pts3D.shape[0]} 3D points from {pts_path}")

    xs = pts3D[:, 0]
    ys = pts3D[:, 1]
    zs = pts3D[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=4)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Triangulated 3D Points")

    # roughly equal axis scaling
    max_range = max(xs.max() - xs.min(),
                    ys.max() - ys.min(),
                    zs.max() - zs.min())
    mid_x = 0.5 * (xs.max() + xs.min())
    mid_y = 0.5 * (ys.max() + ys.min())
    mid_z = 0.5 * (zs.max() + zs.min())

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_points()
