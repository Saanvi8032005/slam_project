import imageio
import os
import os

folder = "/Users/saanvibajaj/Y3_SLAM_Project/gifs/frames"

images = []

files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

for filename in files:
    filepath = os.path.join(folder, filename)
    images.append(imageio.imread(filepath))

output_path = "/Users/saanvibajaj/Y3_SLAM_Project/gifs/pointcloud.gif"
imageio.mimsave(output_path, images, duration=0.05)