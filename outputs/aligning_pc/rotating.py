import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from matplotlib.animation import FuncAnimation, PillowWriter
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# load your SLAM map

# Load point cloud
points = np.loadtxt("global_points_pose.xyz")   # or np.loadtxt("global_points.xyz")

x = points[:,0]
y = points[:,1]
z = points[:,2]

frames = []

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

for angle in range(0,360,4):   # rotation speed
    ax.clear()
    
    ax.scatter(x,y,z,s=1)
    
    ax.view_init(elev=20, azim=angle)
    
    ax.set_axis_off()
    
    fig.canvas.draw()
    
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    frames.append(frame)

imageio.mimsave("pointcloud_rotation.gif", frames, fps=20)