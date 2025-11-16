import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path

# img = cv.imread('simple.jpg', cv.IMREAD_GRAYSCALE)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tracking"
img_path1 = DATA_DIR / "left01.jpg"
img_path2 = DATA_DIR / "img01.jpeg"
img = cv.imread(str(img_path1), cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
plt.imshow(img2)
plt.show()
