import numpy as np
import cv2
import glob
import os
from pathlib import Path

np.set_printoptions(suppress=True)
DATA_DIR = Path("data/calibration")
output_dir = Path("outputs/calibration")
os.makedirs(output_dir, exist_ok=True)

# termination criteria: Keeps improving the detected corner positions until
# either 30 iterations are done, or the improvement is less than 0.001 pixels
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(os.path.join(DATA_DIR, "*.jpg"))

# debugging: checks if images are found in the specified directory
if not images:
    print(f"No images found in directory: {DATA_DIR}")
else:
    print(f"Found {len(images)} images in directory: {DATA_DIR}")

for fname in images:
    # Debugging
    print(f"Processing image: {fname}")
    img = cv2.imread(fname)

    # debugging: check if image is read correctly
    if img is None:
        print(f"Failed to read image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        # debugging
        print(f"Chessboard corners found in image: {fname}")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"Chessboard corners not found in image: {fname}")

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Calibration RMS reprojection error (cv2.calibrateCamera return):", ret)
print("Camera matrix (mtx):\n", mtx)
print("Distortion coefficients (dist):\n", dist.ravel())
results_file = output_dir / "calibration_results.txt"
with results_file.open("w") as f:
    f.write(
        f"Calibration RMS reprojection error "
        f"(cv2.calibrateCamera return): {ret}\n"
    )
    f.write("Camera matrix (mtx):\n")
    f.write(f"{mtx}\n")
    f.write("Distortion coefficients (dist):\n")
    f.write(f"{dist.ravel()}\n")


for img_path in images:  # Iterate over all images
    img = cv2.imread(img_path)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Save the undistorted image to the output directory
    output_path = output_dir / Path(img_path).name
    cv2.imwrite(str(output_path), dst)

tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print("total error: ", tot_error/len(objpoints))
