The implementation of this project was developed independently, with the exception of standard libraries and specific external tools used for evaluation and supporting tasks. All external contributions are acknowledged below:

Core Libraries
-   OpenCV [12]
Used for feature detection (ORB), matching (FLANN), pose estimation (findEssentialMat,recoverPose, solvePnP), triangulation, and camera calibration (calibrateCamera).
-   NumPy: Used for numerical operations and matrix computations throughout the pipeline.

Depth Estimation
- MiDaS (Monocular Depth Estimation): Pre-trained models from MiDaS were used to generate pseudo-depth maps for monocular depth experiments. These provide relative depth rather than metric scale and were not modified.

Evaluation Tools
- TUM RGB-D Evaluation Toolkit [20]
- The scripts evaluate_ate.py and evaluate_rpe.py were used to compute statistics.
- Association Script (associate.py)
Used to synchronise RGB frames with ground truth poses from the TUM dataset.

Camera Calibration
- OpenCV Calibration Tools (Zhang’s Method) [28]: Used to estimate intrinsic camera parameters from checkerboard images.
- Calibration Images: Checkerboard calibration images were captured and processed manually. Corner detection and parameter estimation were performed using OpenCV functions.

Visualisation and Processing
- Open3D: Used for point cloud visualisation, transformation, and inspection.
- CloudCompare / MeshLab: Used for visualising and exporting final point clouds and generating figures