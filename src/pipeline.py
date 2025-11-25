"""
pipeline.py

Third script to run the individual stages in order:
1) Tracking / matching  (combined.py)
2) Pose estimation      (pose_estimation.py)
"""

from pathlib import Path

# Adjust these imports to match your actual package structure.
# Example assumes:
#   project/src/tracking/combined.py
#   project/src/pose_estimation/pose_estimation.py
from tracking.combined import matching
from pose_estimation.pose_estimation import pose_estimate
from triangulation.triangulation import triangulate_from_files
from visualising.visualising import visualize_points
from aligning_pc.aligning_pc import align_point_clouds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSE_OUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset"


def load_image_files():
    """
    Load image file paths from the dataset directory.
    Adjust this function based on your dataset structure.
    """
    #   image_files = list(DATA_DIR.glob("*.png"))
    image_files = sorted(DATA_DIR.glob("*.png"))   # ← sorted is crucial
    print(f"[PIPE] Found {len(image_files)} images")
    #   print("FIRST 5 FILES:", image_files[:5])
    return image_files


def stage_tracking(img1, img2):
    """
    Stage 1: run feature detection + matching + filtering.
    Writes pts1, pts2 to an .npz file for pose_estimation to use.
    """
    MATCHER = "flann"
    FILTER = "hist"

    print("\n=== STAGE 1: TRACKING / MATCHING ===")
    matching(matcher=MATCHER,
             filter_method=FILTER,
             img1_path=img1,
             img2_path=img2,
             save_npz=True,
             return_data=False,
             )
    matches_file = "matches.npz"
    return matches_file


def stage_pose(matches_file):
    """
    Stage 2: run pose estimation + triangulation
    using the matches saved by stage_tracking().
    """
    print("\n=== STAGE 2: POSE ESTIMATION ===")
    # If your pose_estimate() takes a filename argument, pass it here.
    # Otherwise, it can just use its default (matches_left03_left04.npz).
    pose_estimate(matches_file=matches_file)
    pose_file = "pose.npz"
    return pose_file


def stage_triangulate(matches_file, pose_file):
    print("\n=== STAGE 3: TRIANGULATION ===")
    triangulate_from_files(matches_file=matches_file,
                           pose_file=pose_file)
    points_file = "points.npy"
    return points_file


def stage_align_pc(pose_files, points_files):
    print("\n=== STAGE 4: ALIGNING POINT CLOUDS ===")
    align_point_clouds(
        pose_files=pose_files,
        point_files=points_files,
        output_name="global_points.npy",
        save=True
    )
    return "global_points.npy"


def stage_visualise(points_file):
    print("\n=== STAGE 5: VISUALISATION ===")
    visualize_points(points_file=points_file)


if __name__ == "__main__":

    image_files = load_image_files()
    pose_files = []
    point_files = []
    #   for i in range(len(image_files) - 1):
    for i in range(len(image_files) - 1):
        img1 = image_files[i]
        img2 = image_files[i + 1]
        pair_id = f"{i:03d}"

        matches_file = stage_tracking(img1, img2)
        pose_file = stage_pose(matches_file)
        points_file = stage_triangulate(matches_file, pose_file)

        pose_files.append(pose_file)
        point_files.append(points_file)

    global_points_file = stage_align_pc(pose_files, point_files)
    stage_visualise(global_points_file)
    print("\n[PIPE] Done processing all image pairs.")
