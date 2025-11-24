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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSE_OUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset"


def stage_tracking():
    """
    Stage 1: run feature detection + matching + filtering.
    Writes pts1, pts2 to an .npz file for pose_estimation to use.
    """
    MATCHER = "flann"
    FILTER = "hist"

    print("\n=== STAGE 1: TRACKING / MATCHING ===")
    matching(matcher=MATCHER, filter_method=FILTER)
    # matching()
    # already saves: outputs/pose_estimation/matches_left03_left04.npz


def stage_pose():
    """
    Stage 2: run pose estimation + triangulation
    using the matches saved by stage_tracking().
    """
    print("\n=== STAGE 2: POSE ESTIMATION ===")
    # If your pose_estimate() takes a filename argument, pass it here.
    # Otherwise, it can just use its default (matches_left03_left04.npz).
    pose_estimate('matches_left03_left04.npz')


def stage_triangulate():
    print("\n=== STAGE 3: TRIANGULATION ===")
    triangulate_from_files()


def stage_visualise():
    print("\n=== STAGE 4: VISUALISATION ===")
    visualize_points()


if __name__ == "__main__":
    # Set these to True/False depending on what you want to run.
    RUN_TRACKING = True
    RUN_POSE = True
    RUN_TRIANGULATE = True
    RUN_VISUALISE = True

    if RUN_TRACKING:
        stage_tracking()

    if RUN_POSE:
        stage_pose()

    if RUN_TRIANGULATE:
        stage_triangulate()

    if RUN_VISUALISE:
        stage_visualise()
