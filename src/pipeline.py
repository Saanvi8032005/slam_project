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
DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset" / "rgb"
TEMP_DIR = PROJECT_ROOT / "outputs" / "temp"


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


def stage_tracking(img1, img2, pair_id):
    """
    Stage 1: run feature detection + matching + filtering.
    Writes pts1, pts2 to an .npz file for pose_estimation to use.
    """
    MATCHER = "flann"
    FILTER = "hist"

    print(f"\n=== STAGE 1: TRACKING / MATCHING ({pair_id}) ===")

    out_file = f"matches_{pair_id}.npz"
    matching(matcher=MATCHER,
             filter_method=FILTER,
             img1_path=img1,
             img2_path=img2,
             save_npz=True,
             unit_test=False,
             return_data=False,
             out_name=out_file,
             )
    matches_file = TEMP_DIR / out_file
    return matches_file


def stage_pose(matches_file, pair_id):
    """
    Stage 2: run pose estimation + triangulation
    using the matches saved by stage_tracking().
    pose_estimate() is assumed to:
      - load outputs/pose_estimation/matches.npz
      - compute R, t, K
      - save them to outputs/temp/pose.npz

    This is then immediately read pose.npz into memory.
    """
    print(f"\n=== STAGE 2: POSE ESTIMATION ({pair_id})===")

    out_file = TEMP_DIR / f"pose_{pair_id}.npz"

    pose_estimate(
        matches_file=str(matches_file),
        out_name=str(out_file)
    )

    pose_path = TEMP_DIR / f"pose_{pair_id}.npz"
    #   data = np.load(pose_path)
    """
    R = data["R"]
    t = data["t"]
    K = data["K"]"""
    print(f"[PIPE] Loaded pose from {pose_path}")
    return pose_path


def stage_triangulate(matches_file, pose_file, pair_id):
    print("\n=== STAGE 3: TRIANGULATION ===")

    out_file = TEMP_DIR / f"points_{pair_id}.npy"
    triangulate_from_files(matches_file=str(matches_file),
                           pose_file=str(pose_file),
                           out_file=str(out_file)
                           )
    points_file = out_file
    return points_file


def stage_align_pc(pose_files, points_files):
    print("\n=== STAGE 4: ALIGNING POINT CLOUDS ===")

    global_out = TEMP_DIR / "global_points.npy"
    align_point_clouds(
        pose_files=[str(p) for p in pose_files],
        point_files=[str(p) for p in points_files],
        output_name=str(global_out),
        save=True
    )
    return global_out


def stage_visualise(points_file):
    print("\n=== STAGE 5: VISUALISATION ===")
    visualize_points(points_file=str(points_file))


if __name__ == "__main__":

    image_files = load_image_files()
    pose_files = []
    point_files = []
    #   for i in range(len(image_files) - 1):
    for i in range(len(image_files) - 1):
        pair_id = f"{i:03d}"
        img1 = image_files[i]
        img2 = image_files[i + 1]

        matches_file = stage_tracking(img1, img2, pair_id)
        pose_file = stage_pose(matches_file, pair_id)
        points_file = stage_triangulate(matches_file, pose_file, pair_id)

        pose_files.append(pose_file)
        point_files.append(points_file)

    global_points_file = stage_align_pc(pose_files, point_files)
    #   stage_visualise(global_points_file)
    print("\n[PIPE] Done processing all image pairs.")
