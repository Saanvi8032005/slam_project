"""
pipeline.py

Third script to run the individual stages in order:
1) Tracking / matching  (combined.py)
2) Pose estimation      (pose_estimation.py)
"""

from pathlib import Path
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tracking.tracking import matching
#   from tracking.lsd_tracking import lsd
from pose_estimation.pose_estimation import pose_estimate
from triangulation.triangulation import triangulate_from_data
from visualising.visualising import visualize_points
from aligning_pc.aligning_pc import align_point_clouds
from tests.pose_estimation_eval import (
        build_rgb_to_gt_pose_map,
        relative_pose_from_rgb_ts,
        rotation_error_deg,
        translation_direction_error_deg,
        GT_PATH,
    )

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


def stage_tracking(img1, img2, pair_id, tracking_results):
    """
    Stage 1: run feature detection + matching + filtering.
    Writes pts1, pts2 to an .npz file for pose_estimation to use.
    """
    MATCHER = "flann"
    FILTER = "hist"

    print(f"\n=== STAGE 1: TRACKING / MATCHING ({pair_id}) ===")

    out_file = f"matches_{pair_id}.npz"
    pts1, pts2, kp1, kp2, matches = matching(
             matcher=MATCHER,
             filter_method=FILTER,
             img1_path=img1,
             img2_path=img2,
             save_npz=False,
             unit_test=False,
             return_data=True,
             out_name=out_file,
             )

    #   pts1_line, pts2_line = lsd(img1, img2)
    pts1_all = np.vstack([
        np.float32(pts1).reshape(-1, 2),
        #   np.float32(pts1_line).reshape(-1, 2)
    ])

    pts2_all = np.vstack([
        np.float32(pts2).reshape(-1, 2),
        #   np.float32(pts2_line).reshape(-1, 2)
    ])

    entry = {
        "pair_id": pair_id,
        "img1": img1,
        "img2": img2,
        "pts1": pts1_all,
        "pts2": pts2_all,
        "kp1": kp1,
        "kp2": kp2,
        "matches": matches,
    }
    tracking_results[pair_id] = entry

    return entry


def stage_pose(tracking_entry, pose_store=None, img1=None, img2=None):
    pair_id = tracking_entry["pair_id"]
    pts1 = tracking_entry["pts1"]
    pts2 = tracking_entry["pts2"]

    print(f"\n=== STAGE 2: POSE ESTIMATION ({pair_id})===")
    R, t, K, maskPose = pose_estimate(pts1, pts2)

    if pose_store is not None:
        pose_store[pair_id] = {
            "pair_id": pair_id,
            "R": R,
            "t": t,
            "K": K,
            "mask": maskPose,
        }

    error_print = True
    if error_print:
        ts1 = float(img1.stem)
        ts2 = float(img2.stem)
        rgb_to_gt_pose = build_rgb_to_gt_pose_map(
                            GT_PATH,
                            image_files,
                            max_diff=0.02
                        )
        R_gt, t_gt = relative_pose_from_rgb_ts(rgb_to_gt_pose, ts1, ts2)

        # Errors
        rot_err = rotation_error_deg(R, R_gt)
        dir_err = translation_direction_error_deg(t, t_gt)
        print(f"[POSE][GT] Rotation error:            {rot_err:.3f} deg")
        print(f"[POSE][GT] Translation direction err: {dir_err:.3f} deg")

    return R, t, K, maskPose


def stage_triangulate(tracking_entry, pose_entry, points_store=None):
    """
    Stage 3: Triangulation from in-memory pts + pose.

    tracking_entry: dict from tracking_results[pair_id]
    pose_entry:     dict from pose_results[pair_id]
    points_store:   optional dict to accumulate 3D points per pair
    """
    print("\n=== STAGE 3: TRIANGULATION ===")

    pair_id = tracking_entry["pair_id"]

    pts1 = tracking_entry["pts1"]
    pts2 = tracking_entry["pts2"]
    R = pose_entry["R"]
    t = pose_entry["t"]
    K = pose_entry["K"]
    mask = pose_entry.get("mask", None)

    pts3D = triangulate_from_data(
        pts1,
        pts2,
        R,
        t,
        K,
        mask=mask,
        out_name=f"points_{pair_id}.npy",
    )
    if pts3D.shape[0] == 0:
        print(f"[PIPE] Skipping pair {pair_id} due to empty triangulation")
    if points_store is not None:
        points_store[pair_id] = {
            "pair_id": pair_id,
            "points": pts3D,
        }
    return pts3D


def stage_align_pc(pose_results, points_results):
    print("\n=== STAGE 4: ALIGNING POINT CLOUDS ===")

    align_point_clouds(
        pose_results,
        points_results,
        output_name="global_points.npy",
        save=False,
    )


def stage_visualise(points_file):
    print("\n=== STAGE 5: VISUALISATION ===")
    visualize_points(points_file=str(points_file))


if __name__ == "__main__":

    image_files = load_image_files()
    tracking_results = {}
    pose_results = {}
    points_results = {}

    #   (len(image_files) - 1):
    for i in range(24):
        print("\n" + "="*200)
        print(f"\n[PIPE] Processing image pair {i} and {i + 1}...")
        print(f"[PIPE] Image 1: {image_files[i]}")
        print(f"[PIPE] Image 2: {image_files[i + 1]}")

        pair_id = f"{i:03d}"
        img1 = image_files[i]
        img2 = image_files[i + 1]

        tracking_entry = stage_tracking(img1, img2, pair_id, tracking_results)
        R, t, K, maskPose = stage_pose(tracking_entry, pose_results, img1, img2)
        pose_entry = pose_results[pair_id]

        pts3D = stage_triangulate(tracking_entry,
                                  pose_entry,
                                  points_store=points_results)

    stage_align_pc(pose_results, points_results)
    #   stage_visualise(global_points_file)
    print("\n[PIPE] Done processing all image pairs.")
