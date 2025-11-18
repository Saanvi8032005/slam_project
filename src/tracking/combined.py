"""
Matching ORB features using FLANN/BF,
with optional histogram or RANSAC filtering.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tracking"
IMG1 = DATA_DIR / "left03.jpg"
IMG2 = DATA_DIR / "left04.jpg"
output_dir = PROJECT_ROOT / "outputs" / "tracking"
pose_estimation_output = PROJECT_ROOT / "outputs" / "pose_estimation"
os.makedirs(output_dir, exist_ok=True)


# -----------------------------------------------------------
# ORB FEATURE EXTRACTION
# -----------------------------------------------------------
def compute_orb_features(im1, im2):
    orb = cv.ORB_create(
        nfeatures=4000,
        fastThreshold=10,
        edgeThreshold=15
    )
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    print(f"[ORB] keypoints: img1={len(kp1)}, img2={len(kp2)}")
    return kp1, des1, kp2, des2


# -----------------------------------------------------------
# MATCHERS
# -----------------------------------------------------------
def match_with_flann(des1, des2):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    return flann.knnMatch(des1, des2, k=2)


def match_with_bf(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    return bf.knnMatch(des1, des2, k=2)


def match_descriptors(des1, des2, method="flann"):
    if method == "flann":
        try:
            return match_with_flann(des1, des2)
        except cv.error:
            print("FLANN failed; using BF instead.")
            return match_with_bf(des1, des2)
    elif method == "bf":
        return match_with_bf(des1, des2)
    else:
        raise ValueError(f"Unknown matcher: {method}")


# -----------------------------------------------------------
# HISTOGRAM FILTER
# -----------------------------------------------------------
def histogram_filter(kp1, kp2, matches, tol_deg=15.0, bins=36):
    if len(matches) == 0:
        print("[HIST] no matches")
        return []

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    deltas = pts2 - pts1
    angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))

    hist, edges = np.histogram(angles, bins=bins, range=(-180, 180))
    peak_idx = np.argmax(hist)
    peak_angle = 0.5 * (edges[peak_idx] + edges[peak_idx+1])

    angle_diff = np.abs((angles - peak_angle + 180) % 360 - 180)
    mask = angle_diff < tol_deg
    filtered = [m for m, k in zip(matches, mask) if k]

    print(f"[HIST] kept {len(filtered)} / {len(matches)}")
    return filtered


# -----------------------------------------------------------
# RANSAC FILTER
# -----------------------------------------------------------
def ransac_filter(kp1, kp2, matches):
    if len(matches) < 8:
        print("[RANSAC] not enough matches")
        return []

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv.findFundamentalMat(
        pts1, pts2, cv.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999
    )
    if mask is None:
        print("[RANSAC] failed")
        return []

    mask = mask.ravel().astype(bool)
    filtered = [m for m, keep in zip(matches, mask) if keep]
    print(f"[RANSAC] kept {len(filtered)} / {len(matches)}")
    return filtered


# -----------------------------------------------------------
# MATCHING PIPELINE WITH FILTER
# -----------------------------------------------------------
def matching(matcher="flann", filter_method="none", save_npz=False,
             return_data=False):
    msg = (
        f"\n=== Running with MATCHER={matcher.upper()} | "
        f"FILTER={filter_method.upper()} ==="
    )
    print(msg)

    # Load images
    im1 = cv.imread(str(IMG1), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(IMG2), cv.IMREAD_GRAYSCALE)

    kp1, des1, kp2, des2 = compute_orb_features(im1, im2)

    knn = match_descriptors(des1, des2, method=matcher)

    # Ratio test
    ratio = 0.75
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    print(f"[RATIO] {len(good)} matches")

    # Apply chosen filter
    if filter_method == "hist":
        good = histogram_filter(kp1, kp2, good)
    elif filter_method == "ransac":
        good = ransac_filter(kp1, kp2, good)
    elif filter_method == "none":
        print("[FILTER] None applied")

    # for pose estimation later
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    if save_npz:
        matches_path = pose_estimation_output / "matches_left03_left04.npz"
        np.savez(matches_path, pts1=pts1, pts2=pts2)
        print(f"[SAVE] Saved {len(pts1)} matches to {matches_path}")

    # Visualise
    vis = cv.drawMatches(
        im1, kp1, im2, kp2,
        sorted(good, key=lambda x: x.distance)[:80],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.imshow(vis, cmap='gray')
    plt.title(f"{matcher.upper()} + {filter_method.upper()}")
    plt.axis('off')
    """
    save_path = output_dir / f"matches_{matcher}_{filter_method}.jpg"
    plt.savefig(save_path, dpi=160)
    """
    plt.show()
    if return_data:
        return pts1, pts2, kp1, kp2, good


# -----------------------------------------------------------
# RUN HERE — CHANGE YOUR SETTINGS HERE
# -----------------------------------------------------------
if __name__ == "__main__":

    # ORB ALR SET
    MATCHER = "flann"        # "flann" or "bf"
    FILTER = "hist"          # "none", "hist", "ransac"

    matching(matcher=MATCHER, filter_method=FILTER)
