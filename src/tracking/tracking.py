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
DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset" / "rgb"

# IMG1 = DATA_DIR / "1305031452.791720.png"
IMG1 = DATA_DIR / "1305031452.823674.png"
IMG2 = DATA_DIR / "1305031452.859642.png"

output_dir = PROJECT_ROOT / "outputs" / "tracking"
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
    if des1 is None or des2 is None:
        print("[MATCH] No descriptors in one of the images")
        return []

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
    filtered = [m for m, keep in zip(matches, mask) if keep]

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
        pts1,
        pts2,
        cv.FM_RANSAC,
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
def matching(matcher="flann",
             filter_method="none",
             img1_path=None,
             img2_path=None,
             save_npz=None,
             unit_test=None,
             return_data=None,
             out_name=None):
    """
    Runs feature extraction, matching, and optional filtering.

    Returns
    -------
    pts1 : (N, 2) float32
        Matched points in image 1.
    pts2 : (N, 2) float32
        Matched points in image 2.
    kp1, kp2 : list[cv.KeyPoint]
        Keypoints for each image.
    matches : list[cv.DMatch]
        Final filtered matches (indices refer to kp1/kp2).
    """

    msg = (
        f"\n=== Running with MATCHER={matcher.upper()} | "
        f"FILTER={filter_method.upper()} ==="
    )
    print(msg)

    # Load images
    im1 = cv.imread(str(img1_path), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(img2_path), cv.IMREAD_GRAYSCALE)

    if im1 is None or im2 is None:
        raise ValueError(f"Could not load images: {img1_path}, {img2_path}")

    kp1, des1, kp2, des2 = compute_orb_features(im1, im2)

    knn = match_descriptors(des1, des2, method=matcher)
    # is this needed? do i need if good is empty too
    if len(knn) == 0:
        print("[MATCH] No matches found")
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            kp1,
            kp2,
            [],
        )

    # Ratio test
    ratio = 0.8   # was 0.75
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    print(f"[RATIO] {len(good)} matches")

    # Apply chosen filter
    if filter_method == "hist":
        good = histogram_filter(kp1, kp2, good)
    elif filter_method == "ransac":
        good = ransac_filter(kp1, kp2, good)
    elif filter_method == "none":
        print("[FILTER] None applied")
    else:
        raise ValueError(f"Unknown filter method: {filter_method}")

    if len(good) == 0:
        print("[MATCH] No matches after filtering")
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            kp1,
            kp2,
            [],
        )

    # for pose estimation later
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    use_lk = True
    if use_lk:
        print("[LK] Filtering matches using Lucas–Kanade optical flow")

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # LK expects shape (N,1,2)
        pts1_fwd = pts1.reshape(-1, 1, 2)

        # Forward: im1 -> im2
        pts2_fwd, st1, err1 = cv.calcOpticalFlowPyrLK(
            im1, im2, pts1_fwd, None, **lk_params
        )

        # Backward: im2 -> im1
        pts1_back, st2, err2 = cv.calcOpticalFlowPyrLK(
            im2, im1, pts2_fwd, None, **lk_params
        )

        st1 = st1.reshape(-1)
        st2 = st2.reshape(-1)
        pts2_fwd_flat = pts2_fwd.reshape(-1, 2)
        pts1_back_flat = pts1_back.reshape(-1, 2)
        err1 = err1.reshape(-1)
        err2 = err2.reshape(-1)

        # Forward–backward error (how far we drift if we go 1->2->1)
        fb_err = np.linalg.norm(pts1_back_flat - pts1, axis=1)

        # Thresholds (can tune these)
        fb_thresh = 1.0    # max allowed FB error in pixels
        err_thresh = 50.0  # max allowed LK photometric error
        max_disp = 25.0    # optional: reject crazy jumps vs descriptor match

        disp = np.linalg.norm(pts2_fwd_flat - pts2, axis=1)

        valid = (
            (st1 == 1)
            & (st2 == 1)
            & (fb_err < fb_thresh)
            & (err1 < err_thresh)
            & (err2 < err_thresh)
            & (disp < max_disp)
        )

        print(f"[LK] Valid after FB + error check: {
            valid.sum()} / {len(valid)}")

        if valid.sum() == 0:
            print("[LK] No valid tracks after LK filtering")
            return (
                np.empty((0, 2), np.float32),
                np.empty((0, 2), np.float32),
                kp1,
                kp2,
                [],
            )

        # Filter points and matches
        pts1 = pts1[valid]
        pts2 = pts2_fwd_flat[valid]
        good = [m for m, keep in zip(good, valid) if keep]
        idx_i = np.array([m.queryIdx for m in good], dtype=int)  # Indices in image 1 (kp1/des1)
        idx_j = np.array([m.trainIdx for m in good], dtype=int)  # Indices in image 2 (kp2/des2)

    if unit_test:
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
        if save_npz:
            save_path = output_dir / f"matches_{matcher}_{filter_method}.jpg"
            plt.savefig(save_path, dpi=160)
        plt.show()

    if return_data:
        return pts1, pts2, kp1, kp2, good, idx_i, idx_j, des1, des2


# -----------------------------------------------------------
# RUN HERE — CHANGE YOUR SETTINGS HERE
# -----------------------------------------------------------
if __name__ == "__main__":

    # ORB ALR SET
    MATCHER = "bf"        # "flann" or "bf"
    FILTER = "hist"          # "none", "hist", "ransac"

    matching(
        matcher=MATCHER,
        filter_method=FILTER,
        img1_path=IMG1,
        img2_path=IMG2,
        save_npz=True,
        unit_test=True,
        return_data=False,
        out_name=None,
    )
