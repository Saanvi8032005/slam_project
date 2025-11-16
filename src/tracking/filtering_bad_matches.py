import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

# --- Paths (adjust as needed) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "tracking"
OUT_ROOT = PROJECT_ROOT / "outputs" / "tracking"

# Pipeline images + folder
IMG1 = DATA / "left03.jpg"
IMG2 = DATA / "left04.jpg"
OUT_HIST = OUT_ROOT / "vectors"
OUT_RANSAC = OUT_ROOT / "RANSAC"


# ---------- Generic helpers ----------

def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_gray_pair(img1_path: Path, img2_path: Path):
    """Load a pair of grayscale images."""
    im1 = cv.imread(str(img1_path), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(img2_path), cv.IMREAD_GRAYSCALE)
    assert im1 is not None and im2 is not None, f"Check paths:\n{img1_path}\n{img2_path}"
    return im1, im2


def compute_orb_features(im1, im2, nfeatures=3000, fastThreshold=10):
    """Detect and compute ORB keypoints & descriptors for two images."""
    orb = cv.ORB_create(nfeatures=nfeatures, fastThreshold=fastThreshold)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    print(f"[ORB] keypoints: img1={len(kp1)}, img2={len(kp2)}")
    return kp1, des1, kp2, des2


def bf_ratio_match(des1, des2, ratio=0.75):
    """BFMatcher (Hamming) KNN + Lowe ratio test."""
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    print(f"[BF] matches after ratio test: {len(good)}")
    return good


def plot_and_save_matches(im1, kp1, im2, kp2, matches, title, save_path, max_draw=80):
    """Draw matches and save to disk."""
    vis = cv.drawMatches(
        im1, kp1, im2, kp2,
        sorted(matches, key=lambda x: x.distance)[:max_draw],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(vis, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ---------- Histogram-of-directions pipeline ----------

def histogram_direction_filter(kp1, kp2, matches, tol_deg=15.0, bins=36,
                               hist_save_path: Path | None = None,
                               after_vis_save_path: Path | None = None,
                               im1=None, im2=None):
    """Apply Selviah-style histogram filter on match directions."""

    if len(matches) == 0:
        print("[HIST] skipped (no good matches)")
        pts_empty = np.empty((0, 2), np.float32)
        return [], pts_empty, pts_empty

    pts1_all = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2_all = np.float32([kp2[m.trainIdx].pt for m in matches])
    deltas = pts2_all - pts1_all
    angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))  # [-180, 180)

    # Build histogram and find dominant motion direction
    hist, edges = np.histogram(angles, bins=bins, range=(-180, 180))
    peak_idx = int(np.argmax(hist))
    peak_angle = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

    # Keep matches within ±tol_deg of the dominant angle (circular distance)
    angle_diff = np.abs((angles - peak_angle + 180) % 360 - 180)
    keep_mask = angle_diff < tol_deg
    matches_in = [m for m, k in zip(matches, keep_mask) if k]

    print(f"[HIST] kept {int(keep_mask.sum())} / {len(matches)} "
          f"(peak ≈ {peak_angle:.1f}° ; tol=±{tol_deg}°)")

    # Plot histogram if requested
    if hist_save_path is not None:
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.figure(figsize=(7, 3))
        plt.bar(centers, hist, width=360 / bins, color='gray', edgecolor='black')
        plt.axvline(peak_angle, color='r', ls='--', label=f"peak {peak_angle:.1f}°")
        plt.xlabel("Motion direction (deg)")
        plt.ylabel("Count")
        plt.legend()
        plt.title("Histogram of match directions")
        plt.tight_layout()
        plt.savefig(hist_save_path, dpi=180)
        plt.close()

    # Visualise after filter if paths & images given
    if after_vis_save_path is not None and im1 is not None and im2 is not None:
        plot_and_save_matches(im1, kp1, im2, kp2, matches_in,
                              "After histogram filter (direction-consistent matches)",
                              after_vis_save_path)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches_in]) if matches_in else np.empty((0, 2), np.float32)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches_in]) if matches_in else np.empty((0, 2), np.float32)
    return matches_in, pts1, pts2


def run_histogram_pipeline(tol_deg=15.0, bins=36, ratio=0.75):
    """Full histogram-of-directions pipeline."""
    ensure_dir(OUT_HIST)

    # 1) Load images
    im1, im2 = load_gray_pair(IMG1, IMG2)

    # 2) ORB
    kp1, des1, kp2, des2 = compute_orb_features(im1, im2)

    # 3) BF + ratio
    good = bf_ratio_match(des1, des2, ratio=ratio)

    # 4) Visualise BEFORE histogram filter
    plot_and_save_matches(
        im1, kp1, im2, kp2, good,
        "Before histogram filter (ratio test only)",
        OUT_HIST / "01_before_hist.jpg"
    )

    # 5) Histogram-of-directions filter
    good_in, pts1, pts2 = histogram_direction_filter(
        kp1, kp2, good,
        tol_deg=tol_deg, bins=bins,
        hist_save_path=OUT_HIST / "02_hist_angles.png",
        after_vis_save_path=OUT_HIST / "03_after_hist.jpg",
        im1=im1, im2=im2
    )

    return dict(
        img1=im1, img2=im2,
        kp1=kp1, kp2=kp2,
        matches=good_in, pts1=pts1, pts2=pts2
    )


# ---------- RANSAC pipeline ----------

def ransac_fundamental_filter(kp1, kp2, matches, im1, im2, out_dir: Path):
    """Apply RANSAC on Fundamental matrix and visualise."""
    if len(matches) < 8:
        print("[RANSAC] skipped (not enough matches)")
        pts_empty = np.empty((0, 2), np.float32)
        return [], pts_empty, pts_empty

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv.findFundamentalMat(
        pts1, pts2, cv.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999
    )

    inliers = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), bool)
    good_in = [m for m, keep in zip(matches, inliers) if keep]
    print(f"[RANSAC] inliers: {inliers.sum()} / {len(matches)}  (ratio {inliers.mean():.2f})")

    # Visualise AFTER RANSAC
    plot_and_save_matches(
        im1, kp1, im2, kp2, good_in,
        "After RANSAC (geometrically consistent)",
        out_dir / "02_after_ransac.jpg"
    )

    pts1_final = np.float32([kp1[m.queryIdx].pt for m in good_in]) if good_in else np.empty((0, 2), np.float32)
    pts2_final = np.float32([kp2[m.trainIdx].pt for m in good_in]) if good_in else np.empty((0, 2), np.float32)
    return good_in, pts1_final, pts2_final


def run_ransac_pipeline(ratio=0.75):
    """Full RANSAC-on-Fundamental pipeline."""
    ensure_dir(OUT_RANSAC)

    # 1) Load images
    im1, im2 = load_gray_pair(IMG1, IMG2)

    # 2) ORB
    kp1, des1, kp2, des2 = compute_orb_features(im1, im2)

    # 3) BF + ratio
    good = bf_ratio_match(des1, des2, ratio=ratio)

    # 4) Visualise BEFORE RANSAC
    plot_and_save_matches(
        im1, kp1, im2, kp2, good,
        "Before RANSAC (ratio test only)",
        OUT_RANSAC / "01_before_ransac.jpg"
    )

    # 5) RANSAC filter
    good_in, pts1_final, pts2_final = ransac_fundamental_filter(
        kp1, kp2, good,
        im1, im2,
        OUT_RANSAC
    )

    return dict(
        img1=im1, img2=im2,
        kp1=kp1, kp2=kp2,
        matches=good_in,
        pts1=pts1_final, pts2=pts2_final
    )


# ---------- Main ----------

if __name__ == "__main__":
    print("=== Histogram-of-directions pipeline ===")
    out_hist = run_histogram_pipeline(tol_deg=15.0, bins=36, ratio=0.75)
    print(f"Final correspondences (hist-filtered): {len(out_hist['matches'])}")

    print("\n=== RANSAC pipeline ===")
    out_ransac = run_ransac_pipeline(ratio=0.75)
    print(f"Final correspondences for next stage (RANSAC inliers): {len(out_ransac['matches'])}")
