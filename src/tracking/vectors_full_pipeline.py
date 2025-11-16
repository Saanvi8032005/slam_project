import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- Paths (adjust as needed) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "tracking"
IMG1 = DATA / "left01.jpg"
IMG2 = DATA / "left02.jpg"
SAVE_HERE = PROJECT_ROOT / "outputs" / "tracking" / "vectors"


def simple_histogram_pipeline(tol_deg=15.0, bins=36, ratio=0.75):
    # 1) Load grayscale images
    im1 = cv.imread(str(IMG1), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(IMG2), cv.IMREAD_GRAYSCALE)
    assert im1 is not None and im2 is not None, f"Check paths:\n{IMG1}\n{IMG2}"

    # 2) ORB detect + describe
    orb = cv.ORB_create(nfeatures=3000, fastThreshold=10)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    print(f"[ORB] keypoints: img1={len(kp1)}, img2={len(kp2)}")

    # 3) BFMatcher (Hamming) + KNN ratio test (keep it simple)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    print(f"[BF] matches after ratio test: {len(good)}")

    # Visualize BEFORE histogram filter
    vis_before = cv.drawMatches(im1, kp1, im2, kp2,
                                sorted(good, key=lambda x: x.distance)[:80],
                                None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10,5)); plt.imshow(vis_before, cmap='gray')
    plt.title("Before histogram filter (ratio test only)"); plt.axis('off'); plt.tight_layout()
    plt.savefig(SAVE_HERE / "01_before_hist.jpg", dpi=160); plt.close()

    if len(good) == 0:
        print("[HIST] skipped (no good matches)")
        return dict(img1=im1, img2=im2, kp1=kp1, kp2=kp2, matches=[],
                    pts1=np.empty((0,2), np.float32), pts2=np.empty((0,2), np.float32))

    # 4) Histogram-of-direction filter (Selviah-style)
    pts1_all = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2_all = np.float32([kp2[m.trainIdx].pt for m in good])
    deltas = pts2_all - pts1_all
    angles = np.degrees(np.arctan2(deltas[:,1], deltas[:,0]))  # [-180, 180)

    # Build histogram and find dominant motion direction
    hist, edges = np.histogram(angles, bins=bins, range=(-180, 180))
    peak_idx = int(np.argmax(hist))
    peak_angle = 0.5 * (edges[peak_idx] + edges[peak_idx+1])

    # Keep matches within ±tol_deg of the dominant angle (circular distance)
    angle_diff = np.abs((angles - peak_angle + 180) % 360 - 180)
    keep_mask = angle_diff < tol_deg
    good_in = [m for m, k in zip(good, keep_mask) if k]

    print(f"[HIST] kept {int(keep_mask.sum())} / {len(good)} (peak ≈ {peak_angle:.1f}° ; tol=±{tol_deg}°)")

    # Plot and save histogram
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(7,3))
    plt.bar(centers, hist, width=360/bins, color='gray', edgecolor='black')
    plt.axvline(peak_angle, color='r', ls='--', label=f"peak {peak_angle:.1f}°")
    plt.xlabel("Motion direction (deg)"); plt.ylabel("Count"); plt.legend()
    plt.title("Histogram of match directions")
    plt.tight_layout(); plt.savefig(SAVE_HERE / "02_hist_angles.png", dpi=180); plt.close()

    # Visualize AFTER histogram filter
    vis_after = cv.drawMatches(im1, kp1, im2, kp2,
                               sorted(good_in, key=lambda x: x.distance)[:80],
                               None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10,5)); plt.imshow(vis_after, cmap='gray')
    plt.title("After histogram filter (direction-consistent matches)")
    plt.axis('off'); plt.tight_layout(); plt.grid()
    plt.savefig(SAVE_HERE / "03_after_hist.jpg", dpi=160); plt.close()

    # Return clean correspondences for the next stage
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_in]) if good_in else np.empty((0,2), np.float32)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_in]) if good_in else np.empty((0,2), np.float32)
    return dict(img1=im1, img2=im2, kp1=kp1, kp2=kp2, matches=good_in, pts1=pts1, pts2=pts2)


if __name__ == "__main__":
    out = simple_histogram_pipeline(tol_deg=15.0, bins=36, ratio=0.75)
    print(f"Final correspondences (hist-filtered): {len(out['matches'])}")
