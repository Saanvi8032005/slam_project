import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- Paths (adjust as needed) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "tracking"
IMG1 = DATA / "left01.jpg"
IMG2 = DATA / "left02.jpg"
SAVE_HERE = PROJECT_ROOT / "outputs" / "tracking" / "RANSAC"


def simple_pipeline():
    # 1) Load grayscale images
    im1 = cv.imread(str(IMG1), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(IMG2), cv.IMREAD_GRAYSCALE)
    assert im1 is not None and im2 is not None, f"Check paths:\n{IMG1}\n{IMG2}"

    # 2) ORB detect + describe (tiny bit tuned for indoor/low contrast)
    orb = cv.ORB_create(nfeatures=3000, fastThreshold=10)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    print(f"[ORB] keypoints: img1={len(kp1)}, img2={len(kp2)}")

    # 3) BFMatcher (Hamming) + KNN ratio test (no FLANN to keep it simple)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    ratio = 0.75
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    print(f"[BF] matches after ratio test: {len(good)}")

    # Visualize BEFORE RANSAC
    vis_before = cv.drawMatches(
        im1, kp1, im2, kp2,
        sorted(good, key=lambda x: x.distance)[:80],
        None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 5))
    plt.imshow(vis_before, cmap='gray')
    plt.title("Before RANSAC (ratio test only)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(SAVE_HERE / "01_before_ransac.jpg", dpi=160)
    plt.close()

    # 4) RANSAC on Fundamental matrix (no intrinsics needed)
    if len(good) >= 8:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.999)
        inliers = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), bool)
        good_in = [m for m, keep in zip(good, inliers) if keep]
        print(f"[RANSAC] inliers: {inliers.sum()} / {len(good)}  (ratio {inliers.mean():.2f})")
    else:
        good_in, pts1, pts2 = [], [], []
        print("[RANSAC] skipped (not enough matches)")

    # Visualize AFTER RANSAC
    vis_after = cv.drawMatches(
        im1, kp1, im2, kp2,
        sorted(good_in, key=lambda x: x.distance)[:80],
        None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 5))
    plt.imshow(vis_after, cmap='gray')
    plt.title("After RANSAC (geometrically consistent)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(SAVE_HERE / "02_after_ransac.jpg", dpi=160)
    plt.close()

    # Return clean correspondences for pose/triangulation next
    pts1_final = np.float32([kp1[m.queryIdx].pt for m in good_in]) if len(good_in) else np.empty((0, 2), np.float32)
    pts2_final = np.float32([kp2[m.trainIdx].pt for m in good_in]) if len(good_in) else np.empty((0, 2), np.float32)
    return dict(img1=im1, img2=im2, kp1=kp1, kp2=kp2, matches=good_in,
                pts1=pts1_final, pts2=pts2_final)


if __name__ == "__main__":
    out = simple_pipeline()
    print(f"Final correspondences for next stage: {len(out['matches'])}")
