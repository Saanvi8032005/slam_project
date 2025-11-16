"""
Matching ORB features between two images using FLANN or BFMatcher.
"""

import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tracking"
IMG1 = DATA_DIR / "left03.jpg"
IMG2 = DATA_DIR / "left04.jpg"
output_dir = PROJECT_ROOT / "outputs" / "tracking"
os.makedirs(output_dir, exist_ok=True)


def compute_orb_features(im1, im2):
    orb = cv.ORB_create(
        nfeatures=4000,
        fastThreshold=10,
        edgeThreshold=15
    )
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")
    return kp1, des1, kp2, des2


def match_with_flann(des1, des2):
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,
        key_size=12,
        multi_probe_level=1
    )
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)
    return knn


def match_with_bf(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    return knn


def match_descriptors(des1, des2, method="flann"):
    """Choose between FLANN and BF."""
    if method == "flann":
        try:
            return match_with_flann(des1, des2)
        except cv.error:
            print("FLANN failed; falling back to BFMatcher(Hamming).")
            return match_with_bf(des1, des2)
    elif method == "bf":
        return match_with_bf(des1, des2)
    else:
        raise ValueError(f"Unknown matching method: {method}")


def matching(method="flann"):
    im1 = cv.imread(str(IMG1), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(IMG2), cv.IMREAD_GRAYSCALE)
    assert im1 is not None and im2 is not None

    kp1, des1, kp2, des2 = compute_orb_features(im1, im2)

    knn = match_descriptors(des1, des2, method=method)

    ratio = 0.75
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    print(f"Matches after ratio test: {len(good)}")

    vis = cv.drawMatches(
        im1, kp1, im2, kp2,
        sorted(good, key=lambda x: x.distance)[:80],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.imshow(vis, cmap='gray')
    plt.savefig(output_dir / f"ORB + {method.upper()}.jpg", dpi=160)
    plt.title(f"ORB + {method.upper()} good matches")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # change here: "flann" or "bf"
    matching(method="flann")
