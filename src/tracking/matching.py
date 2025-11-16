import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

# --- Paths (adjust names) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tracking"
IMG1 = DATA_DIR / "left01.jpg"           # left / query
IMG2 = DATA_DIR / "left02.jpg"           # right / train


def matching():
    # --- Load grayscale images ---
    im1 = cv.imread(str(IMG1), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(IMG2), cv.IMREAD_GRAYSCALE)
    assert im1 is not None and im2 is not None

    # --- ORB: detect + describe ---
    orb = cv.ORB_create(
        nfeatures=4000,      # more features for indoor scenes
        fastThreshold=10,    # lower -> more keypoints in low contrast
        edgeThreshold=15
    )
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")

    # --- FLANN (LSH) for ORB's binary descriptors ---
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,      # 6–12
                        key_size=12,         # 10–20
                        multi_probe_level=1)  # 1–2
    search_params = dict(checks=50)

    try:
        flann = cv.FlannBasedMatcher(index_params, search_params)
        knn = flann.knnMatch(des1, des2, k=2)
    except cv.error:
        # Fallback: BFMatcher with Hamming (works for ORB)
        print("FLANN LSH not available; falling back to BFMatcher(Hamming).")
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)

    # --- Lowe ratio test ---
    ratio = 1
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    print(f"Matches after ratio test: {len(good)}")

    # --- Visualize good matches ---
    vis = cv.drawMatches(
        im1, kp1, im2, kp2,
        sorted(good, key=lambda x: x.distance)[:80],
        None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(vis, cmap='gray')
    plt.title("ORB + FLANN (LSH) good matches")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    matching()
