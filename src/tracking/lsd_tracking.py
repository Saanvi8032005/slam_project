import cv2 as cv
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
#   import cv2.ximgproc as ximgproc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "rgb_dataset" / "rgb"

IMG1 = DATA_DIR / "1305031452.791720.png"
IMG2 = DATA_DIR / "1305031452.823674.png"

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
    print(f"[ORB] keypoints: img1={len(kp1)}, img2={len(kp2)}")
    return kp1, des1, kp2, des2

    """
    def detect_lines(gray):
    lines = lsd.detect(gray)  # shape (N, 1, 4) or (N, 4)
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    lines = lines.reshape(-1, 4).astype(np.float32)  # [x1, y1, x2, y2]
    return lines
    """


def find_midpoints(lines):
    midpoint = 0.5 * (lines[:, 0:2] + lines[:, 2:4])
    return midpoint


"""
def lsd_matching(img1_path=None,
                 img2_path=None,
                 out_name=None):

    print("=== Running LSD ===")

    # Load images
    im1 = cv.imread(str(img1_path), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(img2_path), cv.IMREAD_GRAYSCALE)

    lsd = cv.line_descriptor.LSDDetector.createLSDDetector()
    bd = cv.line_descriptor.BinaryDescriptor.createBinaryDescriptor()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # --- 2) Detect line segments ---
    keylines1 = lsd.detect(im1, scale=1, numOctaves=1, mask=None)
    keylines2 = lsd.detect(im2, scale=1, numOctaves=1, mask=None)
    print(f"Detected {len(keylines1)} lines in img1")
    print(f"Detected {len(keylines2)} lines in img2")
    if len(keylines1) == 0 or len(keylines2) == 0:
        print("[LSD] No lines detected in one of the images")
        return

    # --- 3) Compute LBD descriptors for those lines ---
    keylines1, descriptors1 = bd.compute(im1, keylines1)
    keylines2, descriptors2 = bd.compute(im2, keylines2)
    if descriptors1 is None or descriptors2 is None:
        print("[LBD] No descriptors computed")
        return
    print("Descriptors shape im1:", descriptors1.shape)
    print("Descriptors shape im2:", descriptors2.shape)

    matches = bf.match(descriptors1, descriptors2)
    #   matcher = cv.line_descriptor.BinaryDescriptorMatcher
    # .createBinaryDescriptorMatcher()
    #   matches = matcher.match(descriptors1, descriptors2)

    print(f"[LBD] descriptors: img1={len(descriptors1)}")
    print(f"[LBD] keylines: img1={len(keylines1)}")
    print(f"[LBD] showing {len(matches)} best matches")

    color1 = cv.cvtColor(im1, cv.COLOR_GRAY2BGR)
    color2 = cv.cvtColor(im2, cv.COLOR_GRAY2BGR)
    h1, w1 = color1.shape[:2]
    h2, w2 = color2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = color1
    canvas[:h2, w1:w1 + w2] = color2

    for m in matches:
        kl1 = keylines1[m.queryIdx]
        kl2 = keylines2[m.trainIdx]

        p1 = (int(kl1.startPointX), int(kl1.startPointY))
        q1 = (int(kl1.endPointX),   int(kl1.endPointY))

        p2 = (int(kl2.startPointX) + w1, int(kl2.startPointY))
        q2 = (int(kl2.endPointX)   + w1, int(kl2.endPointY))

        cv.line(canvas, p1, q1, (0, 255, 0), 1, cv.LINE_AA)
        cv.line(canvas, p2, q2, (0, 255, 0), 1, cv.LINE_AA)
    cv.imshow("LSD + LBD line matches", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return keylines1, keylines2, descriptors1, descriptors2, matches

"""


def filter_keylines(keylines, min_length=30):
    """Keep only reasonably long lines."""
    return [kl for kl in keylines if kl.lineLength >= min_length]


def draw_keylines(image, keylines, title="Detected Lines"):
    """Draws LSD KeyLines on an image using matplotlib."""
    img_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    for kl in keylines:
        pt1 = (int(kl.startPointX), int(kl.startPointY))
        pt2 = (int(kl.endPointX),   int(kl.endPointY))
        cv.line(img_color, pt1, pt2, (0, 255, 0), 1, cv.LINE_AA)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_color[..., ::-1])
    plt.title(title)
    plt.axis("off")
    plt.show()


def draw_line_matches(
        img1, img2, keylines1, keylines2, matches, title="Line Matches"):
    """Visualizes line matches by drawing segments and connecting midpoints."""

    # Convert to color
    img1_c = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_c = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    # Create a big canvas (side-by-side)
    h = max(img1_c.shape[0], img2_c.shape[0])
    w = img1_c.shape[1] + img2_c.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:img1_c.shape[0], :img1_c.shape[1]] = img1_c
    canvas[:img2_c.shape[0], img1_c.shape[1]:] = img2_c

    offset = img1_c.shape[1]

    # Plot each match
    for m in matches:
        kl1 = keylines1[m.queryIdx]
        kl2 = keylines2[m.trainIdx]

        p1a = (int(kl1.startPointX), int(kl1.startPointY))
        p1b = (int(kl1.endPointX),   int(kl1.endPointY))
        p2a = (int(kl2.startPointX + offset), int(kl2.startPointY))
        p2b = (int(kl2.endPointX + offset), int(kl2.endPointY))

        # Random colour for the pair
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw both line segments
        cv.line(canvas, p1a, p1b, color, 1)
        cv.line(canvas, p2a, p2b, color, 1)

        # Draw connecting midpoint line
        mid1 = (int((p1a[0]+p1b[0])//2), int((p1a[1]+p1b[1])//2))
        mid2 = (int((p2a[0]+p2b[0])//2), int((p2a[1]+p2b[1])//2))
        cv.line(canvas, mid1, mid2, color, 1, cv.LINE_AA)

    plt.figure(figsize=(12, 8))
    plt.imshow(canvas[..., ::-1])
    plt.title(title)
    plt.axis("off")
    plt.show()


def lsd(
        img1_path,
        img2_path,
        min_length=50,
        ratio_thresh=0.8,
        angle_thresh_deg=15.0,
        length_ratio_thresh=0.5,
):
    """
    LSD + LBD + BFMatcher with geometric filtering.

    Returns
    -------
    pts1, pts2 : (N, 2) float32
        Matched midpoints of line segments in img1 and img2.
    keylines1, keylines2 : list[KeyLine]
        Filtered KeyLines for each image.
    matches_final : list[cv.DMatch]
        Filtered matches between descriptors1 and descriptors2.
    """

    print("=== Running LSD+LBD matching ===")

    # --- 1) Load images ---
    im1 = cv.imread(str(img1_path), cv.IMREAD_GRAYSCALE)
    im2 = cv.imread(str(img2_path), cv.IMREAD_GRAYSCALE)
    if im1 is None or im2 is None:
        raise ValueError(f"Could not load images: {img1_path}, {img2_path}")
    #   im1 = cv.GaussianBlur(im1, (5,5), 1.0)
    #   clahe = cv.createCLAHE(clipLimit=3.0)
    #   im1 = clahe.apply(im1)
    im1 = cv.normalize(im1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    im1 = cv.GaussianBlur(im1, (3, 3), 0)
    im1 = cv.equalizeHist(im1)

    im2 = cv.normalize(im1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    im2 = cv.GaussianBlur(im1, (3, 3), 0)
    im2 = cv.equalizeHist(im1)

    # --- 2) Create detector / descriptor / matcher ---
    lsd = cv.line_descriptor.LSDDetector.createLSDDetector()
    bd = cv.line_descriptor.BinaryDescriptor.createBinaryDescriptor()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    # --- 3) Detect line segments --- not working very well
    keylines1 = lsd.detect(im1, scale=1, numOctaves=1, mask=None)
    keylines2 = lsd.detect(im2, scale=1, numOctaves=1, mask=None)

    if keylines1 is None or keylines2 is None:
        print("[LSD] No lines detected in one of the images")
        return None, None, [], [], []

    print(f"[LSD] raw lines: img1={len(keylines1)}, img2={len(keylines2)}")

    # Filter short lines (remove tiny noisy ones)
    keylines1 = filter_keylines(keylines1, min_length=min_length)
    keylines2 = filter_keylines(keylines2, min_length=min_length)
    print(f"[LSD] filtered lines (len>{min_length}): "
          f"img1={len(keylines1)}, img2={len(keylines2)}")

    if len(keylines1) == 0 or len(keylines2) == 0:
        print("[LSD] No sufficiently long lines after filtering")
        return None, None, [], [], []

    # --- 4) Compute LBD descriptors ---
    keylines1, desc1 = bd.compute(im1, keylines1)
    keylines2, desc2 = bd.compute(im2, keylines2)

    if desc1 is None or desc2 is None:
        print("[LBD] No descriptors computed")
        return None, None, [], [], []

    print(f"[LBD] descriptors: img1={desc1.shape}, img2={desc2.shape}")

    # --- 5) Descriptor matching: kNN + ratio test ---
    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    good_ratio = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_ratio.append(m)
    print(f"[MATCH] after ratio test ({ratio_thresh}): "
          f"{len(good_ratio)} / {len(knn_matches)}")

    # --- 6) Geometric filtering (orientation + length) ---
    matches_final = []
    for m in good_ratio:
        kl1 = keylines1[m.queryIdx]
        kl2 = keylines2[m.trainIdx]

        # orientation (KeyLine.angle is in radians)
        a1 = kl1.angle
        a2 = kl2.angle
        da = np.degrees((a1 - a2 + np.pi) % (2 * np.pi) - np.pi)
        if abs(da) > angle_thresh_deg:
            continue

        # length ratio
        L1 = kl1.lineLength
        L2 = kl2.lineLength
        length_ratio = min(L1, L2) / max(L1, L2)
        if length_ratio < length_ratio_thresh:
            continue

        matches_final.append(m)

    print(f"[MATCH] after geom filter: {len(matches_final)}")

    if len(matches_final) == 0:
        print("[MATCH] No matches survived geometric filtering")
        return None, None, keylines1, keylines2, []

    # --- 7) Build midpoint correspondences for SLAM ---
    pts1 = []
    pts2 = []
    for m in matches_final:
        kl1 = keylines1[m.queryIdx]
        kl2 = keylines2[m.trainIdx]

        # Endpoints in img1
        p1_start = (kl1.startPointX, kl1.startPointY)
        p1_end = (kl1.endPointX,   kl1.endPointY)

        # Endpoints in img2
        p2_start = (kl2.startPointX, kl2.startPointY)
        p2_end = (kl2.endPointX,   kl2.endPointY)

        pts1.append(p1_start)
        pts1.append(p1_end)
        pts2.append(p2_start)
        pts2.append(p2_end)

    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)

    print(f"[SLAM] returning {pts1.shape[0]} line-midpoint matches")

    InPipeline = False
    if InPipeline:
        return pts1, pts2
    else:
        return pts1, pts2, keylines1, keylines2, matches_final, im1, im2


if __name__ == "__main__":

    # ORB ALR SET
    MATCHER = "flann"        # "flann" or "bf"
    FILTER = "hist"          # "none", "hist", "ransac"
    """
    lsd_matching(
        img1_path=IMG1,
        img2_path=IMG2,
        out_name=None,
    )

    """
    ts1, pts2, keylines1, keylines2, matches_final, im1, im2 = lsd(IMG1, IMG2)
    draw_keylines(im1, keylines1, title="Image 1 - KeyLines")
    draw_keylines(im2, keylines2, title="Image 2 - KeyLines")

    draw_line_matches(
        im1, im2,
        keylines1,
        keylines2,
        matches_final,
        title="Matched Line Segments"
    )
