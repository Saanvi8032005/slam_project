import numpy as np
from scipy.spatial import cKDTree


def best_fit_transform(A, B):
    """
    Computes the least-squares rigid transform T that aligns A → B.
    Returns R, t.
    """
    assert A.shape == B.shape

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Remove centroids
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance via SVD
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # translation
    t = centroid_B.reshape(3,1) - R @ centroid_A.reshape(3,1)

    return R, t


def icp(A, B, init_R=None, init_t=None, max_iters=30, tolerance=1e-5):
    """
    A: source (frame i)  Nx3
    B: target (frame i+1) Nx3

    init_R, init_t = initial pose estimation from recoverPose

    Returns refined_R, refined_t
    """

    src = A.copy()
    dst = B.copy()

    # Apply initial guess if provided
    if init_R is not None:
        src = (init_R @ src.T).T
        if init_t is not None:
            src += init_t.reshape(1,3)

    prev_error = np.inf

    tree = cKDTree(dst)

    R_total = init_R if init_R is not None else np.eye(3)
    t_total = init_t if init_t is not None else np.zeros((3,1))

    for i in range(max_iters):

        # 1) Nearest neighbours
        distances, indices = tree.query(src)
        matched = dst[indices]

        # 2) Compute best transform src → matched
        R_delta, t_delta = best_fit_transform(src, matched)

        # 3) Apply incremental transform
        src = (R_delta @ src.T).T + t_delta.reshape(1,3)

        # Update global transform
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta

        # 4) Convergence check
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return R_total, t_total
