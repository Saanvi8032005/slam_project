
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_PATH = PROJECT_ROOT / "data" / "ground_truth" / "groundtruth.txt"

def read_trajectory(filename, matrix=True):
    """
    Read a trajectory from a text file. 
    
    Input:
    filename -- file to be read
    matrix -- convert poses to 4x4 matrices
    
    Output:
    dictionary of stamped 3D poses
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[float(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list_ok = []
    for i,l in enumerate(list):
        if l[4:8]==[0,0,0,0]:
            continue
        isnan = False
        for v in l:
            if numpy.isnan(v): 
                isnan = True
                break
        if isnan:
            sys.stderr.write("Warning: line %d of file '%s' has NaNs, skipping line\n"%(i,filename))
            continue
        list_ok.append(l)
    if matrix :
      traj = dict([(l[0],transform44(l[0:])) for l in list_ok])
    else:
      traj = dict([(l[0],l[1:8]) for l in list_ok])
    return traj


def load_tum_groundtruth(path):
    """
    Load TUM groundtruth file.
    Returns:
        stamps: np.array (N,) timestamps
        poses:  np.array (N, 7) columns [tx, ty, tz, qx, qy, qz, qw]
    """
    stamps = []
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            stamps.append(t)
            poses.append([tx, ty, tz, qx, qy, qz, qw])
    return np.array(stamps), np.array(poses)


def find_pose_for_timestamp(stamps, poses, t_query, max_diff=0.02):
    """
    Return pose (tx, ty, tz, qx, qy, qz, qw) for the nearest stamp to t_query.
    Raises if no match within max_diff seconds.
    """
    idx = np.argmin(np.abs(stamps - t_query))
    if abs(stamps[idx] - t_query) > max_diff:
        raise ValueError(f"No GT stamp within {max_diff}s of {t_query} (best diff={abs(stamps[idx]-t_query):.4f})")
    return poses[idx]


def quat_to_rot(qx, qy, qz, qw):
    """
    Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix.
    TUM format: qx qy qz qw.
    """
    q = np.array([qw, qx, qy, qz], dtype=np.float64)  # [w, x, y, z]
    q /= np.linalg.norm(q)
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def relative_pose_from_gt(gt1, gt2):
    """
    gt1, gt2: arrays [tx, ty, tz, qx, qy, qz, qw]
    Returns:
        R_gt, t_gt such that X2 = R_gt * X1 + t_gt
        (same convention as OpenCV recoverPose)
    """
    tx1, ty1, tz1, qx1, qy1, qz1, qw1 = gt1
    tx2, ty2, tz2, qx2, qy2, qz2, qw2 = gt2

    t_wc1 = np.array([tx1, ty1, tz1])
    t_wc2 = np.array([tx2, ty2, tz2])
    R_wc1 = quat_to_rot(qx1, qy1, qz1, qw1)
    R_wc2 = quat_to_rot(qx2, qy2, qz2, qw2)

    # From world→cam to cam1→cam2
    # X_w = R_wc1 X1 + t_wc1 = R_wc2 X2 + t_wc2
    # => X2 = R_wc2^T R_wc1 X1 + R_wc2^T (t_wc1 - t_wc2)
    R_gt = R_wc2.T @ R_wc1
    t_gt = R_wc2.T @ (t_wc1 - t_wc2)

    return R_gt, t_gt


def rotation_error_deg(R_est, R_gt):
    R_err = R_est @ R_gt.T
    tr = np.clip(np.trace(R_err), -1.0, 3.0)
    angle = np.arccos((tr - 1.0) / 2.0)
    return np.rad2deg(angle)


def translation_dir_error_deg(t_est, t_gt):
    t_est = np.asarray(t_est, dtype=np.float64).reshape(3)
    t_gt  = np.asarray(t_gt,  dtype=np.float64).reshape(3)

    t_est /= np.linalg.norm(t_est)
    t_gt  /= np.linalg.norm(t_gt)

    # account for sign ambiguity from recoverPose
    def angle(a, b):
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return np.rad2deg(np.arccos(dot))

    return min(angle(t_est, t_gt), angle(-t_est, t_gt))


if __name__ == "__main__":
    t_img1 = 1305031449.7996  # example – set to your actual first image timestamp
    t_img2 = 1305031449.8495  # example – set to your actual second image timestamp


    # 1) Load GT and get the two poses
    stamps, poses = load_tum_groundtruth(GT_PATH)
    gt1 = find_pose_for_timestamp(stamps, poses, t_img1)
    gt2 = find_pose_for_timestamp(stamps, poses, t_img2)

    R_gt, t_gt = relative_pose_from_gt(gt1, gt2)

    # 2) Either recompute, or plug your printed numbers:
    R_est = np.array([
        [-0.92820637, -0.30485674, -0.21329629],
        [-0.37015676,  0.69863695,  0.61228292],
        [-0.03764191,  0.64727797, -0.76132405],
    ])

    t_est = np.array([-0.44224359, -0.38860252,  0.80833699])

    # 3) Compare
    rot_err = rotation_error_deg(R_est, R_gt)
    t_err   = translation_dir_error_deg(t_est, t_gt)

    print(f"Rotation error:            {rot_err:.3f} deg")
    print(f"Translation direction err: {t_err:.3f} deg")