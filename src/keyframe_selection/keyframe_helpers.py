import numpy as np
#   from keyframe_selec import Keyframe


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3,)
    return T


def kps_to_xy(kps):
    return np.array([kp.pt for kp in kps], dtype=np.float32)
