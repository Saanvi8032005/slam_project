# src/backend/pose_graph_optim.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

try:
    from scipy.optimize import least_squares
except ImportError as e:
    raise ImportError("Install scipy: pip install scipy") from e


# ---------- Lie algebra helpers (SE3) ----------

def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=np.float64)

def vee(W: np.ndarray) -> np.ndarray:
    return np.array([W[2,1], W[0,2], W[1,0]], dtype=np.float64)

def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    W = hat(w / theta)
    return np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)

def so3_log(R: np.ndarray) -> np.ndarray:
    # clamp trace for numerical stability
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    w_hat = (R - R.T) / (2*np.sin(theta))
    return theta * vee(w_hat)

def se3_exp(xi: np.ndarray) -> np.ndarray:
    # xi = [w(3), v(3)]
    w = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(w)
    R = so3_exp(w)

    if theta < 1e-12:
        V = np.eye(3)
    else:
        W = hat(w)
        theta2 = theta * theta
        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta2 * W
            + (theta - np.sin(theta)) / (theta2 * theta) * (W @ W)
        )

    t = V @ v
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def se3_log(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]
    t = T[:3, 3]
    w = so3_log(R)
    theta = np.linalg.norm(w)

    if theta < 1e-12:
        V_inv = np.eye(3)
    else:
        W = hat(w)
        theta2 = theta * theta
        # inverse of V (closed form)
        V_inv = (
            np.eye(3)
            - 0.5 * W
            + (1/theta2) * (1 - (theta*np.sin(theta))/(2*(1-np.cos(theta)))) * (W @ W)
        )

    v = V_inv @ t
    return np.hstack([w, v])

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti


# ---------- Pose graph optimisation ----------

def optimise_pose_graph(slam_map, max_nfev: int = 50, robust: bool = True, verbose: int = 0) -> None:
    """
    Optimise keyframe poses T_cw using SE3 pose graph optimisation.
    Updates slam_map.keyframes[kf_id].T_cw in-place.

    Parameters
    ----------
    slam_map : Map
        The SLAM map containing keyframes and edges.
    max_nfev : int
        Maximum number of function evaluations for the optimizer.
    robust : bool
        Use robust loss function (Huber) if True.
    verbose : int
        Verbosity level for the optimizer (0 = silent, 2 = detailed).
    """
    if len(slam_map.keyframes) < 2 or len(slam_map.edges) < 1:
        print("[PGO] Nothing to optimise (need >=2 keyframes and >=1 edge).")
        return

    # Fixed anchor to remove gauge freedom (keep first keyframe fixed)
    kf_ids = sorted(slam_map.keyframes.keys())
    anchor_id = kf_ids[0]
    opt_ids = [kid for kid in kf_ids if kid != anchor_id]

    id_to_block = {kid: idx for idx, kid in enumerate(opt_ids)}  # pose block index

    # initial parameter vector: xi for each optimised pose, applied as left-mult update
    x0 = np.zeros(6 * len(opt_ids), dtype=np.float64)

    def pack_idx(kid: int) -> slice:
        j = id_to_block[kid]
        return slice(6*j, 6*j+6)

    def get_pose_from_x(kid: int, x: np.ndarray) -> np.ndarray:
        T0 = slam_map.keyframes[kid].T_cw
        if kid == anchor_id:
            return T0
        dT = se3_exp(x[pack_idx(kid)])
        return dT @ T0  # left-multiply update

    def residuals(x: np.ndarray) -> np.ndarray:
        res = []
        for e in slam_map.edges:
            Ti = get_pose_from_x(e.kf_i, x)
            Tj = get_pose_from_x(e.kf_j, x)

            T_pred = Tj @ inv_T(Ti)          # cam_i -> cam_j predicted
            T_err  = inv_T(e.T_ij) @ T_pred  # error transform
            r_ij = se3_log(T_err)            # 6D residual
            res.append(r_ij)
        return np.concatenate(res) if res else np.zeros(0)

    loss = "huber" if robust else "linear"
    f_scale = 1.0  # tweak if needed
    print(f"[PGO] Optimising poses: {len(slam_map.keyframes)} KFs, {len(slam_map.edges)} edges, anchor=KF{anchor_id}, loss={loss}")

    sol = least_squares(residuals, x0, loss=loss, f_scale=f_scale, max_nfev=max_nfev, verbose=verbose)

    # Apply solution to poses
    x_opt = sol.x
    for kid in opt_ids:
        slam_map.keyframes[kid].T_cw = get_pose_from_x(kid, x_opt)

    print(f"[PGO] Done. success={sol.success} cost={sol.cost:.3f} nfev={sol.nfev}")
