"""
scale_check.py

Drop this file into your project (e.g. src/eval/scale_check.py) and call
`run_scale_check(slam_map, gt_path, out_dir)` at the end of your pipeline run.

Purpose:
- Extract estimated camera centres from your Keyframe Tcw poses
- Load TUM ground-truth positions (tx,ty,tz)
- Compare per-step motion statistics (mean/median step length)
- Estimate the global scale factor (EST units -> meters)
- Plot step-length histograms + optional top-down trajectory plot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ScaleCheckResult:
    scale_est_median: float
    scale_est_mean: float
    est_step_stats: Tuple[float, float, float, float]  # mean, median, min, max
    gt_step_stats: Tuple[float, float, float, float]   # mean, median, min, max
    n_est: int
    n_gt: int
    note: str


def _camera_centres_from_Tcw(Tcw_list: Iterable[np.ndarray]) -> np.ndarray:
    """
    Convert a list of Tcw (world->camera) poses into camera centres C (in world frame):
        x_c = R x_w + t
        C_w = -R^T t
    """
    centres: List[np.ndarray] = []
    for Tcw in Tcw_list:
        if Tcw is None:
            continue
        Tcw = np.asarray(Tcw)
        if Tcw.shape != (4, 4):
            raise ValueError(f"Tcw must be 4x4, got {Tcw.shape}")
        R = Tcw[:3, :3]
        t = Tcw[:3, 3]
        C = -R.T @ t
        centres.append(C)
    if len(centres) < 2:
        raise ValueError("Need at least 2 poses to compute step lengths.")
    return np.vstack(centres)  # (N,3)


def _step_lengths(centres_xyz: np.ndarray) -> np.ndarray:
    """Euclidean distances between consecutive camera centres."""
    d = centres_xyz[1:] - centres_xyz[:-1]
    return np.linalg.norm(d, axis=1)


def _load_tum_xyz(gt_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load TUM groundtruth in format:
        timestamp tx ty tz qx qy qz qw
    Returns:
        stamps (N,), xyz (N,3)
    """
    gt_path = Path(gt_path)
    stamps: List[float] = []
    xyz: List[List[float]] = []
    with gt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            stamps.append(float(parts[0]))
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(xyz) < 2:
        raise ValueError(f"Groundtruth file has too few poses: {gt_path}")
    return np.asarray(stamps), np.asarray(xyz)


def _stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    return float(np.mean(x)), float(np.median(x)), float(np.min(x)), float(np.max(x))


def run_scale_check(
    slam_map,
    gt_path: str | Path,
    out_dir: str | Path = "outputs/eval",
    *,
    plot: bool = True,
    topdown_axes: Tuple[int, int] = (0, 2),  # plot X vs Z by default
) -> ScaleCheckResult:
    """
    Main entry point to call from your pipeline.

    Expects:
      slam_map.keyframes: dict-like, values have attribute `.Tcw` (4x4 world->camera)

    Produces:
      - prints step stats + estimated scale
      - saves plots to out_dir (if plot=True)

    Returns:
      ScaleCheckResult (also useful for logging in progress reports)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Extract estimated poses in a stable order ----
    # If your map is keyed by kf_id, this gives consistent ordering.
    try:
        kfs_sorted = [slam_map.keyframes[k] for k in sorted(slam_map.keyframes.keys())]
    except Exception:
        # fallback: just iterate values
        kfs_sorted = list(slam_map.keyframes.values())

    Tcw_list = [kf.Tcw for kf in kfs_sorted if getattr(kf, "Tcw", None) is not None]
    centres_est = _camera_centres_from_Tcw(Tcw_list)
    steps_est = _step_lengths(centres_est)

    # ---- 2) Load GT and compute GT step lengths ----
    _, centres_gt = _load_tum_xyz(gt_path)
    steps_gt = _step_lengths(centres_gt)

    # ---- 3) Estimate global scale (EST units -> meters) ----
    # This is a diagnostic: if constant ~0.1, your "1 unit step" corresponds to 0.1m.
    # Use robust median and also mean.
    scale_median = float(np.median(steps_gt) / max(np.median(steps_est), 1e-12))
    scale_mean = float(np.mean(steps_gt) / max(np.mean(steps_est), 1e-12))

    est_stats = _stats(steps_est)
    gt_stats = _stats(steps_gt)

    # ---- 4) Print an evaluation summary ----
    print("\n================ SCALE CHECK ================")
    print(f"[EST] steps (units): mean={est_stats[0]:.6f}, median={est_stats[1]:.6f}, "
          f"min={est_stats[2]:.6f}, max={est_stats[3]:.6f}, n={len(steps_est)}")
    print(f"[GT ] steps (m):     mean={gt_stats[0]:.6f}, median={gt_stats[1]:.6f}, "
          f"min={gt_stats[2]:.6f}, max={gt_stats[3]:.6f}, n={len(steps_gt)}")
    print(f"[SCALE] estimated (median) EST->m: {scale_median:.6f}")
    print(f"[SCALE] estimated (mean)   EST->m: {scale_mean:.6f}")

    note = (
        "If the scale estimate is roughly constant across runs/sequences, your trajectory "
        "shape is likely correct and the main issue is monocular scale ambiguity. "
        "If you later compute scale over time windows and it varies strongly, that indicates "
        "scale drift or unstable translation-direction estimates."
    )
    print(f"[NOTE] {note}")
    print("============================================\n")

    # ---- 5) Optional plots ----
    if plot:
        # Histogram: step lengths
        plt.figure()
        plt.hist(steps_est, bins=50)
        plt.xlabel("Estimated step length (arbitrary units)")
        plt.ylabel("Count")
        plt.title("Estimated per-step translation magnitudes")
        plt.savefig(out_dir / "est_step_hist.png", dpi=120)
        plt.close()

        plt.figure()
        plt.hist(steps_gt, bins=50)
        plt.xlabel("Ground truth step length (meters)")
        plt.ylabel("Count")
        plt.title("Ground truth per-step translation magnitudes")
        plt.savefig(out_dir / "gt_step_hist.png", dpi=120)
        plt.close()

        # Top-down trajectory plot: EST (scaled) vs GT (no timestamp alignment here, just shape)
        ax0, ax1 = topdown_axes
        centres_est_scaled = centres_est * scale_median

        plt.figure()
        plt.plot(centres_gt[:, ax0], centres_gt[:, ax1], label="GT")
        plt.plot(centres_est_scaled[:, ax0], centres_est_scaled[:, ax1], label="EST (scaled)")
        plt.xlabel(["X", "Y", "Z"][ax0])
        plt.ylabel(["X", "Y", "Z"][ax1])
        plt.title("Top-down trajectory (GT vs EST scaled)")
        plt.axis("equal")
        plt.legend()
        plt.savefig(out_dir / "traj_topdown_gt_vs_est_scaled.png", dpi=120)
        plt.close()

        # Save raw arrays for debugging
        np.savetxt(out_dir / "centres_est.txt", centres_est)
        np.savetxt(out_dir / "centres_est_scaled.txt", centres_est_scaled)
        np.savetxt(out_dir / "centres_gt.txt", centres_gt)

        print(f"[SAVED] Plots and arrays saved to: {out_dir.resolve()}")

    return ScaleCheckResult(
        scale_est_median=scale_median,
        scale_est_mean=scale_mean,
        est_step_stats=est_stats,
        gt_step_stats=gt_stats,
        n_est=len(centres_est),
        n_gt=len(centres_gt),
        note=note,
    )


# -----------------------------
# Optional: run standalone
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Path to TUM groundtruth.txt")
    parser.add_argument("--out", default="outputs/eval", help="Output directory for plots")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    # Standalone mode requires you to wire in your slam_map object.
    # Usually you'll call run_scale_check(...) from inside your pipeline instead.
    raise SystemExit(
        "Standalone mode: import this file and call run_scale_check(slam_map, gt_path, out_dir).\n"
        "Example (in pipeline.py):\n"
        "  from eval.scale_check import run_scale_check\n"
        "  run_scale_check(slam_map, 'data/.../groundtruth.txt', out_dir='outputs/eval')\n"
    )
