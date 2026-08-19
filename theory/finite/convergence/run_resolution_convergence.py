# Run from repository root:
#   python .\theory\finite\convergence\run_resolution_convergence.py --workers 8
"""Drag-force-vs-velocity resolution convergence for FiniteLaunchDrag.

`FiniteLaunchDrag.impact_parameter_integral` (called once per velocity node
inside `drag`) nests two numerical grids under the ``"vectorized"`` method:
a log-spaced midpoint rule over the impact parameter `b` (`rhores` nodes) and
a midpoint rule over the regularised scattering-angle integral `orbit_angle`
(`dphires` nodes). This script checks how the *predicted drag force itself*
depends on each grid's resolution, by sweeping one axis at a time while
holding the other fixed at the finest tested resolution:

  "rhores"  scan: sweep the impact-parameter resolution, dphires held fixed
  "dphires" scan: sweep the scattering-angle resolution, rhores held fixed

Both scans use ONLY ``method="vectorized"`` -- no `scipy.integrate.quad` call
is made anywhere in this script (unlike earlier versions of this file, which
used `quad` for a ground-truth reference and a timing benchmark).

For each of 4 conditions, one plot is produced: drag force (N) on the y-axis
against bulk velocity (cm/s, log-spaced) on the x-axis. Each plot has 8
lines -- 4 resolutions x 2 scans. Colour encodes resolution (light -> dark as
resolution increases), linestyle encodes scan type (solid = impact-parameter
scan, dotted = scattering-angle scan), so the two scan types and the
convergence trend within each are both visible at a glance.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import numpy as np

THEORY_DIR = Path(__file__).resolve().parents[2]
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from finite.finite_launch import FiniteLaunchDrag  # noqa: E402
from finite.progress import run_pool_with_heartbeat  # noqa: E402

DEFAULT_CONDITIONS = (0, 1, 2, 3)
# Same 4-value, x4-spaced list the earlier version of this script used for
# its resolution scans.
DEFAULT_RESOLUTIONS = (45, 180, 720, 2880)
DEFAULT_N_VELOCITIES = 16
DEFAULT_VELOCITY_MIN_CM_S = 1.0e5
DEFAULT_VELOCITY_MAX_CM_S = 1.0e8
CM_PER_M = 100.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compute_force_task(task: tuple[int, float, str, int, int]) -> dict[str, object]:
    condition, velocity_cm_s, scan_type, resolution, hold_resolution = task
    velocity_m_s = velocity_cm_s / CM_PER_M

    if scan_type == "rhores":
        rhores, dphires = resolution, hold_resolution
    elif scan_type == "dphires":
        rhores, dphires = hold_resolution, resolution
    else:
        raise ValueError(f"unknown scan_type {scan_type!r}")

    drag = FiniteLaunchDrag(condition, method="vectorized", rhores=rhores, dphires=dphires)
    force_n = drag.drag(velocity_m_s)

    return {
        "condition": condition,
        "scan_type": scan_type,
        "resolution": resolution,
        "rhores": rhores,
        "dphires": dphires,
        "velocity_cm_s": velocity_cm_s,
        "velocity_m_s": velocity_m_s,
        "force_n": force_n,
    }


def make_force_velocity_plot(rows: list[dict[str, object]], condition: int, resolutions: tuple[int, ...]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    resolutions_sorted = sorted(resolutions)
    cmap = plt.get_cmap("viridis")
    n_res = len(resolutions_sorted)
    colors = {
        res: cmap(i / (n_res - 1) if n_res > 1 else 0.5)
        for i, res in enumerate(resolutions_sorted)
    }
    linestyles = {"rhores": "-", "dphires": ":"}
    scan_labels = {
        "rhores": "impact-parameter scan (rhores)",
        "dphires": "scattering-angle scan (dphires)",
    }

    for scan_type in ("rhores", "dphires"):
        for res in resolutions_sorted:
            curve = sorted(
                (
                    row
                    for row in rows
                    if row["condition"] == condition
                    and row["scan_type"] == scan_type
                    and row["resolution"] == res
                ),
                key=lambda row: float(row["velocity_cm_s"]),
            )
            v = np.array([float(row["velocity_cm_s"]) for row in curve])
            f = np.array([float(row["force_n"]) for row in curve])
            valid = np.isfinite(f) & (f > 0.0)
            ax.plot(
                v[valid],
                f[valid],
                color=colors[res],
                linestyle=linestyles[scan_type],
                linewidth=1.8,
                label=f"{scan_labels[scan_type]}, n={res}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bulk velocity $v_b$ (cm/s)")
    ax.set_ylabel("drag force $F$ (N)")
    ax.set_title(f"Condition {condition}: drag force vs velocity (vectorized method only)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"condition_{condition}_force_vs_velocity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", type=int, default=list(DEFAULT_CONDITIONS))
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=list(DEFAULT_RESOLUTIONS),
        help="Resolution values (rhores for the impact-parameter scan, dphires for the "
        "scattering-angle scan). The axis not being swept is held at the largest value "
        "in this list.",
    )
    parser.add_argument("--n-velocities", type=int, default=DEFAULT_N_VELOCITIES)
    parser.add_argument("--velocity-min-cm-s", type=float, default=DEFAULT_VELOCITY_MIN_CM_S)
    parser.add_argument("--velocity-max-cm-s", type=float, default=DEFAULT_VELOCITY_MAX_CM_S)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=12.0,
        help="Print a status line at least this often even if no task has finished yet.",
    )
    args = parser.parse_args()

    resolutions = tuple(sorted(args.resolutions))
    hold_resolution = max(resolutions)
    velocities_cm_s = np.logspace(
        math.log10(args.velocity_min_cm_s),
        math.log10(args.velocity_max_cm_s),
        args.n_velocities,
    )

    tasks = [
        (condition, float(velocity_cm_s), scan_type, resolution, hold_resolution)
        for condition in args.conditions
        for scan_type in ("rhores", "dphires")
        for resolution in resolutions
        for velocity_cm_s in velocities_cm_s
    ]

    start = time.perf_counter()
    print(
        f"Running {len(tasks)} force evaluations across {len(args.conditions)} conditions, "
        f"2 scan types, {len(resolutions)} resolutions, {args.n_velocities} velocities "
        f"on up to {args.workers} workers (method=vectorized only, no quad calls).",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = run_pool_with_heartbeat(
            pool, tasks, compute_force_task, heartbeat_seconds=args.heartbeat_seconds, label="force_vs_velocity"
        )

    rows.sort(key=lambda row: (row["condition"], row["scan_type"], row["resolution"], row["velocity_cm_s"]))
    write_csv(OUTDIR / "force_vs_velocity.csv", rows)

    covered_conditions = sorted({row["condition"] for row in rows})
    for condition in covered_conditions:
        make_force_velocity_plot(rows, condition, resolutions)

    print(
        f"Wrote {len(covered_conditions)} plots (condition_*_force_vs_velocity.png) and "
        f"force_vs_velocity.csv to {OUTDIR}.",
        flush=True,
    )
    print(f"Finished in {(time.perf_counter()-start):.1f}s.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
