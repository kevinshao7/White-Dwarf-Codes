# Run from repository root:
#   python .\theory\finite\convergence\run_resolution_convergence.py --workers 8
"""Drag-force-vs-velocity resolution convergence for FiniteLaunchDrag, condition 0 only.

`FiniteLaunchDrag` (``method="vectorized"``) has three independent resolution
knobs: ``vres`` (midpoint rule over the relative-velocity Maxwellian inside
``drag``), ``rhores`` (log-spaced midpoint rule over the impact parameter `b`
inside `impact_parameter_integral`), and ``dphires`` (midpoint rule over the
regularised scattering-angle integral inside `orbit_angle`). This script
sweeps each one individually -- holding the other two fixed at the finest
resolution tested -- to see how the *predicted drag force itself* depends on
each grid's resolution, for condition 0 (T and density read directly off the
`FiniteLaunchDrag` instance so the plot labels can never drift from
`dragbase2.py`).

Only ``method="vectorized"`` is used -- no `scipy.integrate.quad` call is made
anywhere in this script.

Two plots are produced:

  1. ``condition_0_drag_vs_velocity.png`` -- drag force (N) vs bulk velocity
     (cm/s, log-spaced), log-log.
  2. ``condition_0_relative_error_vs_velocity.png`` -- relative error (log
     scale) vs the same velocity axis. The reference for each scan is that
     scan's own finest tested resolution (the largest value in
     ``--resolutions``, already computed as part of the sweep) -- there is no
     separate quad-based ground truth.

Each plot has up to 3 scans x len(resolutions) lines. Colour encodes
resolution (light -> dark as resolution increases), linestyle encodes which
parameter is being swept (solid = vres, dotted = rhores, dashed = dphires),
so both the per-scan convergence trend and the three scans are visible at a
glance.
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

CONDITION = 0
SCAN_TYPES = ("vres", "rhores", "dphires")
SCAN_LABELS = {
    "vres": "velocity resolution (vres)",
    "rhores": "impact-parameter resolution (rhores)",
    "dphires": "scattering-angle resolution (dphires)",
}
SCAN_LINESTYLES = {"vres": "-", "rhores": ":", "dphires": "--"}

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


def compute_force_task(task: tuple[str, int, int, int, float]) -> dict[str, object]:
    scan_type, resolution, hold_resolution, velocity_index, velocity_cm_s = task
    velocity_m_s = velocity_cm_s / CM_PER_M

    if scan_type == "vres":
        vres, rhores, dphires = resolution, hold_resolution, hold_resolution
    elif scan_type == "rhores":
        vres, rhores, dphires = hold_resolution, resolution, hold_resolution
    elif scan_type == "dphires":
        vres, rhores, dphires = hold_resolution, hold_resolution, resolution
    else:
        raise ValueError(f"unknown scan_type {scan_type!r}")

    drag = FiniteLaunchDrag(CONDITION, method="vectorized", vres=vres, rhores=rhores, dphires=dphires)
    force_n = drag.drag(velocity_m_s)

    return {
        "scan_type": scan_type,
        "resolution": resolution,
        "vres": vres,
        "rhores": rhores,
        "dphires": dphires,
        "velocity_index": velocity_index,
        "velocity_cm_s": velocity_cm_s,
        "velocity_m_s": velocity_m_s,
        "force_n": force_n,
    }


def add_relative_error(rows: list[dict[str, object]], hold_resolution: int) -> None:
    """Relative error vs. each scan's own finest tested resolution (in place)."""
    reference_force = {
        (row["scan_type"], row["velocity_index"]): row["force_n"]
        for row in rows
        if row["resolution"] == hold_resolution
    }
    for row in rows:
        ref = reference_force[(row["scan_type"], row["velocity_index"])]
        row["relative_error"] = abs(row["force_n"] - ref) / abs(ref) if ref != 0.0 else float("nan")


def _resolution_colors(resolutions: tuple[int, ...]):
    import matplotlib.pyplot as plt

    resolutions_sorted = sorted(resolutions)
    cmap = plt.get_cmap("viridis")
    n_res = len(resolutions_sorted)
    return {
        res: cmap(i / (n_res - 1) if n_res > 1 else 0.5)
        for i, res in enumerate(resolutions_sorted)
    }


def make_drag_force_plot(rows: list[dict[str, object]], resolutions: tuple[int, ...], condition_label: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = _resolution_colors(resolutions)

    for scan_type in SCAN_TYPES:
        for res in sorted(resolutions):
            curve = sorted(
                (row for row in rows if row["scan_type"] == scan_type and row["resolution"] == res),
                key=lambda row: float(row["velocity_cm_s"]),
            )
            v = np.array([float(row["velocity_cm_s"]) for row in curve])
            f = np.array([float(row["force_n"]) for row in curve])
            valid = np.isfinite(f) & (f > 0.0)
            ax.plot(
                v[valid],
                f[valid],
                color=colors[res],
                linestyle=SCAN_LINESTYLES[scan_type],
                linewidth=1.8,
                label=f"{SCAN_LABELS[scan_type]}, n={res}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bulk velocity $v_b$ (cm/s)")
    ax.set_ylabel("drag force $F$ (N)")
    ax.set_title(f"{condition_label}\ndrag force vs velocity (vectorized method)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(OUTDIR / "condition_0_drag_vs_velocity.png", dpi=200)
    plt.close(fig)


def make_relative_error_plot(
    rows: list[dict[str, object]], resolutions: tuple[int, ...], hold_resolution: int, condition_label: str
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = _resolution_colors(resolutions)

    for scan_type in SCAN_TYPES:
        for res in sorted(resolutions):
            if res == hold_resolution:
                # This is the reference resolution for its own scan -- error
                # is identically zero and cannot be shown on a log axis.
                continue
            curve = sorted(
                (row for row in rows if row["scan_type"] == scan_type and row["resolution"] == res),
                key=lambda row: float(row["velocity_cm_s"]),
            )
            v = np.array([float(row["velocity_cm_s"]) for row in curve])
            err = np.array([float(row["relative_error"]) for row in curve])
            valid = np.isfinite(err) & (err > 0.0)
            ax.plot(
                v[valid],
                err[valid],
                color=colors[res],
                linestyle=SCAN_LINESTYLES[scan_type],
                linewidth=1.8,
                label=f"{SCAN_LABELS[scan_type]}, n={res}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bulk velocity $v_b$ (cm/s)")
    ax.set_ylabel("relative error vs. finest tested resolution")
    ax.set_title(
        f"{condition_label}\nrelative error vs velocity (reference = finest tested resolution, n={hold_resolution})"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(OUTDIR / "condition_0_relative_error_vs_velocity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=list(DEFAULT_RESOLUTIONS),
        help="Resolution values, applied to whichever of vres/rhores/dphires is being swept. "
        "The two axes not being swept are held at the largest value in this list.",
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
        (scan_type, resolution, hold_resolution, velocity_index, float(velocity_cm_s))
        for scan_type in SCAN_TYPES
        for resolution in resolutions
        for velocity_index, velocity_cm_s in enumerate(velocities_cm_s)
    ]

    start = time.perf_counter()
    print(
        f"Running {len(tasks)} force evaluations for condition {CONDITION} across "
        f"{len(SCAN_TYPES)} scan types ({', '.join(SCAN_TYPES)}), {len(resolutions)} resolutions, "
        f"{args.n_velocities} velocities on up to {args.workers} workers "
        f"(method=vectorized only, no quad calls).",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = run_pool_with_heartbeat(
            pool, tasks, compute_force_task, heartbeat_seconds=args.heartbeat_seconds, label="force_vs_velocity"
        )

    rows.sort(key=lambda row: (row["scan_type"], row["resolution"], row["velocity_index"]))
    add_relative_error(rows, hold_resolution)
    write_csv(OUTDIR / "condition_0_resolution_scan.csv", rows)

    reference_drag = FiniteLaunchDrag(CONDITION, method="vectorized")
    condition_label = f"Condition {CONDITION} (T = {reference_drag.T:.0f} K, density = {reference_drag.gcc:.1e} g/cc)"

    make_drag_force_plot(rows, resolutions, condition_label)
    make_relative_error_plot(rows, resolutions, hold_resolution, condition_label)

    print(
        f"Wrote condition_0_drag_vs_velocity.png, condition_0_relative_error_vs_velocity.png, "
        f"and condition_0_resolution_scan.csv to {OUTDIR}.",
        flush=True,
    )
    print(f"Finished in {(time.perf_counter()-start):.1f}s.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
