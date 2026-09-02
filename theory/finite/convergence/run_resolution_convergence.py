# Run from repository root:
#   python .\theory\finite\convergence\run_resolution_convergence.py
"""Finite-launch drag convergence tests using vectorized quadrature.

One 2-by-2 figure is produced: velocity, impact-parameter, and
scattering-angle resolution scans, plus the drag-force shape as bmax changes
in screening-length units.  Resolution errors are computed over a log-spaced
bulk-velocity grid; bmax errors are retained in the CSV.
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
    "vres": "velocity resolution",
    "rhores": "impact-parameter resolution",
    "dphires": "scattering-angle resolution",
}

# Production defaults: resolution scans change only their named grid.  The
# bmax scan uses all three values unchanged.
DEFAULT_VRES = 100
DEFAULT_RHORES = 300
DEFAULT_DPHIRES = 300
DEFAULT_RESOLUTIONS = (30, 100, 300, 1000)
DEFAULT_BMAX_OVER_SCREENING_LENGTH = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
BMAX_ERROR_REFERENCE = 1000.0
DEFAULT_N_VELOCITIES = 16
DEFAULT_VELOCITY_MIN_CM_S = 1.0e5
DEFAULT_VELOCITY_MAX_CM_S = 1.0e8
DEFAULT_WORKERS = 24
CM_PER_M = 100.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def default_resolution() -> dict[str, int]:
    return {"vres": DEFAULT_VRES, "rhores": DEFAULT_RHORES, "dphires": DEFAULT_DPHIRES}


def bmax_to_rhomax_fraction(bmax_over_screening_length: float) -> float:
    """Convert bmax/lambda_S to the launch radius in a_H units.

    ``FiniteLaunchDrag`` sets bmax equal to its launch radius.  Its
    ``rhomax_fraction`` is this radius in hydrogen interparticle spacings;
    lambda_S/a_H is ``ustart/k0``.
    """
    baseline = FiniteLaunchDrag(CONDITION, method="vectorized", **default_resolution())
    return bmax_over_screening_length * baseline.ustart / baseline.k0


def compute_force_task(task: tuple[str, float, int, float]) -> dict[str, object]:
    test_type, test_value, velocity_index, velocity_cm_s = task
    resolution = default_resolution()
    rhomax_fraction = 1.0
    if test_type in SCAN_TYPES:
        resolution[test_type] = int(test_value)
    elif test_type == "bmax_over_screening_length":
        rhomax_fraction = bmax_to_rhomax_fraction(test_value)
    else:
        raise ValueError(f"unknown test_type {test_type!r}")

    drag = FiniteLaunchDrag(
        CONDITION,
        method="vectorized",
        rhomax_fraction=rhomax_fraction,
        **resolution,
    )
    force_n = drag.drag(velocity_cm_s / CM_PER_M)
    return {
        "test_type": test_type,
        "test_value": test_value,
        **resolution,
        "rhomax_fraction": rhomax_fraction,
        "velocity_index": velocity_index,
        "velocity_cm_s": velocity_cm_s,
        "force_n": force_n,
    }


def add_relative_errors(
    rows: list[dict[str, object]], resolutions: tuple[int, ...], bmax_values: tuple[float, ...]
) -> None:
    """Use finest resolution and bmax/lambda_S=1000 as their references."""
    references = {
        (row["test_type"], row["velocity_index"]): row["force_n"]
        for row in rows
        if (
            row["test_type"] in SCAN_TYPES and int(row["test_value"]) == max(resolutions)
        )
        or (
            row["test_type"] == "bmax_over_screening_length"
            and float(row["test_value"]) == BMAX_ERROR_REFERENCE
        )
    }
    for row in rows:
        reference = references[(row["test_type"], row["velocity_index"])]
        row["relative_error"] = abs(row["force_n"] - reference) / abs(reference) if reference else float("nan")


def rainbow_colors(values: tuple[float, ...]) -> dict[float, object]:
    """Assign one distinct rainbow colour to every resolution/cutoff value."""
    import matplotlib.pyplot as plt

    values = tuple(sorted(values))
    return dict(zip(values, plt.colormaps["turbo"](np.linspace(0.05, 0.95, len(values)))))


def plot_error_curves(
    axis,
    rows: list[dict[str, object]],
    test_type: str,
    values: tuple[float, ...],
    label_prefix: str,
) -> None:
    reference = max(values)
    colors = rainbow_colors(values)
    for value in sorted(values):
        if value == reference:
            continue  # Reference error is zero and cannot be drawn on a log axis.
        curve = sorted(
            (row for row in rows if row["test_type"] == test_type and float(row["test_value"]) == value),
            key=lambda row: float(row["velocity_cm_s"]),
        )
        velocity = np.array([float(row["velocity_cm_s"]) for row in curve])
        error = np.array([float(row["relative_error"]) for row in curve])
        valid = np.isfinite(error) & (error > 0.0)
        axis.plot(
            velocity[valid],
            error[valid],
            color=colors[value],
            linestyle="solid",
            linewidth=1.8,
            label=f"{label_prefix}={value:g}",
        )


def make_convergence_figure(
    rows: list[dict[str, object]], resolutions: tuple[int, ...], bmax_values: tuple[float, ...], condition_label: str
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for axis, scan_type in zip(axes.flat[:3], SCAN_TYPES):
        plot_error_curves(
            axis,
            rows,
            scan_type,
            tuple(float(value) for value in resolutions),
            "n",
        )
        axis.set_title(f"{SCAN_LABELS[scan_type]} scan")
        axis.legend(fontsize=8, title="resolution")

    bmax_axis = axes.flat[3]
    bmax_colors = rainbow_colors(bmax_values)
    for bmax in bmax_values:
        curve = sorted(
            (
                row
                for row in rows
                if row["test_type"] == "bmax_over_screening_length" and float(row["test_value"]) == bmax
            ),
            key=lambda row: float(row["velocity_cm_s"]),
        )
        velocity = np.array([float(row["velocity_cm_s"]) for row in curve])
        force = np.array([float(row["force_n"]) for row in curve])
        valid = np.isfinite(force) & (force > 0.0)
        bmax_axis.plot(
            velocity[valid],
            force[valid],
            color=bmax_colors[bmax],
            linewidth=1.8,
            label=rf"$b_{{max}}/\lambda_S={bmax:g}$",
        )
    bmax_axis.set_title(r"drag-force shape as $b_{max}$ increases")
    bmax_axis.legend(fontsize=8, title=r"$b_{max}/\lambda_S$")

    for axis in axes.flat[:3]:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("bulk velocity $v_b$ (cm/s)")
        axis.set_ylabel("relative error in drag force")
        axis.grid(True, which="both", alpha=0.25)

    bmax_axis.set_xscale("log")
    bmax_axis.set_yscale("log")
    bmax_axis.set_xlabel("bulk velocity $v_b$ (cm/s)")
    bmax_axis.set_ylabel("drag force $F$ (N)")
    bmax_axis.grid(True, which="both", alpha=0.25)

    fig.suptitle(f"{condition_label}\nresolution convergence and bmax shape dependence (vectorized method)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTDIR / "condition_0_convergence.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument(
        "--bmax-over-screening-length",
        nargs="+",
        type=float,
        default=list(DEFAULT_BMAX_OVER_SCREENING_LENGTH),
        help="bmax/lambda_S values; 1000 is required as the error reference.",
    )
    parser.add_argument("--n-velocities", type=int, default=DEFAULT_N_VELOCITIES)
    parser.add_argument("--velocity-min-cm-s", type=float, default=DEFAULT_VELOCITY_MIN_CM_S)
    parser.add_argument("--velocity-max-cm-s", type=float, default=DEFAULT_VELOCITY_MAX_CM_S)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--heartbeat-seconds", type=float, default=12.0)
    args = parser.parse_args()

    resolutions = tuple(sorted(set(args.resolutions)))
    bmax_values = tuple(sorted(set(args.bmax_over_screening_length)))
    if not resolutions or min(resolutions) < 1:
        parser.error("--resolutions must contain positive integers")
    if not bmax_values or min(bmax_values) <= 0.0 or BMAX_ERROR_REFERENCE not in bmax_values:
        parser.error("--bmax-over-screening-length must contain positive values including 1000")

    velocities_cm_s = np.logspace(math.log10(args.velocity_min_cm_s), math.log10(args.velocity_max_cm_s), args.n_velocities)
    tasks = [
        (scan_type, float(resolution), velocity_index, float(velocity_cm_s))
        for scan_type in SCAN_TYPES
        for resolution in resolutions
        for velocity_index, velocity_cm_s in enumerate(velocities_cm_s)
    ]
    tasks.extend(
        ("bmax_over_screening_length", bmax, velocity_index, float(velocity_cm_s))
        for bmax in bmax_values
        for velocity_index, velocity_cm_s in enumerate(velocities_cm_s)
    )

    start = time.perf_counter()
    print(
        f"Running {len(tasks)} force evaluations for condition {CONDITION}: {len(SCAN_TYPES)} resolution scans and "
        f"{len(bmax_values)} bmax/lambda_S values on up to {args.workers} workers.",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = run_pool_with_heartbeat(pool, tasks, compute_force_task, heartbeat_seconds=args.heartbeat_seconds, label="convergence")

    rows.sort(key=lambda row: (str(row["test_type"]), float(row["test_value"]), int(row["velocity_index"])))
    add_relative_errors(rows, resolutions, bmax_values)
    write_csv(OUTDIR / "condition_0_convergence.csv", rows)
    reference_drag = FiniteLaunchDrag(CONDITION, method="vectorized", **default_resolution())
    label = f"Condition {CONDITION} (T = {reference_drag.T:.0f} K, density = {reference_drag.gcc:.1e} g/cc)"
    make_convergence_figure(rows, resolutions, bmax_values, label)
    print(f"Wrote condition_0_convergence.png and condition_0_convergence.csv to {OUTDIR}.", flush=True)
    print(f"Finished in {(time.perf_counter() - start):.1f}s.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
