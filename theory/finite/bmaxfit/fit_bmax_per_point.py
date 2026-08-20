"""Fit an independent b_max/a_H at every (condition, velocity) LAMMPS point.

Where `fit_bmax_to_lammps.py` (`theory/finite/lammps_fit/`) fits one shared
`b_max/a_H` per condition across all of its selected points, this script
asks whether that constant is really constant: for every selected point it
solves `b_max/a_H` in isolation, so the sole free parameter exactly
reproduces that one point's LAMMPS acceleration. Sweeping the result
against velocity is a direct check of the trend `fit_bmax_two_regimes.py`
(this directory) treats as a two-value step function -- if the per-point
best fit drifts smoothly with velocity rather than clustering into two flat
plateaus, that says the two-regime model is the wrong shape too, just a
better one than a single constant.

Root-find rather than least_squares: one point, one parameter, so the
residual is a single scalar, and its zero (in log(b_max/a_H), since b_max
spans decades) is exactly what `least_squares` would converge to anyway --
without the Jacobian/covariance machinery `fit_bmax_to_lammps.fit_condition`
needs for an over-determined fit. `scipy.optimize.brentq` needs a bracket
(a sign change): if the model cannot reach the data acceleration anywhere
in `[bmax_min, bmax_max]`, the point is reported unconverged at whichever
bound came closer, mirroring `fit_bmax_to_lammps`'s `at_upper_bound`
handling rather than silently reporting a boundary value as a real fit.

Parallel across (condition, point) tasks -- each task is a full,
independent root-find that runs entirely inside one worker process, so
there is no nested pool (contrast `fit_bmax_to_lammps`, which parallelizes
the drag evaluations *within* one shared least_squares fit: that pattern
doesn't help here since each per-point fit only takes a handful of cheap
1-D evaluations).

Run from repository root:
    python .\\theory\\finite\\bmaxfit\\fit_bmax_per_point.py --workers 8
"""

from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

if str(OUTDIR) not in sys.path:
    sys.path.insert(0, str(OUTDIR))

import common  # noqa: E402
from common import base  # noqa: E402

DEFAULT_POINTS_PER_CONDITION = 16

# log-residual sentinel used when the model acceleration is non-positive or
# non-finite (e.g. bmax so small the drag underflows to 0): a large
# one-sided value so the bracket check below correctly treats it as "model
# is far too small", never as an accidental sign match with the other bound.
_UNREACHABLE_LOG_RESIDUAL = -1.0e30

_BRENTQ_XTOL = 1.0e-10
_BRENTQ_RTOL = 1.0e-10


def _model_and_log_residual(
    condition: int, bmax_over_aH: float, point: common.DataPoint, method: str, resolution: int, vres: int
) -> tuple[float, float]:
    """Return ``(model_acceleration_cm_s2, log(model) - log(data))``."""
    drag = base.make_drag_for_fit(condition, bmax_over_aH, method, resolution, vres)
    force_n = base.quiet_drag(drag, point.velocity_cm_s * base.CM_PER_S_TO_M_PER_S)
    model_accel = abs(force_n / drag.ms) * 100.0
    if model_accel <= 0.0 or not math.isfinite(model_accel):
        return model_accel, _UNREACHABLE_LOG_RESIDUAL
    return model_accel, math.log(model_accel) - math.log(point.acceleration_cm_s2)


def solve_bmax_for_point(
    task: tuple[int, common.DataPoint, str, int, int, float, float],
) -> dict[str, object]:
    """Worker: root-find ``b_max/a_H`` for one (condition, point) pair."""
    condition, point, method, resolution, vres, bmax_min, bmax_max = task

    def objective(log_bmax: float) -> float:
        _, residual = _model_and_log_residual(condition, math.exp(log_bmax), point, method, resolution, vres)
        return residual

    log_min = math.log(bmax_min)
    log_max = math.log(bmax_max)
    resid_min = objective(log_min)
    resid_max = objective(log_max)

    at_lower_bound = False
    at_upper_bound = False
    if resid_min * resid_max < 0.0:
        best_log_bmax = brentq(objective, log_min, log_max, xtol=_BRENTQ_XTOL, rtol=_BRENTQ_RTOL, maxiter=200)
        best_bmax = math.exp(best_log_bmax)
        converged = True
    else:
        if abs(resid_min) <= abs(resid_max):
            best_bmax = bmax_min
            at_lower_bound = True
        else:
            best_bmax = bmax_max
            at_upper_bound = True
        # Only a genuine root if it landed exactly on a bound; otherwise the
        # model can't reach the data anywhere in [bmax_min, bmax_max].
        converged = resid_min == 0.0 or resid_max == 0.0

    model_accel, log_residual = _model_and_log_residual(condition, best_bmax, point, method, resolution, vres)

    a_H_m, debye_length_m = base.hydrogen_spacing_and_debye_length_m(condition)
    bmax_over_lD = math.nan
    if condition in common.WEAKLY_COUPLED_CONDITIONS:
        bmax_over_lD = best_bmax * a_H_m / debye_length_m

    return {
        "condition": condition,
        "velocity_cm_s": point.velocity_cm_s,
        "data_acceleration_cm_s2": point.acceleration_cm_s2,
        "data_acceleration_sigma_cm_s2": point.acceleration_sigma_cm_s2,
        "best_bmax_over_aH": best_bmax,
        "best_bmax_over_debye_length": bmax_over_lD,
        "model_acceleration_cm_s2": model_accel,
        "log_residual": log_residual,
        "converged": converged,
        "at_lower_bound": at_lower_bound,
        "at_upper_bound": at_upper_bound,
        "source": point.source,
        "method": method,
        "resolution_rhores_dphires": resolution,
        "vres": vres,
    }


def make_per_point_plot(condition: int, rows: list[dict[str, object]], all_points: list[common.DataPoint]) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))

    cond_all_v = np.array([p.velocity_cm_s for p in all_points if p.condition == condition])
    cond_all_a = np.array([p.acceleration_cm_s2 for p in all_points if p.condition == condition])
    axis.scatter(cond_all_v, cond_all_a, s=6, color="lightgray", label="LAMMPS points (not fit)", zorder=1)
    axis2 = axis.twinx()

    cond_rows = [row for row in rows if row["condition"] == condition]
    velocity = np.array([row["velocity_cm_s"] for row in cond_rows])
    bmax = np.array([row["best_bmax_over_aH"] for row in cond_rows])
    converged = np.array([bool(row["converged"]) for row in cond_rows])
    order = np.argsort(velocity)
    velocity, bmax, converged = velocity[order], bmax[order], converged[order]

    axis2.plot(
        velocity[converged], bmax[converged], "-o", color="tab:blue", markersize=5,
        label="per-point best fit", zorder=3,
    )
    if np.any(~converged):
        axis2.scatter(
            velocity[~converged], bmax[~converged], marker="x", color="tab:red", s=40,
            label="unconverged (bound-clamped)", zorder=4,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("velocity [cm/s]")
    axis.set_ylabel("acceleration [cm/s^2]", color="gray")
    axis2.set_ylabel("b_max / a_H (per-point best fit)", color="tab:blue")
    axis.set_title(f"Condition {condition}: per-point b_max fit vs. velocity")
    axis.grid(True, which="both", alpha=0.25)

    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = axis2.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(OUTDIR / f"condition_{condition}_bmax_per_point.png", dpi=200)
    plt.close(fig)


def make_combined_plot(rows: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(9, 6.5))
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}

    for condition in sorted({row["condition"] for row in rows}):
        cond_rows = [row for row in rows if row["condition"] == condition and row["converged"]]
        if not cond_rows:
            continue
        velocity = np.array([row["velocity_cm_s"] for row in cond_rows])
        bmax = np.array([row["best_bmax_over_aH"] for row in cond_rows])
        order = np.argsort(velocity)
        axis.plot(
            velocity[order], bmax[order], "-o", markersize=4,
            color=colors.get(condition), label=f"condition {condition}",
        )

    axis.set_xscale("log")
    axis.set_xlabel("velocity [cm/s]")
    axis.set_ylabel("b_max / a_H (per-point best fit)")
    axis.set_title("Per-point b_max fit vs. velocity, all conditions")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "bmax_per_point_all_conditions.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = common.build_common_parser(__doc__)
    parser.add_argument(
        "--points-per-condition", type=int, default=DEFAULT_POINTS_PER_CONDITION,
        help="quantile groups per condition before per-point solving (see fit_bmax_to_lammps.select_fit_points)",
    )
    args = parser.parse_args()

    all_points, filtered, conditions = common.load_and_filter_points(args)
    fit_points = base.select_fit_points(filtered, args.points_per_condition)
    fit_points = sorted(fit_points, key=lambda p: (p.condition, p.velocity_cm_s))
    print(
        f"Loaded {len(all_points)} raw points, {len(filtered)} after filtering, "
        f"{len(fit_points)} selected for per-point b_max solves across conditions {sorted(conditions)}.",
        flush=True,
    )

    tasks = [
        (point.condition, point, args.method, args.resolution, args.vres, args.bmax_min, args.bmax_max)
        for point in fit_points
    ]

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = base.run_pool_with_heartbeat(
            pool, tasks, solve_bmax_for_point,
            heartbeat_seconds=args.heartbeat_seconds, label="per-point fit", quiet=args.quiet,
        )

    n_unconverged = sum(1 for row in rows if not row["converged"])
    if n_unconverged:
        print(
            f"WARNING: {n_unconverged}/{len(rows)} points could not be bracketed in "
            f"[{args.bmax_min:g}, {args.bmax_max:g}] and were bound-clamped -- see the "
            "'converged'/'at_lower_bound'/'at_upper_bound' columns.",
            flush=True,
        )

    base.write_csv(OUTDIR / "bmax_per_point_fit.csv", rows)
    for condition in sorted(conditions):
        if any(row["condition"] == condition for row in rows):
            make_per_point_plot(condition, rows, all_points)
    make_combined_plot(rows)

    print(f"Finished in {(time.perf_counter()-start)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
