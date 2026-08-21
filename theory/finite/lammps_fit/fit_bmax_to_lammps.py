# Run from repository root:
#   python .\theory\finite\lammps_fit\fit_bmax_to_lammps.py --workers 8
"""Fit the impact-parameter cutoff b_max/a_H to LAMMPS drag simulations.

Single free parameter per condition: `rhomax_fraction = b_max / a_H = r_i /
a_H`, with the launch radius itself `r_i = rhomax_fraction * a_H` (`a_H` the
hydrogen interparticle spacing) -- this is a deliberate simplification of the
old `theory/validation/impactparameterfit/fit_bmax_to_lammps.py`, which fit
`b_max` against a separately-tunable launch radius `r_i = 50 a_H` (that
script's `cutoff_radius_factor`). In the finite-launch model implemented here
(`theory/finite/finite_launch.py`), there is no separate scattering-angle
cutoff to tie to `b_max`: the deflection is integrated exactly from launch to
closest approach, so `b_max` alone bounds both integrals by construction.
`FiniteLaunchDrag.launch_pmax` forces `b_max == r_i` always (a tangent
launch), so `rhomax_fraction` moves both together rather than being bounded
above by a separately-fixed `r_i` -- the fit's upper bound defaults to
infinity (`DEFAULT_BMAX_MAX`); a best fit that still runs away unbounded is a
sign that even an arbitrarily distant launch sphere cannot reach the data
(the model saturates as `r_i -> infinity`, since the Yukawa screening kills
the contribution from very large impact parameters), not a bug -- report it,
don't hide it.

Point selection is likewise simplified from the old script's per-campaign
regex grouping: points are ranked by velocity, split into
`--points-per-condition` roughly-equal quantile groups, and the
lowest-relative-sigma point is kept from each group. This approximates
log-spacing in velocity without assuming anything about how the source LAMMPS
runs were organized.

Depends only on `theory/finite/`, `theory/dragbase2.py`, and
`theory/dataprocessing/output{,_dais}/results.npy` -- nothing under
`theory/validation/`.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

THEORY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from finite.finite_launch import DEFAULT_METHOD, FiniteLaunchDrag  # noqa: E402
from finite.progress import run_pool_with_heartbeat  # noqa: E402
from dragbase2 import DragFourth  # noqa: E402

CM_PER_S_TO_M_PER_S = 1.0e-2
ALL_CONDITIONS = (0, 1, 2, 3)
DAIS_CONDITIONS = (0, 2)
# gccarr = [1e-5, 1, 1e-5, 1] in DragFourth: conditions 0 and 2 are the
# weakly-coupled (low mass-density) cases, so only for those does the
# electron Debye length set the relevant screening scale for b_max.
WEAKLY_COUPLED_CONDITIONS = (0, 2)
DEFAULT_BMAX_MIN = 1.0e-2
# b_max is tied to r_i (FiniteLaunchDrag.launch_pmax: b_max == r_i always), so
# there is no b_max/a_H <= 1 ceiling to enforce -- fitting rhomax_fraction
# unbounded above just moves the whole launch sphere outward.
DEFAULT_BMAX_MAX = math.inf
DEFAULT_POINTS_PER_CONDITION = 8
# rhores = dphires = 360 keeps the 'vectorized' scheme within ~9e-4 relative
# error of the quad_quad reference (worst case 8.7e-4) across the conditions
# and speeds in theory/finite/convergence/resolution_convergence.csv -- the
# next tested resolution down, 180, exceeds the 1e-3 target (3.5e-3 at
# condition 2, v=1e4). vres is halved from the 201 used to generate that
# convergence data; its own convergence has not been separately checked
# (the convergence script never sweeps vres). Re-run
# theory/finite/convergence/run_resolution_convergence.py and adjust these if
# you change conditions, velocity range, or need tighter accuracy.
DEFAULT_RESOLUTION = 360
DEFAULT_VRES = 101


@dataclass(frozen=True)
class DataPoint:
    condition: int
    velocity_cm_s: float
    acceleration_cm_s2: float
    velocity_sigma_cm_s: float
    acceleration_sigma_cm_s2: float
    campaign_id: int
    source: str


def positive_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(result) or result <= 0.0:
        return math.nan
    return result


def exp_velocity(time_s: np.ndarray, tau: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-time_s / tau)


def load_expfit_points(results_path: Path, conditions: set[int], samples_per_fit: int) -> list[DataPoint]:
    """Expand each LAMMPS exponential velocity-decay fit into synthetic
    (velocity, acceleration) samples along its decay curve.

    `results.npy` is shaped (condition, campaign, 6):
    [amplitude, amplitude_sigma, tau, tau_sigma, start_time, end_time], the
    parameters of v(t) = amplitude * exp(-t/tau) fit to one LAMMPS run.
    """
    data = np.load(results_path, allow_pickle=False)
    points: list[DataPoint] = []
    if data.ndim != 3 or data.shape[2] < 6:
        raise ValueError(f"Expected results array shaped (condition, campaign, 6), got {data.shape}")

    for condition in sorted(conditions):
        if condition >= data.shape[0]:
            continue
        for campaign_id, row in enumerate(data[condition]):
            amplitude = positive_float(row[0])
            amplitude_sigma = positive_float(row[1])
            tau = positive_float(row[2])
            tau_sigma = positive_float(row[3])
            start_time = positive_float(row[4])
            end_time = positive_float(row[5])
            if not all(np.isfinite(v) for v in (amplitude, amplitude_sigma, tau, tau_sigma, start_time, end_time)):
                continue

            times = np.linspace(start_time, end_time, samples_per_fit)
            velocities = exp_velocity(times, tau, amplitude)
            accelerations = velocities / tau
            velocity_sigma = np.sqrt(
                np.square(velocities * amplitude_sigma / amplitude)
                + np.square(times * velocities * tau_sigma / np.square(tau))
            )
            acceleration_sigma = accelerations * np.sqrt(
                np.square(velocity_sigma / velocities) + np.square(tau_sigma / tau)
            )

            for sample_index, (velocity, acceleration, v_err, a_err) in enumerate(
                zip(velocities, accelerations, velocity_sigma, acceleration_sigma)
            ):
                velocity = positive_float(velocity)
                acceleration = positive_float(acceleration)
                if not np.isfinite(velocity) or not np.isfinite(acceleration):
                    continue
                points.append(
                    DataPoint(
                        condition=condition,
                        velocity_cm_s=velocity,
                        acceleration_cm_s2=acceleration,
                        velocity_sigma_cm_s=positive_float(v_err),
                        acceleration_sigma_cm_s2=positive_float(abs(a_err)),
                        campaign_id=campaign_id,
                        source=f"{results_path.relative_to(REPO_ROOT)}[{condition},{campaign_id},{sample_index}]",
                    )
                )
    return points


def load_all_points(lammps_results: Path, dais_results: Path, conditions: set[int], samples_per_fit: int) -> list[DataPoint]:
    dais_conditions = conditions.intersection(DAIS_CONDITIONS)
    lammps_conditions = conditions.difference(DAIS_CONDITIONS)
    points: list[DataPoint] = []
    if lammps_conditions:
        points.extend(load_expfit_points(lammps_results, lammps_conditions, samples_per_fit))
    if dais_conditions:
        points.extend(load_expfit_points(dais_results, dais_conditions, samples_per_fit))
    return points


def filter_points(points: list[DataPoint], min_v: float, max_v: float, max_relative_sigma: float) -> list[DataPoint]:
    filtered = []
    for point in points:
        if not (min_v <= point.velocity_cm_s <= max_v):
            continue
        if np.isfinite(point.acceleration_sigma_cm_s2):
            if point.acceleration_sigma_cm_s2 / point.acceleration_cm_s2 > max_relative_sigma:
                continue
        filtered.append(point)
    return filtered


def relative_sigma(point: DataPoint) -> float:
    if not np.isfinite(point.acceleration_sigma_cm_s2):
        return math.inf
    return point.acceleration_sigma_cm_s2 / point.acceleration_cm_s2


def select_fit_points(points: list[DataPoint], points_per_condition: int) -> list[DataPoint]:
    """Rank by velocity, split into `points_per_condition` quantile groups,
    keep the lowest-relative-sigma point from each group.

    Approximates "log-spaced in velocity, most precise available" without
    assuming anything about how the source LAMMPS campaigns were organized
    (contrast the old script's regex-parsed per-campaign grouping).
    """
    selected: list[DataPoint] = []
    for condition in sorted({point.condition for point in points}):
        group = sorted((p for p in points if p.condition == condition), key=lambda p: p.velocity_cm_s)
        if len(group) <= points_per_condition:
            selected.extend(group)
            continue
        for chunk in np.array_split(np.array(group, dtype=object), points_per_condition):
            if len(chunk) == 0:
                continue
            best = min(chunk, key=relative_sigma)
            selected.append(best)
    return selected


def quiet_drag(drag: FiniteLaunchDrag, velocity_m_s: float) -> float:
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(drag.drag(velocity_m_s))


def make_drag_for_fit(condition: int, bmax_over_aH: float, method: str, resolution: int, vres: int) -> FiniteLaunchDrag:
    return FiniteLaunchDrag(
        condition,
        method=method,
        rhomax_fraction=bmax_over_aH,
        rhores=resolution,
        dphires=resolution,
        vres=vres,
    )


def parse_gpu_devices(raw: str | None) -> list[int] | None:
    """Parse ``--gpu-devices`` (e.g. ``"0,1"``) into a device-id list.

    ``None``/empty means CPU dispatch via the existing ``ProcessPoolExecutor``
    path -- the unchanged default; nothing about GPU dispatch is exercised
    unless this is explicitly set.
    """
    if not raw:
        return None
    return [int(item) for item in raw.split(",") if item.strip()]


_SCALE_CACHE: dict[int, tuple[float, float]] = {}


def hydrogen_spacing_and_debye_length_m(condition: int) -> tuple[float, float]:
    """Return (a_H, electron Debye length) in meters for a condition.

    Both are fixed algebraic quantities set in `DragFourth.__init__`
    (`theory/dragbase2.py`), independent of the fit's `rhomax_fraction` --
    cache per condition instead of instantiating on every call.
    """
    cached = _SCALE_CACHE.get(condition)
    if cached is None:
        base = DragFourth(condition)
        a_H = 1.0 / float(base.ustart)
        debye_length = float(base.lD)
        cached = (a_H, debye_length)
        _SCALE_CACHE[condition] = cached
    return cached


def _fit_point_row(
    condition: int, bmax_over_aH: float, point: DataPoint, force_n: float, ms: float
) -> dict[str, object]:
    model_acceleration_cm_s2 = abs(force_n / ms) * 100.0
    log_residual = math.nan
    weighted_log_residual = math.nan
    if model_acceleration_cm_s2 > 0.0 and point.acceleration_cm_s2 > 0.0:
        log_residual = math.log(model_acceleration_cm_s2) - math.log(point.acceleration_cm_s2)
        sigma_log = max(relative_sigma(point), 0.05) if np.isfinite(relative_sigma(point)) else 1.0
        weighted_log_residual = log_residual / sigma_log

    return {
        "condition": condition,
        "bmax_over_aH": bmax_over_aH,
        "velocity_cm_s": point.velocity_cm_s,
        "data_acceleration_cm_s2": point.acceleration_cm_s2,
        "data_acceleration_sigma_cm_s2": point.acceleration_sigma_cm_s2,
        "model_acceleration_cm_s2": model_acceleration_cm_s2,
        "drag_N": force_n,
        "source": point.source,
        "log_residual": log_residual,
        "weighted_log_residual": weighted_log_residual,
    }


def run_fit_point_case(task: tuple[int, float, DataPoint, str, int, int]) -> dict[str, object]:
    condition, bmax_over_aH, point, method, resolution, vres = task
    drag = make_drag_for_fit(condition, bmax_over_aH, method, resolution, vres)
    force_n = quiet_drag(drag, point.velocity_cm_s * CM_PER_S_TO_M_PER_S)
    return _fit_point_row(condition, bmax_over_aH, point, force_n, drag.ms)


def run_fit_points_gpu(
    condition: int,
    bmax_over_aH: float,
    points: list[DataPoint],
    method: str,
    resolution: int,
    vres: int,
    gpu_devices: list[int],
) -> list[dict[str, object]]:
    """GPU counterpart of calling ``run_fit_point_case`` once per point.

    All points in one ``least_squares`` iteration share the same trial
    ``bmax_over_aH``, so this builds a single ``FiniteLaunchDrag`` and
    evaluates every point in one batched ``FiniteLaunchDrag.drag_batch``
    call per GPU (`theory/finite/finite_launch.py`), splitting the point
    list ~evenly across ``gpu_devices``. Row order in the result need not
    match ``points``' order -- ``least_squares``'s sum-of-squares residual
    and the CSV/plot consumers downstream are both order-independent, only
    the per-point (data, prediction) pairing has to be correct, which it is
    by construction here. ``cupy`` is imported lazily so the CPU-only path
    elsewhere in this module never needs it installed.

    CAVEAT: only exercised against the numpy backend of ``drag_batch`` (see
    that method's docstring) -- there is no GPU in the development
    environment this was written in, so the ``cupy`` code path itself has
    not been run. Spot-check a handful of points' ``--gpu-devices`` output
    against the CPU path's before trusting a fit that used it.
    """
    import cupy

    if not points:
        return []
    drag = make_drag_for_fit(condition, bmax_over_aH, method, resolution, vres)
    chunks = np.array_split(np.arange(len(points)), len(gpu_devices))

    def run_chunk(device_id: int, idx: np.ndarray) -> list[dict[str, object]]:
        if idx.size == 0:
            return []
        chunk = [points[i] for i in idx]
        vb = np.array([p.velocity_cm_s * CM_PER_S_TO_M_PER_S for p in chunk], dtype=np.float64)
        with cupy.cuda.Device(device_id), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            forces = cupy.asnumpy(drag.drag_batch(cupy.asarray(vb), xp=cupy))
        return [_fit_point_row(condition, bmax_over_aH, p, float(f), drag.ms) for p, f in zip(chunk, forces)]

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_devices)) as pool:
        futures = [pool.submit(run_chunk, device, idx) for device, idx in zip(gpu_devices, chunks)]
        for future in futures:
            rows.extend(future.result())
    return rows


def covariance_sigma(result, n_points: int) -> float:
    if result.jac is None or result.jac.size == 0:
        return math.nan
    dof = max(1, n_points - len(result.x))
    reduced_chi2 = float(np.sum(np.square(result.fun)) / dof)
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * reduced_chi2
    except np.linalg.LinAlgError:
        return math.nan
    sigma = math.sqrt(float(cov[0, 0]))
    return sigma if np.isfinite(sigma) else math.nan


def fit_condition(
    pool: ProcessPoolExecutor,
    condition: int,
    points: list[DataPoint],
    bmax_min: float,
    bmax_max: float,
    method: str,
    resolution: int,
    vres: int,
    max_nfev: int,
    progress: bool,
    heartbeat_seconds: float,
    gpu_devices: list[int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    """``gpu_devices``, when given (e.g. ``[0, 1]``), evaluates every point
    batch for a trial ``bmax_over_aH`` via ``run_fit_points_gpu`` instead of
    submitting one ``run_fit_point_case`` task per point to ``pool`` -- see
    that function's docstring for the batching rationale and its GPU-
    untested caveat. ``pool`` is unused in that case but still required by
    callers that don't know in advance whether GPU dispatch is active.
    """
    eval_counter = 0

    def evaluate_points(bmax_over_aH: float, subset: list[DataPoint], label: str) -> list[dict[str, object]]:
        if gpu_devices:
            return run_fit_points_gpu(condition, bmax_over_aH, subset, method, resolution, vres, gpu_devices)
        tasks = [(condition, bmax_over_aH, point, method, resolution, vres) for point in subset]
        return run_pool_with_heartbeat(
            pool, tasks, run_fit_point_case, heartbeat_seconds=heartbeat_seconds, label=label, quiet=not progress,
        )

    def residual_vector(params: np.ndarray) -> np.ndarray:
        nonlocal eval_counter
        eval_counter += 1
        bmax_over_aH = float(params[0])
        try:
            rows = evaluate_points(bmax_over_aH, points, f"cond{condition} eval{eval_counter}")
        except Exception as exc:  # keep the optimizer alive; report a huge residual instead
            if progress:
                print(f"[fit eval failed] condition={condition} bmax/aH={bmax_over_aH:.6g} error={exc!r}", flush=True)
            return np.full(len(points), 1.0e30, dtype=float)
        residuals = np.array(
            [row["weighted_log_residual"] if np.isfinite(row["weighted_log_residual"]) else 1.0e30 for row in rows],
            dtype=float,
        )
        if progress:
            print(
                f"[fit] condition={condition} eval={eval_counter} bmax/aH={bmax_over_aH:.6g} "
                f"rms_weighted_residual={math.sqrt(float(np.mean(np.square(residuals)))):.4g}",
                flush=True,
            )
        return residuals

    # 1.0 is the natural first guess, but only if it is strictly interior to
    # the bounds -- with the default bounds (0.01, 1.0) it sits exactly on
    # the upper bound, which can pin a gradient-based optimizer at the edge.
    if bmax_min < 1.0 < bmax_max:
        initial = 1.0
    else:
        initial = math.sqrt(bmax_min * bmax_max)
    result = least_squares(
        residual_vector,
        x0=np.array([initial], dtype=float),
        bounds=(np.array([bmax_min]), np.array([bmax_max])),
        x_scale="jac",
        max_nfev=max_nfev,
    )
    best_bmax_over_aH = float(result.x[0])
    sigma = covariance_sigma(result, len(points))

    prediction_rows = evaluate_points(best_bmax_over_aH, points, f"cond{condition} final")
    reduced_chi2 = float(np.sum(np.square(result.fun)) / max(1, len(points) - 1))

    a_H_m, debye_length_m = hydrogen_spacing_and_debye_length_m(condition)
    best_bmax_over_lD = math.nan
    best_bmax_over_lD_sigma = math.nan
    if condition in WEAKLY_COUPLED_CONDITIONS:
        aH_over_lD = a_H_m / debye_length_m
        best_bmax_over_lD = best_bmax_over_aH * aH_over_lD
        if np.isfinite(sigma):
            best_bmax_over_lD_sigma = sigma * aH_over_lD

    summary = {
        "condition": condition,
        "n_points": len(points),
        "best_bmax_over_aH": best_bmax_over_aH,
        "best_bmax_over_aH_sigma": sigma,
        "best_bmax_over_debye_length": best_bmax_over_lD,
        "best_bmax_over_debye_length_sigma": best_bmax_over_lD_sigma,
        "hydrogen_spacing_m": a_H_m,
        "debye_length_m": debye_length_m,
        "at_upper_bound": math.isclose(best_bmax_over_aH, bmax_max, rel_tol=1e-6),
        "reduced_chi2": reduced_chi2,
        "n_function_evals": result.nfev,
        "converged": bool(result.success),
        "method": method,
        "resolution_rhores_dphires": resolution,
        "vres": vres,
    }

    # +/-1 sigma model curves for the shaded uncertainty band in the overlay
    # plot. Only evaluated at the fit points already computed above -- cheap
    # relative to the least_squares iterations themselves (two extra passes
    # instead of dozens).
    uncertainty_rows: dict[str, list[dict[str, object]]] = {"low": [], "high": []}
    if np.isfinite(sigma) and sigma > 0:
        bmax_low = max(bmax_min, best_bmax_over_aH - sigma)
        bmax_high = min(bmax_max, best_bmax_over_aH + sigma)
        uncertainty_rows["low"] = evaluate_points(bmax_low, points, f"cond{condition} -1sigma")
        uncertainty_rows["high"] = evaluate_points(bmax_high, points, f"cond{condition} +1sigma")

    return summary, prediction_rows, uncertainty_rows


def make_overlay_plot(
    condition: int,
    summary: dict[str, object],
    prediction_rows: list[dict[str, object]],
    all_points: list[DataPoint],
    uncertainty_rows: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))

    all_v = np.array([p.velocity_cm_s for p in all_points if p.condition == condition])
    all_a = np.array([p.acceleration_cm_s2 for p in all_points if p.condition == condition])
    axis.scatter(all_v, all_a, s=8, color="lightgray", label="LAMMPS points (not fit)", zorder=1)

    fit_v = np.array([row["velocity_cm_s"] for row in prediction_rows])
    fit_a = np.array([row["data_acceleration_cm_s2"] for row in prediction_rows])
    fit_sigma = np.array([row["data_acceleration_sigma_cm_s2"] for row in prediction_rows])
    axis.errorbar(fit_v, fit_a, yerr=fit_sigma, fmt="o", color="tab:red", markersize=6, capsize=3, label="fit points", zorder=4)

    order = np.argsort(fit_v)
    model_v = fit_v[order]
    model_a = np.array([row["model_acceleration_cm_s2"] for row in prediction_rows])[order]

    # Lightly shaded +/-1 sigma band from the fitted b_max uncertainty, drawn
    # under the fit points/curve.
    low_rows = uncertainty_rows.get("low") if uncertainty_rows else None
    high_rows = uncertainty_rows.get("high") if uncertainty_rows else None
    if low_rows and high_rows:
        low_a = np.array([row["model_acceleration_cm_s2"] for row in low_rows])[order]
        high_a = np.array([row["model_acceleration_cm_s2"] for row in high_rows])[order]
        band_low = np.minimum(low_a, high_a)
        band_high = np.maximum(low_a, high_a)
        axis.fill_between(
            model_v, band_low, band_high,
            color="tab:blue", alpha=0.15, linewidth=0,
            label=r"model $\pm1\sigma$", zorder=1.5,
        )

    model_label = f"model, $b_{{max}}/a_H={summary['best_bmax_over_aH']:.4g}$"
    best_bmax_over_lD = summary.get("best_bmax_over_debye_length", math.nan)
    if condition in WEAKLY_COUPLED_CONDITIONS and np.isfinite(best_bmax_over_lD):
        model_label += f"\n$b_{{max}}/\\lambda_{{De}}={best_bmax_over_lD:.3g}$"
    axis.plot(model_v, model_a, color="tab:blue", linewidth=2.0, marker="s", markersize=4, label=model_label, zorder=3)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("velocity [cm/s]")
    axis.set_ylabel("acceleration [cm/s^2]")
    title = f"Condition {condition}: b_max fit ($r_i=a_H$ fixed), reduced chi2={summary['reduced_chi2']:.3g}"
    if condition in WEAKLY_COUPLED_CONDITIONS and np.isfinite(best_bmax_over_lD):
        sigma_lD = summary.get("best_bmax_over_debye_length_sigma", math.nan)
        sigma_str = f"{sigma_lD:.2g}" if np.isfinite(sigma_lD) else "n/a"
        title += f"\n$b_{{max}}/\\lambda_{{De}}={best_bmax_over_lD:.3g}\\pm{sigma_str}$ (weakly coupled)"
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"condition_{condition}_bmax_fit_overlay.png", dpi=200)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", type=int, default=list(ALL_CONDITIONS))
    parser.add_argument("--lammps-results", type=Path, default=REPO_ROOT / "theory" / "dataprocessing" / "output" / "results.npy")
    parser.add_argument("--dais-results", type=Path, default=REPO_ROOT / "theory" / "dataprocessing" / "output_dais" / "results.npy")
    parser.add_argument("--samples-per-fit", type=int, default=20)
    parser.add_argument("--min-velocity-cm-s", type=float, default=1.0e2)
    parser.add_argument("--max-velocity-cm-s", type=float, default=1.0e8)
    parser.add_argument("--max-relative-sigma", type=float, default=0.5)
    parser.add_argument("--points-per-condition", type=int, default=DEFAULT_POINTS_PER_CONDITION)
    parser.add_argument("--bmax-min", type=float, default=DEFAULT_BMAX_MIN)
    parser.add_argument("--bmax-max", type=float, default=DEFAULT_BMAX_MAX)
    parser.add_argument("--method", choices=("quad_quad", "vectorized"), default="vectorized")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="rhores = dphires used for every drag evaluation during the fit")
    parser.add_argument("--vres", type=int, default=DEFAULT_VRES)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device ids (e.g. '0,1') to batch every fit-point "
            "evaluation across via FiniteLaunchDrag.drag_batch(xp=cupy) instead of the "
            "CPU --workers process pool. Requires cupy; not exercised on real GPU "
            "hardware in development -- see run_fit_points_gpu's docstring."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=12.0,
        help="Print a status line at least this often even if no point evaluation has finished yet.",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-evaluation progress printing")
    args = parser.parse_args()
    gpu_devices = parse_gpu_devices(args.gpu_devices)

    if not args.lammps_results.exists():
        parser.error(f"--lammps-results not found: {args.lammps_results}")

    conditions = set(args.conditions)
    if conditions.intersection(DAIS_CONDITIONS) and not args.dais_results.exists():
        parser.error(f"--dais-results not found: {args.dais_results}")

    all_points = load_all_points(args.lammps_results, args.dais_results, conditions, args.samples_per_fit)
    filtered = filter_points(all_points, args.min_velocity_cm_s, args.max_velocity_cm_s, args.max_relative_sigma)
    fit_points = select_fit_points(filtered, args.points_per_condition)
    print(
        f"Loaded {len(all_points)} raw points, {len(filtered)} after filtering, "
        f"{len(fit_points)} selected for fitting across conditions {sorted(conditions)}.",
        flush=True,
    )

    start = time.perf_counter()
    summaries: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []
    # GPU dispatch never touches the CPU pool (see fit_condition/evaluate_points),
    # so skip spinning up worker processes that would just sit idle.
    with contextlib.nullcontext(None) if gpu_devices else ProcessPoolExecutor(max_workers=args.workers) as pool:
        for condition in sorted(conditions):
            condition_points = [p for p in fit_points if p.condition == condition]
            if not condition_points:
                print(f"Condition {condition}: no fit points after selection, skipping.", flush=True)
                continue
            print(f"Condition {condition}: fitting b_max/a_H against {len(condition_points)} points.", flush=True)
            summary, prediction_rows, uncertainty_rows = fit_condition(
                pool,
                condition,
                condition_points,
                args.bmax_min,
                args.bmax_max,
                args.method,
                args.resolution,
                args.vres,
                args.max_nfev,
                progress=not args.quiet,
                heartbeat_seconds=args.heartbeat_seconds,
                gpu_devices=gpu_devices,
            )
            if summary["at_upper_bound"]:
                print(
                    f"  WARNING: condition {condition} best fit sits at the explicit b_max/a_H upper "
                    f"bound ({args.bmax_max:g}) passed via --bmax-max; the default bound is infinity, "
                    f"so this only fires when that default was overridden.",
                    flush=True,
                )
            print(
                f"  best b_max/a_H = {summary['best_bmax_over_aH']:.6g} +/- {summary['best_bmax_over_aH_sigma']:.3g}, "
                f"reduced chi2 = {summary['reduced_chi2']:.4g}, converged={summary['converged']}.",
                flush=True,
            )
            if condition in WEAKLY_COUPLED_CONDITIONS and np.isfinite(summary["best_bmax_over_debye_length"]):
                print(
                    f"  best b_max/lambda_De = {summary['best_bmax_over_debye_length']:.6g} "
                    f"+/- {summary['best_bmax_over_debye_length_sigma']:.3g} (weakly coupled condition).",
                    flush=True,
                )
            summaries.append(summary)
            all_prediction_rows.extend(prediction_rows)
            make_overlay_plot(condition, summary, prediction_rows, all_points, uncertainty_rows)

    write_csv(OUTDIR / "bmax_fit_summary.csv", summaries)
    write_csv(OUTDIR / "bmax_fit_predictions.csv", all_prediction_rows)
    print(f"Finished in {(time.perf_counter()-start)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
