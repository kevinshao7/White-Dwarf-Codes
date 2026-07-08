from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CM_PER_S_TO_M_PER_S, condition_label, make_drag, quiet_drag, write_csv

REPO_ROOT = Path(__file__).resolve().parents[3]
ALL_CONDITIONS = (0, 1, 2, 3)
STRONGLY_COUPLED_CONDITIONS = (1, 3)
COUPLING_PARAMETER = {0: 0.03, 1: 1.94, 2: 0.42, 3: 1.05}
LAMMPS_SOURCE_RE = re.compile(r"\[(\d+),(\d+),(\d+)\]$")


@dataclass(frozen=True)
class DataPoint:
    condition: int
    velocity_cm_s: float
    acceleration_cm_s2: float
    velocity_sigma_cm_s: float
    acceleration_sigma_cm_s2: float
    source: str


def positive_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(result) or result <= 0.0:
        return math.nan
    return result


def exp_velocity(time: np.ndarray, tau: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-time / tau)


def load_lammps_expfit_points(results_path: Path, conditions: set[int], samples_per_fit: int) -> list[DataPoint]:
    data = np.load(results_path, allow_pickle=False)
    points: list[DataPoint] = []
    if data.ndim != 3 or data.shape[2] < 6:
        raise ValueError(f"Expected results array shaped (condition, case, 6), got {data.shape}")

    for condition in sorted(conditions):
        if condition >= data.shape[0]:
            continue
        for row_index, row in enumerate(data[condition]):
            amplitude = positive_float(row[0])
            amplitude_sigma = positive_float(row[1])
            tau = positive_float(row[2])
            tau_sigma = positive_float(row[3])
            start_time = positive_float(row[4])
            end_time = positive_float(row[5])
            if not all(np.isfinite(value) for value in [amplitude, amplitude_sigma, tau, tau_sigma, start_time, end_time]):
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

            for sample_index, (velocity, acceleration, velocity_err, acceleration_err) in enumerate(
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
                        velocity_sigma_cm_s=positive_float(velocity_err),
                        acceleration_sigma_cm_s2=positive_float(abs(acceleration_err)),
                        source=f"{results_path.relative_to(REPO_ROOT)}[{condition},{row_index},{sample_index}]",
                    )
                )
    return points


def load_points_from_csv(path: Path, conditions: set[int]) -> list[DataPoint]:
    points: list[DataPoint] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            condition = int(row["condition"])
            if condition not in conditions:
                continue
            velocity = positive_float(row.get("velocity_cm_s"))
            acceleration = positive_float(row.get("acceleration_cm_s2"))
            if not np.isfinite(velocity) or not np.isfinite(acceleration):
                continue
            points.append(
                DataPoint(
                    condition=condition,
                    velocity_cm_s=velocity,
                    acceleration_cm_s2=acceleration,
                    velocity_sigma_cm_s=positive_float(row.get("velocity_sigma_cm_s")),
                    acceleration_sigma_cm_s2=positive_float(row.get("acceleration_sigma_cm_s2")),
                    source=str(path),
                )
            )
    return points


def filter_points(
    points: list[DataPoint],
    min_velocity_cm_s: float,
    max_velocity_cm_s: float,
    max_relative_sigma: float,
) -> list[DataPoint]:
    filtered = []
    for point in points:
        if point.velocity_cm_s < min_velocity_cm_s or point.velocity_cm_s > max_velocity_cm_s:
            continue
        if np.isfinite(point.acceleration_sigma_cm_s2):
            relative_sigma = point.acceleration_sigma_cm_s2 / point.acceleration_cm_s2
            if relative_sigma > max_relative_sigma:
                continue
        filtered.append(point)
    return filtered


def relative_acceleration_error(point: DataPoint) -> float:
    if not np.isfinite(point.acceleration_sigma_cm_s2):
        return math.inf
    return point.acceleration_sigma_cm_s2 / point.acceleration_cm_s2


def lammps_campaign_id(point: DataPoint) -> int | None:
    match = LAMMPS_SOURCE_RE.search(point.source)
    return int(match.group(2)) if match else None


def lowest_relative_error_points(points: list[DataPoint], count: int) -> list[DataPoint]:
    return sorted(points, key=lambda point: (relative_acceleration_error(point), point.velocity_cm_s))[:count]


def group_points_by_lammps_campaign(points: list[DataPoint]) -> dict[int, list[DataPoint]]:
    campaigns: dict[int, list[DataPoint]] = {}
    for point in points:
        campaign_id = lammps_campaign_id(point)
        if campaign_id is not None:
            campaigns.setdefault(campaign_id, []).append(point)
    return campaigns


def campaign_velocity(campaign_points: list[DataPoint]) -> float:
    return max(point.velocity_cm_s for point in campaign_points)


def select_log_spaced_campaign_points(
    points: list[DataPoint],
    count: int,
    min_velocity_cm_s: float = 1.0e5,
    max_velocity_cm_s: float = 1.0e8,
) -> list[DataPoint]:
    campaigns = group_points_by_lammps_campaign(points)
    if len(campaigns) < count:
        return lowest_relative_error_points(points, count)

    candidates = {
        campaign_id: campaign_points
        for campaign_id, campaign_points in campaigns.items()
        if min_velocity_cm_s <= campaign_velocity(campaign_points) <= max_velocity_cm_s
    }
    if len(candidates) < count:
        candidates = campaigns

    selected_campaigns: list[list[DataPoint]] = []
    targets = np.geomspace(min_velocity_cm_s, max_velocity_cm_s, count)
    for target in targets:
        campaign_id, campaign_points = min(
            candidates.items(),
            key=lambda item: abs(math.log(campaign_velocity(item[1]) / target)),
        )
        selected_campaigns.append(campaign_points)
        del candidates[campaign_id]
        if not candidates:
            break

    selected = [lowest_relative_error_points(campaign_points, 1)[0] for campaign_points in selected_campaigns]
    return sorted(selected, key=lambda point: point.velocity_cm_s)


def select_fit_points(points: list[DataPoint], points_per_condition: int) -> tuple[list[DataPoint], dict[int, str]]:
    selected: list[DataPoint] = []
    descriptions: dict[int, str] = {}
    for condition in sorted({point.condition for point in points}):
        group = [point for point in points if point.condition == condition]
        if condition in STRONGLY_COUPLED_CONDITIONS:
            condition_points = select_log_spaced_campaign_points(group, points_per_condition)
            if len({lammps_campaign_id(point) for point in condition_points if lammps_campaign_id(point) is not None}) == len(condition_points):
                descriptions[condition] = (
                    "lowest acceleration_sigma / acceleration point from roughly log-spaced LAMMPS campaigns between 1e5 and 1e8 cm/s"
                )
            else:
                descriptions[condition] = "lowest acceleration_sigma / acceleration"
        else:
            condition_points = lowest_relative_error_points(group, points_per_condition)
            descriptions[condition] = "lowest acceleration_sigma / acceleration"
        selected.extend(condition_points)
    return selected, descriptions


def make_drag_for_fit(
    condition: int,
    fit_value: float,
    fit_parameter: str,
    vres: int,
    rhores: int,
    ures: int,
    dphires: int,
):
    if fit_parameter == "rhomax-spacing":
        return make_drag(condition, vres=vres, rhores=rhores, ures=ures, dphires=dphires, rhomax_fraction=fit_value)
    if fit_parameter == "rhomax":
        probe = make_drag(condition, vres=vres, rhores=rhores, ures=ures, dphires=dphires)
        return make_drag(
            condition,
            vres=vres,
            rhores=rhores,
            ures=ures,
            dphires=dphires,
            rhomax_fraction=fit_value * probe.lD * probe.ustart,
        )
    if fit_parameter == "outer-radius":
        return make_drag(condition, vres=vres, rhores=rhores, ures=ures, dphires=dphires, cutoff_radius_factor=fit_value)
    raise ValueError(f"Unknown fit parameter: {fit_parameter}")


def ion_screening_length(drag) -> float:
    return math.sqrt(drag.e0 * drag.kb * drag.T / (drag.nh * np.square(drag.z1 * drag.qe)))


def prediction_metadata(drag, fit_value: float, fit_parameter: str) -> dict[str, float]:
    ion_screening_length_m = ion_screening_length(drag)
    yukawa_screening_length_m = 1.0 / drag.k0
    impact_parameter_upper_bound_m = drag.rhomax_fraction / drag.ustart
    interparticle_spacing_m = 1.0 / drag.ustart
    return {
        "rhomax_fraction_of_interparticle_spacing": float(impact_parameter_upper_bound_m / interparticle_spacing_m),
        "rhomax_fraction_of_debye_length": float(impact_parameter_upper_bound_m / drag.lD),
        "rhomax_fraction_of_ion_screening_length": float(impact_parameter_upper_bound_m / ion_screening_length_m),
        "rhomax_fraction_of_yukawa_screening_length": float(impact_parameter_upper_bound_m / yukawa_screening_length_m),
        "dragbase_rhomax_fraction_of_outer_radius": float(drag.rhomax_fraction),
        "electron_debye_radius_m": float(drag.lD),
        "ion_screening_length_m": float(ion_screening_length_m),
        "yukawa_screening_length_m": float(yukawa_screening_length_m),
        "impact_parameter_upper_bound_m": float(impact_parameter_upper_bound_m),
        "outer_radius_m": float(1.0 / drag.ustart),
        "hydrogen_interparticle_spacing_m": float(interparticle_spacing_m),
    }


def progress_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def run_fit_point_case(task: tuple[int, float, str, DataPoint, int, int, int, int]) -> dict[str, object]:
    condition, fit_value, fit_parameter, point, vres, rhores, ures, dphires = task
    drag = make_drag_for_fit(condition, fit_value, fit_parameter, vres, rhores, ures, dphires)
    force_n = quiet_drag(drag, point.velocity_cm_s * CM_PER_S_TO_M_PER_S)
    model_acceleration = abs(force_n / drag.ms) * 100.0
    log_residual = math.nan
    weighted_log_residual = math.nan
    if model_acceleration > 0.0 and point.acceleration_cm_s2 > 0.0:
        log_residual = math.log(model_acceleration) - math.log(point.acceleration_cm_s2)
        if np.isfinite(point.acceleration_sigma_cm_s2):
            sigma_log = max(point.acceleration_sigma_cm_s2 / point.acceleration_cm_s2, 0.05)
            weighted_log_residual = log_residual / sigma_log
        else:
            weighted_log_residual = log_residual
    return {
        "condition": condition,
        "fit_parameter": fit_parameter,
        "fit_value": fit_value,
        "velocity_cm_s": point.velocity_cm_s,
        "velocity_m_s": point.velocity_cm_s * CM_PER_S_TO_M_PER_S,
        "data_acceleration_cm_s2": point.acceleration_cm_s2,
        "data_acceleration_sigma_cm_s2": point.acceleration_sigma_cm_s2,
        "model_acceleration_cm_s2": model_acceleration,
        "drag_N": force_n,
        "source": point.source,
        "status": "ok",
        "error": "",
        "log_residual": log_residual,
        "weighted_log_residual": weighted_log_residual,
        **prediction_metadata(drag, fit_value, fit_parameter),
    }


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


def model_rows_for_fit_value_parallel(
    pool: ProcessPoolExecutor,
    condition: int,
    fit_value: float,
    fit_parameter: str,
    points: list[DataPoint],
    vres: int,
    rhores: int,
    ures: int,
    dphires: int,
    progress: bool,
    eval_index: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    tasks = [
        (condition, fit_value, fit_parameter, point, vres, rhores, ures, dphires)
        for point in points
    ]
    rows = list(pool.map(run_fit_point_case, tasks))
    label = fit_value_label(fit_parameter)
    weighted_residuals = []
    for point_index, row in enumerate(rows, start=1):
        progress_print(
            progress,
            "[residual] "
            f"condition={condition} {label}={fit_value:.6g} eval={eval_index} "
            f"point={point_index}/{len(rows)} "
            f"v={float(row['velocity_cm_s']):.6e} cm/s "
            f"data={float(row['data_acceleration_cm_s2']):.6e} "
            f"model={float(row['model_acceleration_cm_s2']):.6e} cm/s^2 "
            f"log_residual={float(row['log_residual']):.6g} weighted={float(row['weighted_log_residual']):.6g}",
        )
        weighted_residual = float(row["weighted_log_residual"])
        weighted_residuals.append(weighted_residual if np.isfinite(weighted_residual) else 1.0e30)
    return rows, np.array(weighted_residuals, dtype=float)


def evaluate_fit_parallel(
    pool: ProcessPoolExecutor,
    condition: int,
    initial: float,
    fit_min: float,
    fit_max: float,
    fit_parameter: str,
    points: list[DataPoint],
    vres: int,
    rhores: int,
    ures: int,
    dphires: int,
    max_nfev: int,
    progress: bool,
) -> dict[str, object]:
    label = fit_value_label(fit_parameter)
    eval_counter = 0
    clipped_initial = min(max(initial, fit_min), fit_max)
    progress_print(
        progress,
        f"[fit start] condition={condition} {label}_initial={clipped_initial:.6g} "
        f"bounds=({fit_min:.6g}, {fit_max:.6g}) points={len(points)} parallel_drag_workers={min(len(points), pool._max_workers)}",
    )

    def residual_vector(params: np.ndarray) -> np.ndarray:
        nonlocal eval_counter
        eval_counter += 1
        fit_value = float(params[0])
        try:
            _, residuals = model_rows_for_fit_value_parallel(
                pool,
                condition,
                fit_value,
                fit_parameter,
                points,
                vres,
                rhores,
                ures,
                dphires,
                progress,
                eval_counter,
            )
            return residuals
        except Exception as exc:
            progress_print(progress, f"[fit eval failed] condition={condition} {label}={fit_value:.6g} error={exc!r}")
            return np.full(len(points), 1.0e30, dtype=float)

    try:
        result = least_squares(
            residual_vector,
            x0=np.array([clipped_initial], dtype=float),
            bounds=(np.array([fit_min], dtype=float), np.array([fit_max], dtype=float)),
            x_scale="jac",
            max_nfev=max_nfev,
        )
        best_value = float(result.x[0])
        prediction_rows, weighted_residuals = model_rows_for_fit_value_parallel(
            pool,
            condition,
            best_value,
            fit_parameter,
            points,
            vres,
            rhores,
            ures,
            dphires,
            progress,
            eval_counter + 1,
        )
        log_residuals = np.array([float(row["log_residual"]) for row in prediction_rows if np.isfinite(row["log_residual"])], dtype=float)
    except Exception as exc:
        progress_print(progress, f"[fit failed] condition={condition} error={exc!r}")
        return {
            "condition": condition,
            "fit_value": math.nan,
            "fit_parameter": fit_parameter,
            "score": math.nan,
            "rmse_log": math.nan,
            "n_points": 0,
            "best_fit_value_sigma": math.nan,
            "optimizer_success": False,
            "optimizer_message": repr(exc),
            "optimizer_nfev": eval_counter,
            "prediction_rows": [
                {
                    "condition": condition,
                    "fit_parameter": fit_parameter,
                    "status": "failed",
                    "error": repr(exc),
                }
            ],
        }

    score = float(np.mean(np.square(weighted_residuals))) if len(weighted_residuals) else math.nan
    rmse_log = float(np.sqrt(np.mean(np.square(log_residuals)))) if len(log_residuals) else math.nan
    sigma = covariance_sigma(result, len(weighted_residuals))
    progress_print(
        progress,
        f"[fit done] condition={condition} {label}={best_value:.6g} sigma={sigma:.6g} "
        f"score={score:.6g} rmse_log={rmse_log:.6g} nfev={result.nfev} success={result.success}",
    )
    return {
        "condition": condition,
        "fit_value": best_value,
        "fit_parameter": fit_parameter,
        "score": score,
        "rmse_log": rmse_log,
        "n_points": len(weighted_residuals),
        "best_fit_value_sigma": sigma,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "prediction_rows": prediction_rows,
    }


def run_curve_case(task: tuple[int, float, str, float, int, int, int, int, bool]) -> dict[str, float | int | str]:
    condition, fit_value, fit_parameter, velocity_cm_s, vres, rhores, ures, dphires, progress = task
    label = fit_value_label(fit_parameter)
    try:
        drag = make_drag_for_fit(condition, fit_value, fit_parameter, vres, rhores, ures, dphires)
        force_n = quiet_drag(drag, velocity_cm_s * CM_PER_S_TO_M_PER_S)
        model_acceleration = abs(force_n / drag.ms) * 100.0
        progress_print(
            progress,
            "[curve] "
            f"condition={condition} {label}={fit_value:.6g} "
            f"v={velocity_cm_s:.6e} cm/s model={model_acceleration:.6e} cm/s^2",
        )
        return {
            "condition": condition,
            "fit_parameter": fit_parameter,
            "fit_value": fit_value,
            "velocity_cm_s": velocity_cm_s,
            "model_acceleration_cm_s2": model_acceleration,
            "drag_N": force_n,
            "status": "ok",
            "error": "",
            **prediction_metadata(drag, fit_value, fit_parameter),
        }
    except Exception as exc:
        progress_print(progress, f"[curve failed] condition={condition} {label}={fit_value:.6g} v={velocity_cm_s:.6e} error={exc!r}")
        return {
            "condition": condition,
            "fit_parameter": fit_parameter,
            "fit_value": fit_value,
            "velocity_cm_s": velocity_cm_s,
            "model_acceleration_cm_s2": math.nan,
            "drag_N": math.nan,
            "status": "failed",
            "error": repr(exc),
        }


def fit_value_label(fit_parameter: str) -> str:
    if fit_parameter == "rhomax-spacing":
        return "bmax/aH"
    if fit_parameter == "rhomax":
        return "bmax/lD"
    return fit_parameter


def format_uncertainty(value: float, lower: object, upper: object) -> str:
    try:
        lower_float = float(lower)
        upper_float = float(upper)
    except (TypeError, ValueError):
        lower_float = math.nan
        upper_float = math.nan
    if np.isfinite(lower_float) or np.isfinite(upper_float):
        lower_text = f"{lower_float:.2g}" if np.isfinite(lower_float) else "unbounded"
        upper_text = f"{upper_float:.2g}" if np.isfinite(upper_float) else "unbounded"
        return f"{value:.3g} (-{lower_text}/+{upper_text})"
    return f"{value:.3g} (uncertainty unresolved)"


def plot_results(
    all_points: list[DataPoint],
    fit_points: list[DataPoint],
    curve_rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, object]],
    fit_parameter: str,
) -> None:
    conditions = sorted({point.condition for point in all_points})
    colors = {0: "red", 1: "orange", 2: "green", 3: "blue"}
    fig, ax = plt.subplots(figsize=(12, 8))
    for condition in conditions:
        color = colors.get(condition, None)
        group = [point for point in all_points if point.condition == condition]
        selected_sources = {point.source for point in fit_points if point.condition == condition}
        fit_group = [point for point in fit_points if point.condition == condition]
        best = next(row for row in summary_rows if row["condition"] == condition)
        curve = [row for row in curve_rows if row["condition"] == condition and row["status"] == "ok"]
        curve.sort(key=lambda row: row["velocity_cm_s"])

        ax.errorbar(
            [point.velocity_cm_s for point in group],
            [point.acceleration_cm_s2 for point in group],
            xerr=[point.velocity_sigma_cm_s if np.isfinite(point.velocity_sigma_cm_s) else 0.0 for point in group],
            yerr=[
                point.acceleration_sigma_cm_s2 if np.isfinite(point.acceleration_sigma_cm_s2) else 0.0
                for point in group
            ],
            fmt="o",
            markersize=4,
            alpha=0.35,
            color=color,
            label=f"{condition_label(condition)} data",
        )
        ax.scatter(
            [point.velocity_cm_s for point in fit_group if point.source in selected_sources],
            [point.acceleration_cm_s2 for point in fit_group if point.source in selected_sources],
            s=34,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
            label=f"condition {condition} fit window",
        )
        ax.plot(
            [row["velocity_cm_s"] for row in curve],
            [row["model_acceleration_cm_s2"] for row in curve],
            color=color,
            linewidth=2,
            label=(
                f"condition {condition}, Gamma={COUPLING_PARAMETER.get(condition, math.nan):.2g}, "
                f"bmax/aH="
                f"{format_uncertainty(float(best['rhomax_fraction_of_interparticle_spacing']), best.get('rhomax_fraction_of_interparticle_spacing_sigma'), best.get('rhomax_fraction_of_interparticle_spacing_sigma'))}, "
                f"bmax/lion="
                f"{format_uncertainty(float(best['rhomax_fraction_of_ion_screening_length']), best.get('rhomax_fraction_of_ion_screening_length_sigma'), best.get('rhomax_fraction_of_ion_screening_length_sigma'))}, "
                f"bmax/lY="
                f"{format_uncertainty(float(best['rhomax_fraction_of_yukawa_screening_length']), best.get('rhomax_fraction_of_yukawa_screening_length_sigma'), best.get('rhomax_fraction_of_yukawa_screening_length_sigma'))}"
            ),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("v (cm/s)")
    ax.set_ylabel("a (cm/s^2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "impact_parameter_fit.png", dpi=200)


def adaptive_curve_for_condition(points: list[DataPoint], n_curve: int) -> np.ndarray:
    velocities = np.array([point.velocity_cm_s for point in points], dtype=float)
    accelerations = np.array([point.acceleration_cm_s2 for point in points], dtype=float)
    min_velocity = float(np.min(velocities))
    max_velocity = float(np.max(velocities))
    peak_velocity = float(velocities[int(np.argmax(accelerations))])
    if n_curve <= 3:
        return np.geomspace(min_velocity, max_velocity, n_curve)

    near_min = max(min_velocity, peak_velocity / 3.0)
    near_max = min(max_velocity, peak_velocity * 3.0)
    near_count = max(4, int(round(0.5 * n_curve)))
    lower_count = max(1, (n_curve - near_count) // 2)
    upper_count = n_curve - near_count - lower_count

    pieces = []
    if near_min > min_velocity:
        pieces.append(np.geomspace(min_velocity, near_min, lower_count + 1)[:-1])
    pieces.append(np.geomspace(near_min, near_max, near_count))
    if near_max < max_velocity and upper_count > 0:
        pieces.append(np.geomspace(near_max, max_velocity, upper_count + 1)[1:])

    values = np.unique(np.concatenate(pieces))
    if len(values) < n_curve:
        filler = np.geomspace(min_velocity, max_velocity, n_curve)
        values = np.unique(np.concatenate([values, filler]))
    if len(values) > n_curve:
        required = {0, len(values) - 1}
        order = np.argsort(np.abs(np.log(values / peak_velocity)))
        keep = set(required)
        for index in order:
            keep.add(int(index))
            if len(keep) == n_curve:
                break
        values = np.array([value for index, value in enumerate(values) if index in keep], dtype=float)
    order = np.argsort(np.abs(np.log(values / peak_velocity)))
    keep = set(order[:n_curve])
    return np.array(sorted(value for index, value in enumerate(values) if index in keep), dtype=float)


def condition_curve_velocities(points: list[DataPoint], n_curve: int) -> dict[int, np.ndarray]:
    result = {}
    for condition in sorted({point.condition for point in points}):
        condition_points = [point for point in points if point.condition == condition]
        result[condition] = adaptive_curve_for_condition(condition_points, n_curve)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", type=int, default=list(ALL_CONDITIONS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--fit-parameter",
        choices=["rhomax-spacing", "rhomax", "outer-radius"],
        default="rhomax-spacing",
        help=(
            "rhomax-spacing fits bmax/aH, where aH is the hydrogen interparticle spacing; "
            "rhomax fits bmax/lD; outer-radius fits the starting radius relative to the default interparticle spacing."
        ),
    )
    parser.add_argument("--fit-min", type=float, default=0.03)
    parser.add_argument("--fit-max", type=float, default=1.0)
    parser.add_argument("--fit-initial", type=float, default=None, help="Initial guess for the fitted cutoff; defaults to sqrt(fit-min * fit-max).")
    parser.add_argument("--max-fit-evaluations", type=int, default=20)
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--lammps-results", type=Path, default=REPO_ROOT / "unforced" / "dataarchive" / "nprun4_29" / "results.npy")
    parser.add_argument("--samples-per-lammps-fit", type=int, default=10)
    parser.add_argument(
        "--fit-points-per-condition",
        type=int,
        default=8,
        help="Number of lowest-relative-acceleration-error data points to fit for each condition.",
    )
    parser.add_argument("--min-velocity-cm-s", type=float, default=1.0e2)
    parser.add_argument("--max-velocity-cm-s", type=float, default=1.0e8)
    parser.add_argument("--max-relative-sigma", type=float, default=10.0)
    parser.add_argument("--curve-points", type=int, default=24)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-residual and best-fit-curve progress prints.")
    parser.add_argument("--vres", type=int, default=50)
    parser.add_argument("--rhores", type=int, default=180)
    parser.add_argument("--ures", type=int, default=180)
    parser.add_argument("--dphires", type=int, default=180)
    args = parser.parse_args()
    if args.fit_min <= 0.0 or args.fit_max <= 0.0 or args.fit_min >= args.fit_max:
        raise SystemExit("--fit-min and --fit-max must be positive with fit-min < fit-max.")

    requested_conditions = set(args.conditions)
    unknown_conditions = requested_conditions.difference(ALL_CONDITIONS)
    if unknown_conditions:
        raise SystemExit(f"Unknown condition indexes {sorted(unknown_conditions)}; valid conditions are {list(ALL_CONDITIONS)}.")

    if args.data_csv:
        all_points = load_points_from_csv(args.data_csv, requested_conditions)
    else:
        all_points = load_lammps_expfit_points(args.lammps_results, requested_conditions, args.samples_per_lammps_fit)
    all_points = filter_points(all_points, args.min_velocity_cm_s, args.max_velocity_cm_s, args.max_relative_sigma)
    fit_points, fit_point_selection_by_condition = select_fit_points(all_points, args.fit_points_per_condition)
    if not fit_points:
        raise SystemExit("No usable fit points found. Check --conditions, data path, velocity/error filters, or --fit-points-per-condition.")

    points_by_condition = {
        condition: [point for point in fit_points if point.condition == condition]
        for condition in sorted(requested_conditions)
    }
    fit_initial = args.fit_initial if args.fit_initial is not None else math.sqrt(args.fit_min * args.fit_max)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fit_results = [
            evaluate_fit_parallel(
                pool,
                condition,
                fit_initial,
                args.fit_min,
                args.fit_max,
                args.fit_parameter,
                points,
                args.vres,
                args.rhores,
                args.ures,
                args.dphires,
                args.max_fit_evaluations,
                not args.quiet,
            )
            for condition, points in sorted(points_by_condition.items())
            if points
        ]

        prediction_rows = [row for result in fit_results for row in result["prediction_rows"]]
        summary_rows: list[dict[str, object]] = []
        for condition in sorted(points_by_condition):
            condition_result = next((result for result in fit_results if result["condition"] == condition), None)
            if not condition_result:
                continue
            best_prediction = next(row for row in condition_result["prediction_rows"] if row["status"] == "ok")
            fit_sigma = float(condition_result["best_fit_value_sigma"])
            fit_parameter = str(condition_result["fit_parameter"])
            debye_length = float(best_prediction["electron_debye_radius_m"])
            ion_screening = float(best_prediction["ion_screening_length_m"])
            yukawa_length = float(best_prediction["yukawa_screening_length_m"])
            interparticle_spacing = float(best_prediction["hydrogen_interparticle_spacing_m"])
            bmax_over_interparticle = float(best_prediction["rhomax_fraction_of_interparticle_spacing"])
            bmax_over_debye = float(best_prediction["rhomax_fraction_of_debye_length"])
            bmax_over_ion_screening = float(best_prediction["rhomax_fraction_of_ion_screening_length"])
            bmax_over_yukawa = float(best_prediction["rhomax_fraction_of_yukawa_screening_length"])
            bmax_over_interparticle_sigma = math.nan
            bmax_over_debye_sigma = math.nan
            bmax_over_ion_screening_sigma = math.nan
            bmax_over_yukawa_sigma = math.nan
            if np.isfinite(fit_sigma):
                if fit_parameter == "rhomax-spacing":
                    bmax_over_interparticle_sigma = fit_sigma
                    bmax_over_debye_sigma = fit_sigma * interparticle_spacing / debye_length
                    bmax_over_ion_screening_sigma = fit_sigma * interparticle_spacing / ion_screening
                    bmax_over_yukawa_sigma = fit_sigma * interparticle_spacing / yukawa_length
                elif fit_parameter == "rhomax":
                    bmax_over_interparticle_sigma = fit_sigma * debye_length / interparticle_spacing
                    bmax_over_debye_sigma = fit_sigma
                    bmax_over_ion_screening_sigma = fit_sigma * debye_length / ion_screening
                    bmax_over_yukawa_sigma = fit_sigma * debye_length / yukawa_length
            summary_rows.append(
                {
                    "condition": condition,
                    "condition_label": condition_label(condition),
                    "coupling_parameter": COUPLING_PARAMETER.get(condition, math.nan),
                    "fit_parameter": fit_parameter,
                    "best_fit_value": condition_result["fit_value"],
                    "best_fit_value_sigma": fit_sigma,
                    "best_fit_value_err_minus": fit_sigma,
                    "best_fit_value_err_plus": fit_sigma,
                    "uncertainty_rule": "least_squares covariance from Jacobian",
                    "optimizer_success": condition_result["optimizer_success"],
                    "optimizer_message": condition_result["optimizer_message"],
                    "optimizer_nfev": condition_result["optimizer_nfev"],
                    "score": condition_result["score"],
                    "rmse_log": condition_result["rmse_log"],
                    "n_points": condition_result["n_points"],
                    "fit_points_per_condition": args.fit_points_per_condition,
                    "fit_point_selection": fit_point_selection_by_condition.get(
                        condition, "lowest acceleration_sigma / acceleration"
                    ),
                    "impact_parameter_upper_bound_m": best_prediction["impact_parameter_upper_bound_m"],
                    "rhomax_fraction_of_interparticle_spacing": bmax_over_interparticle,
                    "rhomax_fraction_of_interparticle_spacing_sigma": bmax_over_interparticle_sigma,
                    "rhomax_fraction_of_debye_length": bmax_over_debye,
                    "rhomax_fraction_of_debye_length_sigma": bmax_over_debye_sigma,
                    "rhomax_fraction_of_ion_screening_length": bmax_over_ion_screening,
                    "rhomax_fraction_of_ion_screening_length_sigma": bmax_over_ion_screening_sigma,
                    "rhomax_fraction_of_yukawa_screening_length": bmax_over_yukawa,
                    "rhomax_fraction_of_yukawa_screening_length_sigma": bmax_over_yukawa_sigma,
                    "dragbase_rhomax_fraction_of_outer_radius": best_prediction["dragbase_rhomax_fraction_of_outer_radius"],
                    "electron_debye_radius_m": debye_length,
                    "ion_screening_length_m": ion_screening,
                    "yukawa_screening_length_m": yukawa_length,
                    "hydrogen_interparticle_spacing_m": interparticle_spacing,
                    "outer_radius_m": best_prediction["outer_radius_m"],
                }
            )

        curve_velocities = condition_curve_velocities(all_points, args.curve_points)
        curve_tasks = []
        for summary in summary_rows:
            condition = int(summary["condition"])
            for velocity_cm_s in curve_velocities[condition]:
                curve_tasks.append(
                    (
                        condition,
                        float(summary["best_fit_value"]),
                        args.fit_parameter,
                        float(velocity_cm_s),
                        args.vres,
                        args.rhores,
                        args.ures,
                        args.dphires,
                        not args.quiet,
                    )
                )
        curve_rows = list(pool.map(run_curve_case, curve_tasks))

    write_csv(OUTDIR / "impact_parameter_fit_predictions.csv", prediction_rows)
    write_csv(OUTDIR / "impact_parameter_fit_curve.csv", curve_rows)
    write_csv(OUTDIR / "impact_parameter_fit_summary.csv", summary_rows)
    plot_results(all_points, fit_points, curve_rows, summary_rows, args.fit_parameter)

    for row in summary_rows:
        print(
            f"condition {row['condition']}: best bmax/aH={float(row['rhomax_fraction_of_interparticle_spacing']):.6g} "
            f"+/- {float(row['rhomax_fraction_of_interparticle_spacing_sigma']):.3g}, "
            f"bmax/lion={float(row['rhomax_fraction_of_ion_screening_length']):.6g} "
            f"+/- {float(row['rhomax_fraction_of_ion_screening_length_sigma']):.3g}, "
            f"bmax/lY={float(row['rhomax_fraction_of_yukawa_screening_length']):.6g} "
            f"+/- {float(row['rhomax_fraction_of_yukawa_screening_length_sigma']):.3g}, "
            f"bmax={float(row['impact_parameter_upper_bound_m']):.6e} m, "
            f"fit points={row['n_points']}, RMSE(log)={float(row['rmse_log']):.4g}"
        )


if __name__ == "__main__":
    freeze_support()
    main()
