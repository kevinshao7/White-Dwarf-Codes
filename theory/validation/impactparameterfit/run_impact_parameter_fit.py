from __future__ import annotations

import argparse
import csv
import math
import os
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
WEAKLY_COUPLED_CONDITIONS = (0, 2)
COUPLING_PARAMETER = {0: 0.03, 2: 0.42}


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


def select_lowest_relative_error_points(points: list[DataPoint], points_per_condition: int) -> list[DataPoint]:
    selected: list[DataPoint] = []
    for condition in sorted({point.condition for point in points}):
        group = sorted(
            (point for point in points if point.condition == condition),
            key=lambda point: (relative_acceleration_error(point), point.velocity_cm_s),
        )
        selected.extend(group[:points_per_condition])
    return selected


def make_drag_for_fit(
    condition: int,
    fit_value: float,
    fit_parameter: str,
    vres: int,
    rhores: int,
    ures: int,
    dphires: int,
):
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


def prediction_metadata(drag, fit_value: float, fit_parameter: str) -> dict[str, float]:
    rhomax_fraction_of_debye_length = fit_value if fit_parameter == "rhomax" else math.nan
    yukawa_screening_length_m = 1.0 / drag.k0
    impact_parameter_upper_bound_m = drag.rhomax_fraction / drag.ustart
    return {
        "rhomax_fraction_of_debye_length": rhomax_fraction_of_debye_length,
        "rhomax_fraction_of_yukawa_screening_length": float(impact_parameter_upper_bound_m / yukawa_screening_length_m),
        "dragbase_rhomax_fraction_of_outer_radius": float(drag.rhomax_fraction),
        "electron_debye_radius_m": float(drag.lD),
        "yukawa_screening_length_m": float(yukawa_screening_length_m),
        "impact_parameter_upper_bound_m": float(impact_parameter_upper_bound_m),
        "outer_radius_m": float(1.0 / drag.ustart),
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


def model_rows_for_fit_value(
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
    rows = []
    weighted_residuals = []
    label = fit_value_label(fit_parameter)
    drag = make_drag_for_fit(condition, fit_value, fit_parameter, vres, rhores, ures, dphires)
    metadata = prediction_metadata(drag, fit_value, fit_parameter)
    for point_index, point in enumerate(points, start=1):
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
        rows.append(
            {
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
                **metadata,
            }
        )
        progress_print(
            progress,
            "[residual] "
            f"condition={condition} {label}={fit_value:.6g} eval={eval_index} "
            f"point={point_index}/{len(points)} "
            f"v={point.velocity_cm_s:.6e} cm/s "
            f"data={point.acceleration_cm_s2:.6e} model={model_acceleration:.6e} cm/s^2 "
            f"log_residual={log_residual:.6g} weighted={weighted_log_residual:.6g}",
        )
        weighted_residuals.append(weighted_log_residual if np.isfinite(weighted_log_residual) else 1.0e30)
    return rows, np.array(weighted_residuals, dtype=float)


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


def covariance_sigmas(result, n_points: int) -> np.ndarray:
    if result.jac is None or result.jac.size == 0:
        return np.full(len(result.x), math.nan, dtype=float)
    dof = max(1, n_points - len(result.x))
    reduced_chi2 = float(np.sum(np.square(result.fun)) / dof)
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * reduced_chi2
    except np.linalg.LinAlgError:
        return np.full(len(result.x), math.nan, dtype=float)
    variances = np.diag(cov)
    return np.array([math.sqrt(float(value)) if value >= 0.0 and np.isfinite(value) else math.nan for value in variances])


def evaluate_conditions_parallel(
    pool: ProcessPoolExecutor,
    points_by_condition: dict[int, list[DataPoint]],
    initial: float,
    fit_min: float,
    fit_max: float,
    fit_parameter: str,
    vres: int,
    rhores: int,
    ures: int,
    dphires: int,
    max_nfev: int,
    progress: bool,
) -> list[dict[str, object]]:
    ordered_conditions = sorted(condition for condition, points in points_by_condition.items() if points)
    condition_to_index = {condition: index for index, condition in enumerate(ordered_conditions)}
    clipped_initial = min(max(initial, fit_min), fit_max)
    eval_counter = 0
    label = fit_value_label(fit_parameter)
    progress_print(
        progress,
        f"[fit start] combined_conditions={ordered_conditions} {label}_initial={clipped_initial:.6g} "
        f"bounds=({fit_min:.6g}, {fit_max:.6g}) residual_points={sum(len(points_by_condition[c]) for c in ordered_conditions)} "
        f"parallel_drag_workers={pool._max_workers}",
    )

    def rows_for_params(params: np.ndarray, eval_index: int) -> tuple[list[dict[str, object]], np.ndarray]:
        tasks = []
        for condition in ordered_conditions:
            fit_value = float(params[condition_to_index[condition]])
            for point in points_by_condition[condition]:
                tasks.append((condition, fit_value, fit_parameter, point, vres, rhores, ures, dphires))
        rows = list(pool.map(run_fit_point_case, tasks))
        residuals = []
        for row_index, row in enumerate(rows, start=1):
            condition = int(row["condition"])
            fit_value = float(row["fit_value"])
            progress_print(
                progress,
                "[residual] "
                f"condition={condition} {label}={fit_value:.6g} eval={eval_index} "
                f"point={row_index}/{len(rows)} "
                f"v={float(row['velocity_cm_s']):.6e} cm/s "
                f"data={float(row['data_acceleration_cm_s2']):.6e} "
                f"model={float(row['model_acceleration_cm_s2']):.6e} cm/s^2 "
                f"log_residual={float(row['log_residual']):.6g} weighted={float(row['weighted_log_residual']):.6g}",
            )
            weighted_residual = float(row["weighted_log_residual"])
            residuals.append(weighted_residual if np.isfinite(weighted_residual) else 1.0e30)
        return rows, np.array(residuals, dtype=float)

    def residual_vector(params: np.ndarray) -> np.ndarray:
        nonlocal eval_counter
        eval_counter += 1
        try:
            _, residuals = rows_for_params(params, eval_counter)
            return residuals
        except Exception as exc:
            progress_print(progress, f"[fit eval failed] params={params} error={exc!r}")
            return np.full(sum(len(points_by_condition[c]) for c in ordered_conditions), 1.0e30, dtype=float)

    result = least_squares(
        residual_vector,
        x0=np.full(len(ordered_conditions), clipped_initial, dtype=float),
        bounds=(np.full(len(ordered_conditions), fit_min, dtype=float), np.full(len(ordered_conditions), fit_max, dtype=float)),
        x_scale="jac",
        max_nfev=max_nfev,
    )
    final_rows, final_weighted_residuals = rows_for_params(result.x, eval_counter + 1)
    sigmas = covariance_sigmas(result, len(final_weighted_residuals))
    fit_results = []
    for condition in ordered_conditions:
        condition_rows = [row for row in final_rows if int(row["condition"]) == condition]
        weighted = np.array([float(row["weighted_log_residual"]) for row in condition_rows], dtype=float)
        log_residuals = np.array([float(row["log_residual"]) for row in condition_rows if np.isfinite(row["log_residual"])], dtype=float)
        fit_value = float(result.x[condition_to_index[condition]])
        sigma = float(sigmas[condition_to_index[condition]])
        progress_print(
            progress,
            f"[fit done] condition={condition} {label}={fit_value:.6g} sigma={sigma:.6g} "
            f"score={float(np.mean(np.square(weighted))):.6g} "
            f"rmse_log={float(np.sqrt(np.mean(np.square(log_residuals)))):.6g} nfev={result.nfev} success={result.success}",
        )
        fit_results.append(
            {
                "condition": condition,
                "fit_value": fit_value,
                "fit_parameter": fit_parameter,
                "score": float(np.mean(np.square(weighted))) if len(weighted) else math.nan,
                "rmse_log": float(np.sqrt(np.mean(np.square(log_residuals)))) if len(log_residuals) else math.nan,
                "n_points": len(weighted),
                "best_fit_value_sigma": sigma,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "optimizer_nfev": int(result.nfev),
                "prediction_rows": condition_rows,
            }
        )
    return fit_results


def evaluate_fit(task: tuple[int, float, float, float, str, list[DataPoint], int, int, int, int, int, bool]) -> dict[str, object]:
    condition, initial, fit_min, fit_max, fit_parameter, points, vres, rhores, ures, dphires, max_nfev, progress = task
    label = fit_value_label(fit_parameter)
    eval_counter = 0
    clipped_initial = min(max(initial, fit_min), fit_max)
    progress_print(
        progress,
        f"[fit start] condition={condition} {label}_initial={clipped_initial:.6g} bounds=({fit_min:.6g}, {fit_max:.6g}) points={len(points)}",
    )

    def residual_vector(params: np.ndarray) -> np.ndarray:
        nonlocal eval_counter
        eval_counter += 1
        fit_value = float(params[0])
        try:
            _, residuals = model_rows_for_fit_value(
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
        prediction_rows, weighted_residuals = model_rows_for_fit_value(
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
    return "bmax/lD" if fit_parameter == "rhomax" else fit_parameter


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
    colors = {0: "red", 2: "green"}
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
                f"bmax/lD="
                f"{format_uncertainty(float(best['rhomax_fraction_of_debye_length']), best.get('rhomax_fraction_of_debye_length_sigma'), best.get('rhomax_fraction_of_debye_length_sigma'))}, "
                f"bmax/lS="
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
    parser.add_argument("--conditions", nargs="+", type=int, default=list(WEAKLY_COUPLED_CONDITIONS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--fit-parameter",
        choices=["rhomax", "outer-radius"],
        default="rhomax",
        help="rhomax fits bmax/lD; outer-radius fits the starting radius relative to the default interparticle spacing.",
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
        default=4,
        help="Number of lowest-relative-acceleration-error data points to fit for each weakly coupled condition.",
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
    non_weak_conditions = requested_conditions.difference(WEAKLY_COUPLED_CONDITIONS)
    if non_weak_conditions:
        raise SystemExit(f"This fitter is restricted to weakly coupled condition indexes 0 and 2; got {sorted(non_weak_conditions)}.")

    if args.data_csv:
        all_points = load_points_from_csv(args.data_csv, requested_conditions)
    else:
        all_points = load_lammps_expfit_points(args.lammps_results, requested_conditions, args.samples_per_lammps_fit)
    all_points = filter_points(all_points, args.min_velocity_cm_s, args.max_velocity_cm_s, args.max_relative_sigma)
    fit_points = select_lowest_relative_error_points(all_points, args.fit_points_per_condition)
    if not fit_points:
        raise SystemExit("No usable fit points found. Check --conditions, data path, velocity/error filters, or --fit-points-per-condition.")

    points_by_condition = {
        condition: [point for point in fit_points if point.condition == condition]
        for condition in sorted(requested_conditions)
    }
    fit_initial = args.fit_initial if args.fit_initial is not None else math.sqrt(args.fit_min * args.fit_max)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fit_results = evaluate_conditions_parallel(
            pool,
            points_by_condition,
            fit_initial,
            args.fit_min,
            args.fit_max,
            args.fit_parameter,
            args.vres,
            args.rhores,
            args.ures,
            args.dphires,
            args.max_fit_evaluations,
            not args.quiet,
        )

        prediction_rows = [row for result in fit_results for row in result["prediction_rows"]]
        summary_rows: list[dict[str, object]] = []
        for condition in sorted(points_by_condition):
            condition_result = next((result for result in fit_results if result["condition"] == condition), None)
            if not condition_result:
                continue
            best_prediction = next(row for row in condition_result["prediction_rows"] if row["status"] == "ok")
            fit_sigma = float(condition_result["best_fit_value_sigma"])
            debye_length = float(best_prediction["electron_debye_radius_m"])
            yukawa_length = float(best_prediction["yukawa_screening_length_m"])
            bmax_over_yukawa = float(best_prediction["rhomax_fraction_of_yukawa_screening_length"])
            bmax_over_yukawa_sigma = fit_sigma * debye_length / yukawa_length if np.isfinite(fit_sigma) else math.nan
            summary_rows.append(
                {
                    "condition": condition,
                    "condition_label": condition_label(condition),
                    "coupling_parameter": COUPLING_PARAMETER.get(condition, math.nan),
                    "fit_parameter": args.fit_parameter,
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
                    "fit_point_selection": "lowest acceleration_sigma / acceleration",
                    "impact_parameter_upper_bound_m": best_prediction["impact_parameter_upper_bound_m"],
                    "rhomax_fraction_of_debye_length": best_prediction["rhomax_fraction_of_debye_length"],
                    "rhomax_fraction_of_debye_length_sigma": fit_sigma,
                    "rhomax_fraction_of_yukawa_screening_length": bmax_over_yukawa,
                    "rhomax_fraction_of_yukawa_screening_length_sigma": bmax_over_yukawa_sigma,
                    "dragbase_rhomax_fraction_of_outer_radius": best_prediction["dragbase_rhomax_fraction_of_outer_radius"],
                    "electron_debye_radius_m": debye_length,
                    "yukawa_screening_length_m": yukawa_length,
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
            f"condition {row['condition']}: best bmax/lD={float(row['rhomax_fraction_of_debye_length']):.6g} "
            f"+/- {float(row['rhomax_fraction_of_debye_length_sigma']):.3g}, "
            f"bmax/lS={float(row['rhomax_fraction_of_yukawa_screening_length']):.6g} "
            f"+/- {float(row['rhomax_fraction_of_yukawa_screening_length_sigma']):.3g}, "
            f"bmax={float(row['impact_parameter_upper_bound_m']):.6e} m, "
            f"fit points={row['n_points']}, RMSE(log)={float(row['rmse_log']):.4g}"
        )


if __name__ == "__main__":
    freeze_support()
    main()
