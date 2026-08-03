from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

VALIDATION_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(VALIDATION_DIR / "impactparameterfit"))

from common import CM_PER_S_TO_M_PER_S, condition_label, make_drag, quiet_drag, write_csv
from run_impact_parameter_fit import (
    ALL_CONDITIONS,
    COUPLING_PARAMETER,
    DataPoint,
    campaign_velocity,
    condition_curve_velocities,
    filter_points,
    group_points_by_lammps_campaign,
    ion_screening_length,
    load_lammps_expfit_points,
    load_points_from_csv,
    relative_acceleration_error,
)


def progress_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def lowest_error_point(points: list[DataPoint]) -> DataPoint:
    return min(points, key=lambda point: (relative_acceleration_error(point), point.velocity_cm_s))


def select_regime_campaign_points(
    points: list[DataPoint],
    count: int,
    min_velocity_cm_s: float,
    max_velocity_cm_s: float,
) -> tuple[list[DataPoint], str]:
    """Select one low-error observation from each of several velocity regimes."""
    if count <= 0:
        return [], "no fit points requested"

    campaigns = group_points_by_lammps_campaign(points)
    if not campaigns:
        selected = sorted(points, key=lambda point: (relative_acceleration_error(point), point.velocity_cm_s))[:count]
        return selected, "lowest acceleration_sigma / acceleration; no LAMMPS campaign ids available"

    candidates = {
        campaign_id: campaign_points
        for campaign_id, campaign_points in campaigns.items()
        if min_velocity_cm_s <= campaign_velocity(campaign_points) <= max_velocity_cm_s
    }
    if not candidates:
        candidates = campaigns

    velocities = np.array([campaign_velocity(campaign_points) for campaign_points in candidates.values()], dtype=float)
    low = max(min_velocity_cm_s, float(np.min(velocities)))
    high = min(max_velocity_cm_s, float(np.max(velocities)))
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= low:
        selected_campaigns = sorted(
            candidates.values(),
            key=lambda campaign_points: (relative_acceleration_error(lowest_error_point(campaign_points)), campaign_velocity(campaign_points)),
        )[:count]
    else:
        selected_campaigns = []
        remaining = dict(candidates)
        for target in np.geomspace(low, high, min(count, len(remaining))):
            campaign_id, campaign_points = min(
                remaining.items(),
                key=lambda item: (
                    abs(math.log(campaign_velocity(item[1]) / target)),
                    relative_acceleration_error(lowest_error_point(item[1])),
                ),
            )
            selected_campaigns.append(campaign_points)
            del remaining[campaign_id]
        if len(selected_campaigns) < count and remaining:
            selected_campaigns.extend(
                sorted(
                    remaining.values(),
                    key=lambda campaign_points: (
                        relative_acceleration_error(lowest_error_point(campaign_points)),
                        campaign_velocity(campaign_points),
                    ),
                )[: count - len(selected_campaigns)]
            )

    selected = [lowest_error_point(campaign_points) for campaign_points in selected_campaigns]
    selected.sort(key=lambda point: point.velocity_cm_s)
    return (
        selected,
        f"one lowest-relative-error observation from each of {len(selected)} distinct log-spaced LAMMPS campaigns",
    )


def select_fit_points_by_regime(
    points: list[DataPoint],
    count: int,
    min_velocity_cm_s: float,
    max_velocity_cm_s: float,
) -> tuple[list[DataPoint], dict[int, str]]:
    selected: list[DataPoint] = []
    descriptions: dict[int, str] = {}
    for condition in sorted({point.condition for point in points}):
        group = [point for point in points if point.condition == condition]
        condition_points, description = select_regime_campaign_points(
            group,
            count,
            min_velocity_cm_s,
            max_velocity_cm_s,
        )
        selected.extend(condition_points)
        descriptions[condition] = description
    return selected, descriptions


def make_drag_with_cutoffs(
    condition: int,
    acipc: float,
    impact_parameter_cutoff_m: float,
    vres: int,
    rhores: int,
    ures: int,
    dphires: int,
    cutoff_radius_factor: float,
):
    if acipc <= 0.0 or impact_parameter_cutoff_m <= 0.0:
        raise ValueError("acipc and impact_parameter_cutoff_m must be positive")
    drag = make_drag(
        condition,
        vres=vres,
        rhores=rhores,
        ures=ures,
        dphires=dphires,
        cutoff_radius_factor=cutoff_radius_factor,
        acipc=acipc,
    )
    rhomax_fraction = impact_parameter_cutoff_m * drag.ustart
    if not 0.0 < rhomax_fraction <= 0.4:
        outer_radius_m = 1.0 / drag.ustart
        raise ValueError(
            "impact_parameter_cutoff_m must satisfy "
            f"0 < bmax <= 0.4 * outer_radius; got bmax={impact_parameter_cutoff_m:.6e} m, "
            f"outer_radius={outer_radius_m:.6e} m"
        )
    drag.rhomax_fraction = rhomax_fraction
    return drag


def prediction_metadata(drag, acipc: float, impact_parameter_cutoff_m: float) -> dict[str, float]:
    interparticle_spacing_m = 1.0 / drag.ustart
    ion_length_m = ion_screening_length(drag)
    yukawa_length_m = 1.0 / drag.k0
    angle_radius_cutoff_m = acipc * impact_parameter_cutoff_m
    return {
        "acipc": float(acipc),
        "impact_parameter_cutoff_m": float(impact_parameter_cutoff_m),
        "angle_radius_cutoff_m": float(angle_radius_cutoff_m),
        "rhomax_fraction_of_interparticle_spacing": float(impact_parameter_cutoff_m / interparticle_spacing_m),
        "rhomax_fraction_of_debye_length": float(impact_parameter_cutoff_m / drag.lD),
        "rhomax_fraction_of_ion_screening_length": float(impact_parameter_cutoff_m / ion_length_m),
        "rhomax_fraction_of_yukawa_screening_length": float(impact_parameter_cutoff_m / yukawa_length_m),
        "angle_radius_fraction_of_interparticle_spacing": float(angle_radius_cutoff_m / interparticle_spacing_m),
        "dragbase_rhomax_fraction_of_outer_radius": float(drag.rhomax_fraction),
        "electron_debye_radius_m": float(drag.lD),
        "ion_screening_length_m": float(ion_length_m),
        "yukawa_screening_length_m": float(yukawa_length_m),
        "outer_radius_m": float(interparticle_spacing_m),
    }


def run_fit_point_case(
    task: tuple[int, float, float, DataPoint, int, int, int, int, float]
) -> dict[str, object]:
    condition, acipc, bmax_m, point, vres, rhores, ures, dphires, cutoff_radius_factor = task
    drag = make_drag_with_cutoffs(condition, acipc, bmax_m, vres, rhores, ures, dphires, cutoff_radius_factor)
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
        **prediction_metadata(drag, acipc, bmax_m),
    }


def model_rows_parallel(
    pool: ProcessPoolExecutor,
    condition: int,
    acipc: float,
    bmax_m: float,
    points: list[DataPoint],
    vres: int,
    rhores: int,
    ures: int,
    dphires: int,
    cutoff_radius_factor: float,
    progress: bool,
    eval_index: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    tasks = [
        (condition, acipc, bmax_m, point, vres, rhores, ures, dphires, cutoff_radius_factor)
        for point in points
    ]
    rows = list(pool.map(run_fit_point_case, tasks))
    residuals = []
    for point_index, row in enumerate(rows, start=1):
        progress_print(
            progress,
            "[residual] "
            f"condition={condition} acipc={acipc:.6g} bmax={bmax_m:.6e} m eval={eval_index} "
            f"point={point_index}/{len(rows)} v={float(row['velocity_cm_s']):.6e} cm/s "
            f"data={float(row['data_acceleration_cm_s2']):.6e} "
            f"model={float(row['model_acceleration_cm_s2']):.6e} cm/s^2 "
            f"weighted={float(row['weighted_log_residual']):.6g}",
        )
        weighted = float(row["weighted_log_residual"])
        residuals.append(weighted if np.isfinite(weighted) else 1.0e30)
    return rows, np.array(residuals, dtype=float)


def covariance_matrix(result, n_points: int) -> np.ndarray:
    if result.jac is None or result.jac.size == 0:
        return np.full((2, 2), math.nan)
    dof = max(1, n_points - len(result.x))
    reduced_chi2 = float(np.sum(np.square(result.fun)) / dof)
    try:
        return np.linalg.inv(result.jac.T @ result.jac) * reduced_chi2
    except np.linalg.LinAlgError:
        return np.full((2, 2), math.nan)


def evaluate_fit_parallel(
    pool: ProcessPoolExecutor,
    condition: int,
    points: list[DataPoint],
    acipc_initial: float,
    acipc_min: float,
    acipc_max: float,
    bmax_initial_m: float,
    bmax_min_m: float,
    bmax_max_m: float,
    vres: int,
    rhores: int,
    ures: int,
    dphires: int,
    cutoff_radius_factor: float,
    max_nfev: int,
    progress: bool,
) -> dict[str, object]:
    eval_counter = 0
    x0 = np.log(np.array([acipc_initial, bmax_initial_m], dtype=float))
    lower = np.log(np.array([acipc_min, bmax_min_m], dtype=float))
    upper = np.log(np.array([acipc_max, bmax_max_m], dtype=float))
    x0 = np.minimum(np.maximum(x0, lower), upper)
    progress_print(
        progress,
        f"[fit start] condition={condition} acipc_initial={math.exp(x0[0]):.6g} "
        f"bmax_initial={math.exp(x0[1]):.6e} m points={len(points)}",
    )

    def residual_vector(log_params: np.ndarray) -> np.ndarray:
        nonlocal eval_counter
        eval_counter += 1
        acipc = float(math.exp(log_params[0]))
        bmax_m = float(math.exp(log_params[1]))
        try:
            _, residuals = model_rows_parallel(
                pool,
                condition,
                acipc,
                bmax_m,
                points,
                vres,
                rhores,
                ures,
                dphires,
                cutoff_radius_factor,
                progress,
                eval_counter,
            )
            return residuals
        except Exception as exc:
            progress_print(progress, f"[fit eval failed] condition={condition} acipc={acipc:.6g} bmax={bmax_m:.6e} error={exc!r}")
            return np.full(len(points), 1.0e30, dtype=float)

    try:
        result = least_squares(
            residual_vector,
            x0=x0,
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=max_nfev,
        )
        best_acipc = float(math.exp(result.x[0]))
        best_bmax_m = float(math.exp(result.x[1]))
        prediction_rows, weighted_residuals = model_rows_parallel(
            pool,
            condition,
            best_acipc,
            best_bmax_m,
            points,
            vres,
            rhores,
            ures,
            dphires,
            cutoff_radius_factor,
            progress,
            eval_counter + 1,
        )
        log_residuals = np.array(
            [float(row["log_residual"]) for row in prediction_rows if np.isfinite(row["log_residual"])],
            dtype=float,
        )
    except Exception as exc:
        progress_print(progress, f"[fit failed] condition={condition} error={exc!r}")
        return {
            "condition": condition,
            "acipc": math.nan,
            "impact_parameter_cutoff_m": math.nan,
            "score": math.nan,
            "rmse_log": math.nan,
            "n_points": 0,
            "optimizer_success": False,
            "optimizer_message": repr(exc),
            "optimizer_nfev": eval_counter,
            "prediction_rows": [{"condition": condition, "status": "failed", "error": repr(exc)}],
        }

    cov_log = covariance_matrix(result, len(weighted_residuals))
    acipc_sigma = best_acipc * math.sqrt(float(cov_log[0, 0])) if np.isfinite(cov_log[0, 0]) else math.nan
    bmax_sigma_m = best_bmax_m * math.sqrt(float(cov_log[1, 1])) if np.isfinite(cov_log[1, 1]) else math.nan
    corr = cov_log[0, 1] / math.sqrt(cov_log[0, 0] * cov_log[1, 1]) if np.all(np.isfinite(cov_log)) and cov_log[0, 0] > 0 and cov_log[1, 1] > 0 else math.nan
    score = float(np.mean(np.square(weighted_residuals))) if len(weighted_residuals) else math.nan
    rmse_log = float(np.sqrt(np.mean(np.square(log_residuals)))) if len(log_residuals) else math.nan
    progress_print(
        progress,
        f"[fit done] condition={condition} acipc={best_acipc:.6g} bmax={best_bmax_m:.6e} m "
        f"score={score:.6g} rmse_log={rmse_log:.6g} nfev={result.nfev} success={result.success}",
    )
    return {
        "condition": condition,
        "acipc": best_acipc,
        "acipc_sigma": acipc_sigma,
        "impact_parameter_cutoff_m": best_bmax_m,
        "impact_parameter_cutoff_sigma_m": bmax_sigma_m,
        "log_parameter_correlation": corr,
        "score": score,
        "rmse_log": rmse_log,
        "n_points": len(weighted_residuals),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "prediction_rows": prediction_rows,
    }


def run_curve_case(
    task: tuple[int, float, float, float, int, int, int, int, float]
) -> dict[str, object]:
    condition, acipc, bmax_m, velocity_cm_s, vres, rhores, ures, dphires, cutoff_radius_factor = task
    try:
        drag = make_drag_with_cutoffs(condition, acipc, bmax_m, vres, rhores, ures, dphires, cutoff_radius_factor)
        force_n = quiet_drag(drag, velocity_cm_s * CM_PER_S_TO_M_PER_S)
        model_acceleration = abs(force_n / drag.ms) * 100.0
        return {
            "condition": condition,
            "velocity_cm_s": velocity_cm_s,
            "model_acceleration_cm_s2": model_acceleration,
            "drag_N": force_n,
            "status": "ok",
            "error": "",
            **prediction_metadata(drag, acipc, bmax_m),
        }
    except Exception as exc:
        return {
            "condition": condition,
            "velocity_cm_s": velocity_cm_s,
            "model_acceleration_cm_s2": math.nan,
            "drag_N": math.nan,
            "status": "failed",
            "error": repr(exc),
        }


def default_bounds_for_condition(condition: int, cutoff_radius_factor: float) -> tuple[float, float, float]:
    probe = make_drag(condition, cutoff_radius_factor=cutoff_radius_factor)
    outer_radius_m = 1.0 / probe.ustart
    return 1.0e-4 * outer_radius_m, 0.3 * outer_radius_m, 0.4 * outer_radius_m


def summary_row_from_result(result: dict[str, object], selection: str, cutoff_radius_factor: float) -> dict[str, object]:
    best = next((row for row in result["prediction_rows"] if row.get("status") == "ok"), None)
    condition = int(result["condition"])
    if best is None:
        return {
            "condition": condition,
            "condition_label": condition_label(condition),
            "status": "failed",
            "error": result.get("optimizer_message", ""),
        }
    bmax_m = float(result["impact_parameter_cutoff_m"])
    bmax_sigma_m = float(result["impact_parameter_cutoff_sigma_m"])
    interparticle_spacing_m = float(best["outer_radius_m"])
    return {
        "condition": condition,
        "condition_label": condition_label(condition),
        "coupling_parameter": COUPLING_PARAMETER.get(condition, math.nan),
        "status": "ok",
        "fit_parameters": "acipc,impact_parameter_cutoff_m",
        "acipc": result["acipc"],
        "acipc_sigma": result["acipc_sigma"],
        "impact_parameter_cutoff_m": bmax_m,
        "impact_parameter_cutoff_sigma_m": bmax_sigma_m,
        "angle_radius_cutoff_m": best["angle_radius_cutoff_m"],
        "rhomax_fraction_of_interparticle_spacing": best["rhomax_fraction_of_interparticle_spacing"],
        "rhomax_fraction_of_interparticle_spacing_sigma": bmax_sigma_m / interparticle_spacing_m if np.isfinite(bmax_sigma_m) else math.nan,
        "rhomax_fraction_of_debye_length": best["rhomax_fraction_of_debye_length"],
        "rhomax_fraction_of_ion_screening_length": best["rhomax_fraction_of_ion_screening_length"],
        "rhomax_fraction_of_yukawa_screening_length": best["rhomax_fraction_of_yukawa_screening_length"],
        "electron_debye_radius_m": best["electron_debye_radius_m"],
        "ion_screening_length_m": best["ion_screening_length_m"],
        "yukawa_screening_length_m": best["yukawa_screening_length_m"],
        "outer_radius_m": best["outer_radius_m"],
        "cutoff_radius_factor": cutoff_radius_factor,
        "log_parameter_correlation": result["log_parameter_correlation"],
        "score": result["score"],
        "rmse_log": result["rmse_log"],
        "n_points": result["n_points"],
        "fit_point_selection": selection,
        "optimizer_success": result["optimizer_success"],
        "optimizer_message": result["optimizer_message"],
        "optimizer_nfev": result["optimizer_nfev"],
    }


def plot_results(
    all_points: list[DataPoint],
    fit_points: list[DataPoint],
    curve_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    colors = {0: "red", 1: "orange", 2: "green", 3: "blue"}
    fig, ax = plt.subplots(figsize=(12, 8))
    for condition in sorted({point.condition for point in all_points}):
        color = colors.get(condition)
        group = [point for point in all_points if point.condition == condition]
        selected = {point.source for point in fit_points if point.condition == condition}
        curve = [row for row in curve_rows if int(row["condition"]) == condition and row["status"] == "ok"]
        curve.sort(key=lambda row: float(row["velocity_cm_s"]))
        summary = next((row for row in summary_rows if int(row["condition"]) == condition and row.get("status") == "ok"), None)
        label = f"condition {condition}"
        if summary:
            label += (
                f" acipc={float(summary['acipc']):.3g}, "
                f"bmax/aH={float(summary['impact_parameter_cutoff_m']) / (float(summary['outer_radius_m']) / float(summary['cutoff_radius_factor'])):.3g}"
            )
        ax.errorbar(
            [point.velocity_cm_s for point in group],
            [point.acceleration_cm_s2 for point in group],
            yerr=[point.acceleration_sigma_cm_s2 if np.isfinite(point.acceleration_sigma_cm_s2) else 0.0 for point in group],
            fmt=".",
            color=color,
            alpha=0.25,
            markersize=4,
            capsize=0,
        )
        fit_group = [point for point in group if point.source in selected]
        ax.scatter(
            [point.velocity_cm_s for point in fit_group],
            [point.acceleration_cm_s2 for point in fit_group],
            s=45,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
        )
        if curve:
            ax.plot(
                [float(row["velocity_cm_s"]) for row in curve],
                [float(row["model_acceleration_cm_s2"]) for row in curve],
                color=color,
                linewidth=2.0,
                label=label,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("velocity [cm/s]")
    ax.set_ylabel("acceleration [cm/s^2]")
    ax.set_title("Two-parameter fit: acipc and impact-parameter cutoff")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "acipc_fit.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit acipc and the impact-parameter cutoff length to LAMMPS acceleration data.")
    parser.add_argument("--conditions", nargs="+", type=int, default=list(ALL_CONDITIONS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--lammps-results", type=Path, default=REPO_ROOT / "unforced" / "dataarchive" / "nprun4_29" / "results.npy")
    parser.add_argument("--samples-per-lammps-fit", type=int, default=10)
    parser.add_argument("--fit-points-per-condition", type=int, default=8)
    parser.add_argument("--min-velocity-cm-s", type=float, default=1.0e2)
    parser.add_argument("--max-velocity-cm-s", type=float, default=1.0e8)
    parser.add_argument("--max-relative-sigma", type=float, default=10.0)
    parser.add_argument("--acipc-initial", type=float, default=1.0)
    parser.add_argument("--acipc-min", type=float, default=1.0e-2)
    parser.add_argument("--acipc-max", type=float, default=1.0e2)
    parser.add_argument("--bmax-initial-m", type=float)
    parser.add_argument("--bmax-min-m", type=float)
    parser.add_argument("--bmax-max-m", type=float)
    parser.add_argument("--cutoff-radius-factor", type=float, default=50.0)
    parser.add_argument("--max-fit-evaluations", type=int, default=30)
    parser.add_argument("--curve-points", type=int, default=24)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vres", type=int, default=201)
    parser.add_argument("--rhores", type=int, default=180)
    parser.add_argument("--ures", type=int, default=180)
    parser.add_argument("--dphires", type=int, default=180)
    args = parser.parse_args()

    if args.acipc_min <= 0.0 or args.acipc_max <= args.acipc_min:
        raise SystemExit("--acipc-min must be positive and --acipc-max must exceed it.")
    if args.cutoff_radius_factor <= 0.0:
        raise SystemExit("--cutoff-radius-factor must be positive.")

    requested_conditions = set(args.conditions)
    unknown_conditions = requested_conditions.difference(ALL_CONDITIONS)
    if unknown_conditions:
        raise SystemExit(f"Unknown condition indexes {sorted(unknown_conditions)}; valid conditions are {list(ALL_CONDITIONS)}.")

    if args.data_csv:
        all_points = load_points_from_csv(args.data_csv, requested_conditions)
    else:
        all_points = load_lammps_expfit_points(args.lammps_results, requested_conditions, args.samples_per_lammps_fit)
    all_points = filter_points(all_points, args.min_velocity_cm_s, args.max_velocity_cm_s, args.max_relative_sigma)
    fit_points, selection_by_condition = select_fit_points_by_regime(
        all_points,
        args.fit_points_per_condition,
        args.min_velocity_cm_s,
        args.max_velocity_cm_s,
    )
    if not fit_points:
        raise SystemExit("No usable fit points found. Check data path, filters, or --fit-points-per-condition.")

    points_by_condition = {
        condition: [point for point in fit_points if point.condition == condition]
        for condition in sorted(requested_conditions)
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fit_results = []
        for condition, points in sorted(points_by_condition.items()):
            if not points:
                continue
            default_bmin, default_binitial, default_bmax = default_bounds_for_condition(condition, args.cutoff_radius_factor)
            bmin = args.bmax_min_m if args.bmax_min_m is not None else default_bmin
            binitial = args.bmax_initial_m if args.bmax_initial_m is not None else default_binitial
            bmax = args.bmax_max_m if args.bmax_max_m is not None else default_bmax
            if bmin <= 0.0 or bmax <= bmin:
                raise SystemExit("--bmax-min-m must be positive and --bmax-max-m must exceed it.")
            fit_results.append(
                evaluate_fit_parallel(
                    pool,
                    condition,
                    points,
                    args.acipc_initial,
                    args.acipc_min,
                    args.acipc_max,
                    binitial,
                    bmin,
                    bmax,
                    args.vres,
                    args.rhores,
                    args.ures,
                    args.dphires,
                    args.cutoff_radius_factor,
                    args.max_fit_evaluations,
                    not args.quiet,
                )
            )

        prediction_rows = [row for result in fit_results for row in result["prediction_rows"]]
        summary_rows = [
            summary_row_from_result(
                result,
                selection_by_condition.get(int(result["condition"]), "lowest acceleration_sigma / acceleration"),
                args.cutoff_radius_factor,
            )
            for result in fit_results
        ]
        curve_velocities = condition_curve_velocities(all_points, args.curve_points)
        curve_tasks = []
        for summary in summary_rows:
            if summary.get("status") != "ok":
                continue
            condition = int(summary["condition"])
            for velocity_cm_s in curve_velocities[condition]:
                curve_tasks.append(
                    (
                        condition,
                        float(summary["acipc"]),
                        float(summary["impact_parameter_cutoff_m"]),
                        float(velocity_cm_s),
                        args.vres,
                        args.rhores,
                        args.ures,
                        args.dphires,
                        args.cutoff_radius_factor,
                    )
                )
        curve_rows = list(pool.map(run_curve_case, curve_tasks))

    write_csv(OUTDIR / "acipc_fit_predictions.csv", prediction_rows)
    write_csv(OUTDIR / "acipc_fit_curve.csv", curve_rows)
    write_csv(OUTDIR / "acipc_fit_summary.csv", summary_rows)
    plot_results(all_points, fit_points, curve_rows, summary_rows)

    for row in summary_rows:
        if row.get("status") != "ok":
            print(f"condition {row['condition']}: failed: {row.get('error', '')}")
            continue
        print(
            f"condition {row['condition']}: acipc={float(row['acipc']):.6g}, "
            f"bmax={float(row['impact_parameter_cutoff_m']):.6e} m, "
            f"bmax/aH={float(row['rhomax_fraction_of_interparticle_spacing']):.6g}, "
            f"angle_cutoff={float(row['angle_radius_cutoff_m']):.6e} m, "
            f"fit points={row['n_points']}, RMSE(log)={float(row['rmse_log']):.4g}"
        )


if __name__ == "__main__":
    freeze_support()
    main()
