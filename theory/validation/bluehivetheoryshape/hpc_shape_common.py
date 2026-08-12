from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from common import CM_PER_S_TO_M_PER_S, DEFAULT_CUTOFF_RADIUS_FACTOR, condition_label, make_drag, quiet_drag

OUTDIR = THIS_DIR
TASKS_CSV = OUTDIR / "bluehive_shape_tasks.csv"
RESULTS_DIR = OUTDIR / "task_results"
SLURM_DIR = OUTDIR / "slurm"

CONDITIONS = (0, 1, 2, 3)
BMAX_OVER_AH = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
BASE_BMAX_OVER_AH = 0.1
BASE_RHORES = 10
BASE_UDPHIRES = 100
DEFAULT_VRES = 201
DEFAULT_CURVE_POINTS = 30
DEFAULT_MIN_VELOCITY_CM_S = 1.0e4
DEFAULT_MAX_VELOCITY_CM_S = 1.0e8


def log_velocity_grid(minimum_cm_s: float, maximum_cm_s: float, points: int) -> list[float]:
    if minimum_cm_s <= 0.0 or maximum_cm_s <= minimum_cm_s:
        raise ValueError("velocity bounds must satisfy 0 < min < max")
    if points < 3:
        raise ValueError("curve-points must be at least 3")
    return [float(value) for value in np.geomspace(minimum_cm_s, maximum_cm_s, points)]


def impact_resolution_for_bmax(
    bmax_over_aH: float,
    base_bmax_over_aH: float = BASE_BMAX_OVER_AH,
    base_resolution: int = BASE_RHORES,
) -> int:
    """Scale equal-area impact bins so local physical bin width is comparable."""
    if base_bmax_over_aH <= 0.0:
        raise ValueError("base_bmax_over_aH must be positive")
    if base_resolution < 1:
        raise ValueError("base_resolution must be positive")
    if not math.isfinite(bmax_over_aH) or bmax_over_aH <= 0.0:
        raise ValueError("bmax/aH must be positive and finite")
    scale = bmax_over_aH / base_bmax_over_aH
    return max(2, int(math.ceil(base_resolution * scale**2)))


def angle_resolution_for_bmax(
    bmax_over_aH: float,
    base_bmax_over_aH: float = BASE_BMAX_OVER_AH,
    base_resolution: int = BASE_UDPHIRES,
) -> int:
    """Scale radial/scattering quadratures with the cutoff length."""
    if base_bmax_over_aH <= 0.0:
        raise ValueError("base_bmax_over_aH must be positive")
    if base_resolution < 1:
        raise ValueError("base_resolution must be positive")
    if not math.isfinite(bmax_over_aH) or bmax_over_aH <= 0.0:
        raise ValueError("bmax/aH must be positive and finite")
    scale = bmax_over_aH / base_bmax_over_aH
    return max(2, int(math.ceil(base_resolution * scale)))


def task_resolution(bmax_over_aH: float, vres: int = DEFAULT_VRES) -> dict[str, int]:
    impact_resolution = impact_resolution_for_bmax(bmax_over_aH)
    angle_resolution = angle_resolution_for_bmax(bmax_over_aH)
    return {
        "vres": int(vres),
        "rhores": impact_resolution,
        "ures": angle_resolution,
        "dphires": angle_resolution,
    }


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def task_output_path(task_id: int) -> Path:
    return RESULTS_DIR / f"task_{task_id:05d}.csv"


def run_curve_point(task: dict[str, str | int | float], bmax_over_aH: float) -> dict[str, object]:
    condition = int(task["condition"])
    velocity_cm_s = float(task["velocity_cm_s"])
    vres = int(task["vres"])
    resolution = task_resolution(bmax_over_aH, vres=vres)
    rhores = resolution["rhores"]
    ures = resolution["ures"]
    dphires = resolution["dphires"]

    rhomax_fraction = bmax_over_aH / DEFAULT_CUTOFF_RADIUS_FACTOR
    if rhomax_fraction > 1.0:
        raise ValueError(
            f"bmax/aH={bmax_over_aH:g} exceeds launch radius "
            f"{DEFAULT_CUTOFF_RADIUS_FACTOR:g} aH"
        )

    drag = make_drag(
        condition,
        vres=vres,
        rhores=rhores,
        ures=ures,
        dphires=dphires,
        rhomax_fraction=rhomax_fraction,
        cutoff_radius_factor=DEFAULT_CUTOFF_RADIUS_FACTOR,
    )
    force_n = quiet_drag(drag, velocity_cm_s * CM_PER_S_TO_M_PER_S)
    acceleration_cm_s2 = abs(force_n / drag.ms) * 100.0
    hydrogen_interparticle_spacing_m = 1.0 / (DEFAULT_CUTOFF_RADIUS_FACTOR * drag.ustart)
    finite_radius_m = 1.0 / drag.ustart
    impact_parameter_cutoff_m = bmax_over_aH * hydrogen_interparticle_spacing_m

    return {
        "task_id": int(task["task_id"]),
        "condition": condition,
        "condition_label": condition_label(condition),
        "velocity_cm_s": velocity_cm_s,
        "velocity_m_s": velocity_cm_s * CM_PER_S_TO_M_PER_S,
        "bmax_over_hydrogen_interparticle_spacing": bmax_over_aH,
        "rhomax_fraction_of_naive_outer_radius": rhomax_fraction,
        "base_bmax_over_aH_for_resolution": BASE_BMAX_OVER_AH,
        "base_rhores_at_bmax_0p1": BASE_RHORES,
        "base_ures_dphires_at_bmax_0p1": BASE_UDPHIRES,
        "rhores_scaling": "base_rhores * (bmax/0.1)^2",
        "ures_dphires_scaling": "base_ures_dphires * (bmax/0.1)",
        "cutoff_radius_factor": DEFAULT_CUTOFF_RADIUS_FACTOR,
        "hydrogen_interparticle_spacing_m": hydrogen_interparticle_spacing_m,
        "impact_parameter_cutoff_m": impact_parameter_cutoff_m,
        "finite_radius_m": finite_radius_m,
        "drag_N": force_n,
        "absolute_drag_N": abs(force_n),
        "model_acceleration_cm_s2": acceleration_cm_s2,
        "status": "ok" if math.isfinite(force_n) and force_n != 0.0 else "invalid_drag",
        "vres": vres,
        "rhores": rhores,
        "ures": ures,
        "dphires": dphires,
        "host": os.environ.get("HOSTNAME", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }


def run_condition_velocity_task(task: dict[str, str | int | float]) -> list[dict[str, object]]:
    return [run_curve_point(task, bmax_over_aH) for bmax_over_aH in BMAX_OVER_AH]


def add_shape_columns(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["condition"]), float(row["bmax_over_hydrogen_interparticle_spacing"]))
        groups.setdefault(key, []).append(row)

    for group in groups.values():
        group.sort(key=lambda row: float(row["velocity_cm_s"]))
        drag = [float(row["absolute_drag_N"]) for row in group]
        velocity = [float(row["velocity_cm_s"]) for row in group]
        valid_indices = [
            index
            for index, value in enumerate(drag)
            if math.isfinite(value) and value > 0.0 and math.isfinite(velocity[index]) and velocity[index] > 0.0
        ]
        peak_drag = max((drag[index] for index in valid_indices), default=math.nan)
        slope = [math.nan] * len(group)
        if len(valid_indices) >= 3:
            for valid_position, index in enumerate(valid_indices):
                if valid_position == 0:
                    left = valid_indices[valid_position]
                    right = valid_indices[valid_position + 1]
                elif valid_position == len(valid_indices) - 1:
                    left = valid_indices[valid_position - 1]
                    right = valid_indices[valid_position]
                else:
                    left = valid_indices[valid_position - 1]
                    right = valid_indices[valid_position + 1]
                slope[index] = (math.log(drag[right]) - math.log(drag[left])) / (
                    math.log(velocity[right]) - math.log(velocity[left])
                )
        for index, row in enumerate(group):
            value = float(row["absolute_drag_N"])
            row["peak_drag_N_for_condition_and_bmax"] = peak_drag
            row["drag_normalized_to_peak"] = value / peak_drag if math.isfinite(peak_drag) and peak_drag > 0.0 else math.nan
            row["local_log_slope_dlogdrag_dlogv"] = slope[index]
