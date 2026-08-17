from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from common import CM_PER_S_TO_M_PER_S, DEFAULT_CUTOFF_RADIUS_FACTOR, condition_label, make_drag

OUTDIR = THIS_DIR
TASKS_CSV = OUTDIR / "dungeon_shape_tasks.csv"
RESULTS_DIR = OUTDIR / "task_results"

CONDITIONS = (0, 1, 2, 3)
BMAX_OVER_AH = (0.1, 0.3, 1.0, 3.0, 10.0)
BASE_BMAX_OVER_AH = 0.1
SHARED_BMAX_OVER_AH = 10.0
SHARED_RHORES = 10000
MIN_IMPACT_OVER_MAX = 1.0e-6
IMPACT_GRID = "log"
BASE_RHORES = SHARED_RHORES
BASE_UDPHIRES = 100
DEFAULT_VRES = 201
DEFAULT_CURVE_POINTS = 25
DEFAULT_MIN_VELOCITY_CM_S = 1.0e5
DEFAULT_MAX_VELOCITY_CM_S = 1.0e8


def optional_cupy():
    try:
        import cupy as cp
    except Exception:
        return None
    return cp


def can_use_gpu_reduction(cp, gpu_id: int) -> bool:
    if cp is None or gpu_id < 0:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > gpu_id
    except Exception:
        return False


def log_velocity_grid(minimum_cm_s: float, maximum_cm_s: float, points: int) -> list[float]:
    if minimum_cm_s <= 0.0 or maximum_cm_s <= minimum_cm_s:
        raise ValueError("velocity bounds must satisfy 0 < min < max")
    if points < 3:
        raise ValueError("curve-points must be at least 3")
    return [float(value) for value in np.geomspace(minimum_cm_s, maximum_cm_s, points)]


def task_resolution(
    vres: int = DEFAULT_VRES,
    rhores: int = SHARED_RHORES,
    ures: int = BASE_UDPHIRES,
    dphires: int = BASE_UDPHIRES,
) -> dict[str, int]:
    return {
        "vres": int(vres),
        "rhores": int(rhores),
        "ures": int(ures),
        "dphires": int(dphires),
    }


def speed_grid_and_weights(drag, drift_velocity_m_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    sigmav = math.sqrt(drag.kb * drag.T / drag.mu)
    width = drag.vrel_sigma_width * sigmav
    vmin = drift_velocity_m_s - width
    vmax = drift_velocity_m_s + width
    speed_min = 0.0 if vmin <= 0.0 <= vmax else min(abs(vmin), abs(vmax))
    speed_max = max(abs(vmin), abs(vmax))
    if drag.vres < 1 or speed_max <= speed_min:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), math.nan

    ds = (speed_max - speed_min) / drag.vres
    speeds = speed_min + (np.arange(drag.vres, dtype=np.float64) + 0.5) * ds
    norm = math.sqrt(drag.mu / (2.0 * math.pi * drag.kb * drag.T))
    positive = np.zeros_like(speeds)
    negative = np.zeros_like(speeds)
    positive_mask = (vmin <= speeds) & (speeds <= vmax)
    negative_mask = (vmin <= -speeds) & (-speeds <= vmax)
    positive[positive_mask] = norm * np.exp(
        -drag.mu * np.square(speeds[positive_mask] - drift_velocity_m_s) / (2.0 * drag.kb * drag.T)
    )
    negative[negative_mask] = norm * np.exp(
        -drag.mu * np.square(-speeds[negative_mask] - drift_velocity_m_s) / (2.0 * drag.kb * drag.T)
    )
    return speeds, positive - negative, ds


def launch_impact_grid(maximum_m: float, count: int, minimum_over_maximum: float) -> tuple[np.ndarray, np.ndarray]:
    if maximum_m <= 0.0 or count < 1:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if not 0.0 < minimum_over_maximum < 1.0:
        raise ValueError("MIN_IMPACT_OVER_MAX must satisfy 0 < value < 1")
    if count == 1:
        edges = np.array([0.0, maximum_m], dtype=np.float64)
    else:
        positive_edges = np.geomspace(maximum_m * minimum_over_maximum, maximum_m, count, dtype=np.float64)
        edges = np.concatenate(([0.0], positive_edges))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return centers, widths


def write_rows_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def task_output_path(task_id: int) -> Path:
    return RESULTS_DIR / f"task_{task_id:05d}.csv"


def row_for_cutoff(
    task: dict[str, str | int | float],
    drag,
    force_n: float,
    bmax_over_aH: float,
    elapsed_s: float,
    resolution: dict[str, int],
) -> dict[str, object]:
    condition = int(task["condition"])
    velocity_cm_s = float(task["velocity_cm_s"])
    rhores = resolution["rhores"]
    ures = resolution["ures"]
    dphires = resolution["dphires"]
    rhomax_fraction = bmax_over_aH / DEFAULT_CUTOFF_RADIUS_FACTOR
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
        "base_bmax_over_aH_for_resolution": SHARED_BMAX_OVER_AH,
        "base_rhores_at_bmax_0p1": SHARED_RHORES,
        "base_ures_dphires_at_bmax_0p1": BASE_UDPHIRES,
        "rhores_scaling": "shared log grid at bmax/aH=10, partial sums for smaller cutoffs",
        "ures_dphires_scaling": "fixed for shared-grid task",
        "impact_grid": IMPACT_GRID,
        "min_impact_over_max": MIN_IMPACT_OVER_MAX,
        "max_bmax_over_aH_for_shared_grid": SHARED_BMAX_OVER_AH,
        "cumulative_bmax_grid": True,
        "cutoff_radius_factor": DEFAULT_CUTOFF_RADIUS_FACTOR,
        "hydrogen_interparticle_spacing_m": hydrogen_interparticle_spacing_m,
        "impact_parameter_cutoff_m": impact_parameter_cutoff_m,
        "finite_radius_m": finite_radius_m,
        "drag_N": force_n,
        "absolute_drag_N": abs(force_n),
        "model_acceleration_cm_s2": acceleration_cm_s2,
        "status": "ok" if math.isfinite(force_n) and force_n != 0.0 else "invalid_drag",
        "vres": resolution["vres"],
        "rhores": rhores,
        "ures": ures,
        "dphires": dphires,
        "elapsed_s": elapsed_s,
        "host": os.environ.get("HOSTNAME", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }


def run_condition_velocity_task(task: dict[str, str | int | float]) -> list[dict[str, object]]:
    task_start = time.perf_counter()
    condition = int(task["condition"])
    velocity_cm_s = float(task["velocity_cm_s"])
    vres = int(task["vres"])
    rhores = int(task.get("rhores", SHARED_RHORES))
    ures = int(task.get("ures", BASE_UDPHIRES))
    dphires = int(task.get("dphires", BASE_UDPHIRES))
    gpu_id = int(task.get("gpu_id", os.environ.get("DUNGEON_GPU_ID", "-1")))
    cp = optional_cupy()
    use_gpu_reduction = can_use_gpu_reduction(cp, gpu_id)
    resolution = task_resolution(vres=vres, rhores=rhores, ures=ures, dphires=dphires)
    max_bmax_over_aH = max(BMAX_OVER_AH)
    if max_bmax_over_aH != SHARED_BMAX_OVER_AH:
        raise ValueError("BMAX_OVER_AH must include SHARED_BMAX_OVER_AH as its largest cutoff")
    rhomax_fraction = SHARED_BMAX_OVER_AH / DEFAULT_CUTOFF_RADIUS_FACTOR
    if rhomax_fraction > 1.0:
        raise ValueError(
            f"bmax/aH={SHARED_BMAX_OVER_AH:g} exceeds launch radius "
            f"{DEFAULT_CUTOFF_RADIUS_FACTOR:g} aH"
        )

    print(
        "[task start] "
        f"task_id={int(task['task_id'])} condition={condition} velocity_cm_s={velocity_cm_s:.6e} "
        f"bmax/aH<= {SHARED_BMAX_OVER_AH:g} bmax_count={len(BMAX_OVER_AH)} "
        f"vres={resolution['vres']} rhores={resolution['rhores']} "
        f"ures={resolution['ures']} dphires={resolution['dphires']} impact_grid={IMPACT_GRID} "
        f"gpu_reduction={use_gpu_reduction} gpu_id={gpu_id if use_gpu_reduction else 'cpu'}",
        flush=True,
    )

    drag = make_drag(
        condition,
        rhomax_fraction=rhomax_fraction,
        cutoff_radius_factor=DEFAULT_CUTOFF_RADIUS_FACTOR,
        **resolution,
    )
    velocity_m_s = velocity_cm_s * CM_PER_S_TO_M_PER_S
    speeds, weights, ds = speed_grid_and_weights(drag, velocity_m_s)
    hydrogen_interparticle_spacing_m = 1.0 / (DEFAULT_CUTOFF_RADIUS_FACTOR * drag.ustart)
    cutoffs_m = {bmax: bmax * hydrogen_interparticle_spacing_m for bmax in BMAX_OVER_AH}
    max_cutoff_m = SHARED_BMAX_OVER_AH * hydrogen_interparticle_spacing_m
    launch_centers_m, launch_widths_m = launch_impact_grid(max_cutoff_m, resolution["rhores"], MIN_IMPACT_OVER_MAX)
    integrals = {bmax: 0.0 for bmax in BMAX_OVER_AH}

    active_speeds = 0
    for speed_index, (speed, weight) in enumerate(zip(speeds, weights), 1):
        if speed <= 0.0 or weight == 0.0:
            continue
        active_speeds += 1
        if active_speeds == 1 or active_speeds % 25 == 0:
            print(
                "[task progress] "
                f"task_id={int(task['task_id'])} active_speed={active_speeds} "
                f"speed_index={speed_index}/{len(speeds)} speed_m_s={speed:.6e}",
                flush=True,
            )

        energy = 0.5 * drag.mu * speed**2 + drag.E0Y
        vinf = math.sqrt(energy / (0.5 * drag.mu))
        rhoarr = launch_centers_m * speed / vinf
        half_theta = drag.scattering_half_angle(rhoarr, energy)
        if not np.all(np.isfinite(half_theta)):
            raise FloatingPointError("non-finite scattering angle in cumulative impact integral")
        if use_gpu_reduction:
            with cp.cuda.Device(gpu_id):
                theta_gpu = cp.asarray(half_theta)
                bin_contribution = cp.asarray(launch_centers_m * launch_widths_m) * speed**2 * weight * (
                    2.0 * cp.square(cp.sin(theta_gpu))
                )
                cumulative = cp.asnumpy(cp.cumsum(bin_contribution))
        else:
            bin_contribution = launch_centers_m * launch_widths_m * speed**2 * weight * (
                2.0 * np.square(np.sin(half_theta))
            )
            cumulative = np.cumsum(bin_contribution)
        for bmax, cutoff in cutoffs_m.items():
            last = np.searchsorted(launch_centers_m, cutoff, side="right")
            if last:
                integrals[bmax] += float(cumulative[last - 1])

    if not math.isfinite(ds):
        raise FloatingPointError("invalid velocity-grid spacing")
    prefactor = 2.0 * math.pi * drag.nh * drag.mu * ds
    elapsed_s = time.perf_counter() - task_start

    rows = []
    for index, bmax_over_aH in enumerate(BMAX_OVER_AH, 1):
        force_n = prefactor * integrals[bmax_over_aH]
        print(
            "[task progress] "
            f"task_id={int(task['task_id'])} bmax_index={index}/{len(BMAX_OVER_AH)} "
            f"bmax/aH={bmax_over_aH:g} drag_N={force_n:.6e}",
            flush=True,
        )
        rows.append(row_for_cutoff(task, drag, force_n, bmax_over_aH, elapsed_s, resolution))

    for row in rows:
        row["execution_target"] = "cpu_scattering_gpu_reduction" if use_gpu_reduction else "cpu"
        row["gpu_id"] = gpu_id if use_gpu_reduction else ""

    print(
        "[task done] "
        f"task_id={int(task['task_id'])} active_speeds={active_speeds} elapsed_min={elapsed_s/60.0:.2f}",
        flush=True,
    )
    return rows


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
