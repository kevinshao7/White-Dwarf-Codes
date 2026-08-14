# Run from repository root:
# python .\theory\validation\8_14_monotonicity\run_monotonicity_checks.py
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import sys
import time
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from common import CM_PER_S_TO_M_PER_S, DEFAULT_CUTOFF_RADIUS_FACTOR, condition_label, make_drag, quiet_drag


DEFAULT_BMAX_OVER_AH = (0.5, 1.0, 2.0, 5.0)
DEFAULT_EVALS = (10, 100, 1000, 10000)
DEFAULT_BIN_RHORES = (100, 1000)
DEFAULT_VELOCITIES_CM_S = (1.0e8,)
IMPACT_GRID_EQUAL_AREA = "equal-area"
IMPACT_GRID_LOG = "log"
SCAN_RHORES_ONLY = "rhores_only"
SCAN_ANGLE_ONLY = "angle_only"


def positive_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated numbers") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("provide positive finite values")
    return values


def positive_int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("provide positive integer values")
    return values


def resolution(vres: int, rhores: int, angle_evals: int) -> dict[str, int]:
    return {
        "vres": int(vres),
        "rhores": max(2, int(rhores)),
        "ures": max(2, int(angle_evals)),
        "dphires": max(2, int(angle_evals)),
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


def launch_impact_grid(
    maximum_m: float,
    count: int,
    mode: str,
    minimum_over_maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    if maximum_m <= 0.0 or count < 1:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if mode == IMPACT_GRID_EQUAL_AREA:
        edges = maximum_m * np.sqrt(np.linspace(0.0, 1.0, count + 1, dtype=np.float64))
    elif mode == IMPACT_GRID_LOG:
        if not 0.0 < minimum_over_maximum < 1.0:
            raise ValueError("--min-impact-over-max must satisfy 0 < value < 1 for log impact grids")
        first_edge = maximum_m * minimum_over_maximum
        if count == 1:
            edges = np.array([0.0, maximum_m], dtype=np.float64)
        else:
            positive_edges = np.geomspace(first_edge, maximum_m, count, dtype=np.float64)
            edges = np.concatenate(([0.0], positive_edges))
    else:
        raise ValueError(f"unknown impact grid mode {mode!r}")
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return centers, widths


def compute_cumulative_rows(task: dict[str, object]) -> list[dict[str, object]]:
    start = time.perf_counter()
    res = resolution(vres=int(task["vres"]), rhores=int(task["rhores"]), angle_evals=int(task["angle_evals"]))
    bmax_values = tuple(float(value) for value in task["bmax_over_aH_values"])
    max_bmax = max(bmax_values)
    print(
        "[worker start] "
        f"scan={task['scan_type']} condition={task['condition']} "
        f"v={float(task['velocity_cm_s']):.6e} cm/s bmax/aH<= {max_bmax:g} "
        f"line_evals={task['line_evals']} vres={res['vres']} rhores={res['rhores']} "
        f"ures={res['ures']} dphires={res['dphires']} impact_grid={task['impact_grid']}",
        flush=True,
    )

    rhomax_fraction = max_bmax / DEFAULT_CUTOFF_RADIUS_FACTOR
    drag = make_drag(
        int(task["condition"]),
        cutoff_radius_factor=DEFAULT_CUTOFF_RADIUS_FACTOR,
        rhomax_fraction=rhomax_fraction,
        **res,
    )
    velocity_cm_s = float(task["velocity_cm_s"])
    velocity_m_s = velocity_cm_s * CM_PER_S_TO_M_PER_S
    speeds, weights, ds = speed_grid_and_weights(drag, velocity_m_s)
    hydrogen_spacing_m = 1.0 / (DEFAULT_CUTOFF_RADIUS_FACTOR * drag.ustart)
    launch_radius_m = 1.0 / drag.ustart
    cutoffs_m = {bmax: bmax * hydrogen_spacing_m for bmax in bmax_values}
    integrals = {bmax: 0.0 for bmax in bmax_values}
    launch_centers_m, launch_widths_m = launch_impact_grid(
        maximum_m=max(cutoffs_m.values()),
        count=int(drag.rhores),
        mode=str(task["impact_grid"]),
        minimum_over_maximum=float(task["min_impact_over_max"]),
    )

    for speed, weight in zip(speeds, weights):
        if speed <= 0.0 or weight == 0.0:
            continue
        energy = 0.5 * drag.mu * speed**2 + drag.E0Y
        vinf = math.sqrt(energy / (0.5 * drag.mu))
        rhoarr = launch_centers_m * speed / vinf
        keep = launch_centers_m <= max(cutoffs_m.values())
        if not np.any(keep):
            continue
        last = np.where(keep)[0][-1] + 1
        rhoarr = rhoarr[:last]
        rhostart = launch_centers_m[:last]
        drhostart = launch_widths_m[:last]
        if len(rhostart) < 1:
            continue

        half_theta = drag.scattering_half_angle(rhoarr, energy)
        if not np.all(np.isfinite(half_theta)):
            raise FloatingPointError("non-finite scattering angle in cumulative impact integral")
        bin_contribution = rhostart * drhostart * speed**2 * weight * (2.0 * np.square(np.sin(half_theta)))
        for bmax, cutoff in cutoffs_m.items():
            integrals[bmax] += float(np.sum(bin_contribution[rhostart <= cutoff]))

    prefactor = 2.0 * math.pi * drag.nh * drag.mu * ds
    elapsed_s = time.perf_counter() - start

    rows = []
    for bmax in bmax_values:
        force_n = prefactor * integrals[bmax]
        rows.append(
            {
                "scan_type": task["scan_type"],
                "condition": int(task["condition"]),
                "condition_label": condition_label(int(task["condition"])),
                "velocity_cm_s": velocity_cm_s,
                "velocity_m_s": velocity_m_s,
                "bmax_over_aH": bmax,
                "line_evals": int(task["line_evals"]),
                "fixed_rhores": task["fixed_rhores"],
                "fixed_angle_evals": task["fixed_angle_evals"],
                "cumulative_bmax_grid": True,
                "impact_grid": task["impact_grid"],
                "min_impact_over_max": task["min_impact_over_max"],
                "max_bmax_over_aH_for_shared_grid": max_bmax,
                "drag_N": force_n,
                "absolute_drag_N": abs(force_n),
                "rhomax_fraction_of_launch_radius": rhomax_fraction,
                "hydrogen_interparticle_spacing_m": hydrogen_spacing_m,
                "launch_radius_m": launch_radius_m,
                "impact_parameter_cutoff_m": cutoffs_m[bmax],
                "elapsed_s": elapsed_s,
                **res,
            }
        )
    return rows


def compute_bin_contribution_rows(task: dict[str, object]) -> list[dict[str, object]]:
    start = time.perf_counter()
    max_bmax = max(float(value) for value in task["bmax_over_aH_values"])
    res = resolution(vres=int(task["vres"]), rhores=int(task["rhores"]), angle_evals=int(task["angle_evals"]))
    print(
        "[bin worker start] "
        f"condition={task['condition']} v={float(task['velocity_cm_s']):.6e} cm/s "
        f"bmax/aH<= {max_bmax:g} rhores={res['rhores']} "
        f"ures={res['ures']} dphires={res['dphires']} impact_grid={task['impact_grid']}",
        flush=True,
    )

    rhomax_fraction = max_bmax / DEFAULT_CUTOFF_RADIUS_FACTOR
    drag = make_drag(
        int(task["condition"]),
        cutoff_radius_factor=DEFAULT_CUTOFF_RADIUS_FACTOR,
        rhomax_fraction=rhomax_fraction,
        **res,
    )
    velocity_cm_s = float(task["velocity_cm_s"])
    velocity_m_s = velocity_cm_s * CM_PER_S_TO_M_PER_S
    speeds, weights, ds = speed_grid_and_weights(drag, velocity_m_s)
    hydrogen_spacing_m = 1.0 / (DEFAULT_CUTOFF_RADIUS_FACTOR * drag.ustart)
    launch_radius_m = 1.0 / drag.ustart
    max_cutoff_m = max_bmax * hydrogen_spacing_m
    launch_centers_m, launch_widths_m = launch_impact_grid(
        maximum_m=max_cutoff_m,
        count=int(drag.rhores),
        mode=str(task["impact_grid"]),
        minimum_over_maximum=float(task["min_impact_over_max"]),
    )

    bin_force_before_prefactor = None
    bin_centers_m = None
    bin_widths_m = None
    valid_speed_counts = None
    scattering_weight_sum = None

    for speed, weight in zip(speeds, weights):
        if speed <= 0.0 or weight == 0.0:
            continue
        energy = 0.5 * drag.mu * speed**2 + drag.E0Y
        vinf = math.sqrt(energy / (0.5 * drag.mu))
        rhoarr = launch_centers_m * speed / vinf
        keep = launch_centers_m <= max_cutoff_m
        if not np.any(keep):
            continue
        last = np.where(keep)[0][-1] + 1
        rhoarr = rhoarr[:last]
        rhostart = launch_centers_m[:last]
        drhostart = launch_widths_m[:last]
        if len(rhostart) < 1:
            continue

        half_theta = drag.scattering_half_angle(rhoarr, energy)
        if not np.all(np.isfinite(half_theta)):
            raise FloatingPointError("non-finite scattering angle in bin contribution diagnostic")
        scattering_factor = 2.0 * np.square(np.sin(half_theta))
        contribution = rhostart * drhostart * speed**2 * weight * scattering_factor

        if bin_force_before_prefactor is None:
            bin_force_before_prefactor = np.zeros_like(contribution)
            valid_speed_counts = np.zeros_like(contribution, dtype=np.int64)
            scattering_weight_sum = np.zeros_like(contribution)
            bin_centers_m = rhostart
            bin_widths_m = drhostart
        elif len(contribution) != len(bin_force_before_prefactor):
            raise RuntimeError("bin grid length changed across velocity samples")

        bin_force_before_prefactor += contribution
        valid_speed_counts += 1
        scattering_weight_sum += np.abs(weight) * scattering_factor

    if bin_force_before_prefactor is None or bin_centers_m is None or bin_widths_m is None:
        return []

    prefactor = 2.0 * math.pi * drag.nh * drag.mu * ds
    delta_force_n = prefactor * bin_force_before_prefactor
    total_force_n = float(np.sum(delta_force_n))
    elapsed_s = time.perf_counter() - start

    rows = []
    for index, (center_m, width_m, force_n) in enumerate(zip(bin_centers_m, bin_widths_m, delta_force_n)):
        width_over_aH = float(width_m / hydrogen_spacing_m)
        rows.append(
            {
                "condition": int(task["condition"]),
                "condition_label": condition_label(int(task["condition"])),
                "velocity_cm_s": velocity_cm_s,
                "velocity_m_s": velocity_m_s,
                "max_bmax_over_aH_for_shared_grid": max_bmax,
                "impact_grid": task["impact_grid"],
                "min_impact_over_max": task["min_impact_over_max"],
                "rhores": int(res["rhores"]),
                "ures": int(res["ures"]),
                "dphires": int(res["dphires"]),
                "vres": int(res["vres"]),
                "bin_index": int(index),
                "bin_count": int(len(delta_force_n)),
                "rho_launch_m": float(center_m),
                "rho_launch_over_aH": float(center_m / hydrogen_spacing_m),
                "delta_rho_launch_m": float(width_m),
                "delta_rho_launch_over_aH": width_over_aH,
                "delta_force_N": float(force_n),
                "absolute_delta_force_N": float(abs(force_n)),
                "force_density_N_per_aH": float(force_n / width_over_aH) if width_over_aH > 0.0 else math.nan,
                "absolute_force_density_N_per_aH": float(abs(force_n) / width_over_aH) if width_over_aH > 0.0 else math.nan,
                "cumulative_force_N": float(np.sum(delta_force_n[: index + 1])),
                "total_force_N": total_force_n,
                "fraction_of_total_force": float(force_n / total_force_n) if total_force_n != 0.0 else math.nan,
                "valid_speed_count": int(valid_speed_counts[index]),
                "mean_weighted_scattering_factor": float(scattering_weight_sum[index] / valid_speed_counts[index])
                if valid_speed_counts[index] > 0
                else math.nan,
                "hydrogen_interparticle_spacing_m": hydrogen_spacing_m,
                "launch_radius_m": launch_radius_m,
                "elapsed_s": elapsed_s,
            }
        )
    return rows


def build_tasks(args: argparse.Namespace) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    scan_types = [SCAN_RHORES_ONLY]
    if args.include_angle_scan:
        scan_types.append(SCAN_ANGLE_ONLY)
    for scan_type in scan_types:
        for eval_count in args.evals:
            for velocity_cm_s in args.velocities_cm_s:
                if scan_type == SCAN_RHORES_ONLY:
                    rhores = int(eval_count)
                    angle_evals = int(args.fixed_angle_evals)
                else:
                    rhores = int(args.fixed_rhores)
                    angle_evals = int(eval_count)
                tasks.append(
                    {
                        "scan_type": scan_type,
                        "condition": int(args.condition),
                        "velocity_cm_s": float(velocity_cm_s),
                        "bmax_over_aH_values": tuple(float(value) for value in args.bmax_over_aH),
                        "line_evals": int(eval_count),
                        "vres": int(args.vres),
                        "rhores": rhores,
                        "angle_evals": angle_evals,
                        "fixed_rhores": int(args.fixed_rhores),
                        "fixed_angle_evals": int(args.fixed_angle_evals),
                        "impact_grid": str(args.impact_grid),
                        "min_impact_over_max": float(args.min_impact_over_max),
                    }
                )
    return tasks


def build_bin_tasks(args: argparse.Namespace) -> list[dict[str, object]]:
    tasks = []
    for rhores in args.bin_rhores:
        for velocity_cm_s in args.velocities_cm_s:
            tasks.append(
                {
                    "condition": int(args.condition),
                    "velocity_cm_s": float(velocity_cm_s),
                    "bmax_over_aH_values": tuple(float(value) for value in args.bmax_over_aH),
                    "vres": int(args.vres),
                    "rhores": int(rhores),
                    "angle_evals": int(args.fixed_angle_evals),
                    "impact_grid": str(args.impact_grid),
                    "min_impact_over_max": float(args.min_impact_over_max),
                }
            )
    return tasks


def annotate_monotonicity(rows: list[dict[str, object]], tolerance: float) -> None:
    groups: dict[tuple[str, int, float, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["scan_type"]),
            int(row["condition"]),
            float(row["velocity_cm_s"]),
            int(row["line_evals"]),
        )
        groups.setdefault(key, []).append(row)

    for group in groups.values():
        group.sort(key=lambda row: float(row["bmax_over_aH"]))
        previous_drag = math.nan
        previous_bmax = math.nan
        for row in group:
            drag = float(row["absolute_drag_N"])
            if math.isfinite(previous_drag) and previous_drag > 0.0:
                fractional_drop = (previous_drag - drag) / previous_drag
                violates = fractional_drop > tolerance
            else:
                fractional_drop = math.nan
                violates = False
            row["previous_bmax_over_aH"] = previous_bmax
            row["previous_absolute_drag_N"] = previous_drag
            row["fractional_drop_from_previous_bmax"] = fractional_drop
            row["monotonicity_violation"] = violates
            previous_drag = drag
            previous_bmax = float(row["bmax_over_aH"])

    by_point: dict[tuple[str, int, float, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["scan_type"]),
            int(row["condition"]),
            float(row["velocity_cm_s"]),
            float(row["bmax_over_aH"]),
        )
        by_point.setdefault(key, []).append(row)

    for group in by_point.values():
        group.sort(key=lambda row: int(row["line_evals"]))
        reference = float(group[-1]["absolute_drag_N"])
        for row in group:
            value = float(row["absolute_drag_N"])
            row["reference_line_evals"] = int(group[-1]["line_evals"])
            row["relative_error_vs_highest_line_evals"] = (
                abs(value - reference) / abs(reference)
                if math.isfinite(reference) and reference != 0.0
                else math.nan
            )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_scan(rows: list[dict[str, object]], scan_type: str, output: Path) -> None:
    scan_rows = [row for row in rows if row["scan_type"] == scan_type]
    if not scan_rows:
        return

    velocities = sorted({float(row["velocity_cm_s"]) for row in scan_rows})
    line_evals = sorted({int(row["line_evals"]) for row in scan_rows})
    fig, axes = plt.subplots(1, len(velocities), figsize=(5.8 * len(velocities), 4.8), squeeze=False)
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, len(line_evals)))
    title = "Vary rhores only" if scan_type == SCAN_RHORES_ONLY else "Vary ures=dphires only"
    legend_title = "rhores" if scan_type == SCAN_RHORES_ONLY else "angle evals"

    for axis, velocity_cm_s in zip(axes[0], velocities):
        for color, eval_count in zip(colors, line_evals):
            curve = sorted(
                (
                    row
                    for row in scan_rows
                    if float(row["velocity_cm_s"]) == velocity_cm_s and int(row["line_evals"]) == eval_count
                ),
                key=lambda row: float(row["bmax_over_aH"]),
            )
            x = np.array([float(row["bmax_over_aH"]) for row in curve], dtype=float)
            y = np.array([float(row["absolute_drag_N"]) for row in curve], dtype=float)
            axis.plot(x, y, marker="o", linewidth=1.8, markersize=4, color=color, label=f"{eval_count:g}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$b_{max}/a_H$")
        axis.set_title(f"v={velocity_cm_s:.2e} cm/s")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(title=legend_title, fontsize=8)

    axes[0, 0].set_ylabel("|drag| [N]")
    condition = int(scan_rows[0]["condition"])
    fixed_rhores = int(scan_rows[0]["fixed_rhores"])
    fixed_angle = int(scan_rows[0]["fixed_angle_evals"])
    if scan_type == SCAN_RHORES_ONLY:
        subtitle = f"fixed ures=dphires={fixed_angle:g}"
    else:
        subtitle = f"fixed rhores={fixed_rhores:g}"
    grid_label = str(scan_rows[0].get("impact_grid", "unknown"))
    fig.suptitle(f"{title}: Condition {condition}, {condition_label(condition)}, {subtitle}, {grid_label} grid")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_convergence(rows: list[dict[str, object]], scan_type: str, output: Path) -> None:
    scan_rows = [row for row in rows if row["scan_type"] == scan_type]
    if not scan_rows:
        return

    velocities = sorted({float(row["velocity_cm_s"]) for row in scan_rows})
    bmax_values = sorted({float(row["bmax_over_aH"]) for row in scan_rows})
    fig, axes = plt.subplots(1, len(velocities), figsize=(5.8 * len(velocities), 4.8), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bmax_values)))
    x_label = "rhores" if scan_type == SCAN_RHORES_ONLY else "ures=dphires"
    has_positive_error = any(
        math.isfinite(float(row["relative_error_vs_highest_line_evals"]))
        and float(row["relative_error_vs_highest_line_evals"]) > 0.0
        for row in scan_rows
    )

    for axis, velocity_cm_s in zip(axes[0], velocities):
        for color, bmax in zip(colors, bmax_values):
            curve = sorted(
                (
                    row
                    for row in scan_rows
                    if float(row["velocity_cm_s"]) == velocity_cm_s and float(row["bmax_over_aH"]) == bmax
                ),
                key=lambda row: int(row["line_evals"]),
            )
            x = np.array([int(row["line_evals"]) for row in curve], dtype=float)
            y = np.array([float(row["relative_error_vs_highest_line_evals"]) for row in curve], dtype=float)
            finite = np.isfinite(y) & (y > 0.0)
            if np.any(finite):
                axis.plot(x[finite], y[finite], marker="o", linewidth=1.6, markersize=4, color=color, label=f"{bmax:g}")
        axis.set_xscale("log")
        if has_positive_error:
            axis.set_yscale("log")
        else:
            axis.text(
                0.5,
                0.5,
                "No nonzero convergence error",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(x_label)
        axis.set_title(f"v={velocity_cm_s:.2e} cm/s")
        axis.grid(True, which="both", alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(title=r"$b_{max}/a_H$", fontsize=8)

    axes[0, 0].set_ylabel("relative error vs highest resolution")
    title = "rhores convergence" if scan_type == SCAN_RHORES_ONLY else "angle convergence"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_bin_contributions(rows: list[dict[str, object]], bmax_values: tuple[float, ...], output: Path) -> None:
    if not rows:
        return

    rhores_values = sorted({int(row["rhores"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), squeeze=False)
    colors = plt.cm.plasma(np.linspace(0.18, 0.82, len(rhores_values)))

    for color, rhores in zip(colors, rhores_values):
        curve = sorted((row for row in rows if int(row["rhores"]) == rhores), key=lambda row: float(row["rho_launch_over_aH"]))
        x = np.array([float(row["rho_launch_over_aH"]) for row in curve], dtype=float)
        y_bin = np.array([float(row["absolute_delta_force_N"]) for row in curve], dtype=float)
        y_density = np.array([float(row["absolute_force_density_N_per_aH"]) for row in curve], dtype=float)
        bin_valid = np.isfinite(y_bin) & (y_bin > 0.0) & np.isfinite(x) & (x > 0.0)
        density_valid = np.isfinite(y_density) & (y_density > 0.0) & np.isfinite(x) & (x > 0.0)
        marker_size = 32 if rhores <= 150 else 9
        axes[0, 0].scatter(
            x[bin_valid],
            y_bin[bin_valid],
            s=marker_size,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            label=f"rhores={rhores}",
        )
        axes[0, 1].scatter(
            x[density_valid],
            y_density[density_valid],
            s=marker_size,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            label=f"rhores={rhores}",
        )

    for axis in axes[0]:
        for bmax in bmax_values:
            axis.axvline(bmax, color="0.35", linewidth=0.9, alpha=0.35)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"launch impact parameter $\rho/a_H$")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    axes[0, 0].set_ylabel(r"$|\Delta F|$ per bin [N]")
    axes[0, 0].set_title(r"Bin contribution $\Delta F$")
    axes[0, 1].set_ylabel(r"$|dF/d(\rho/a_H)|$ [N]")
    axes[0, 1].set_title("Bin force density")
    condition = int(rows[0]["condition"])
    velocity_cm_s = float(rows[0]["velocity_cm_s"])
    fig.suptitle(
        f"Condition {condition}: {condition_label(condition)}, v={velocity_cm_s:.2e} cm/s, "
        f"fixed dphires={int(rows[0]['dphires'])}, {rows[0].get('impact_grid', 'unknown')} grid"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run_bin_tasks(tasks: list[dict[str, object]], workers: int) -> list[dict[str, object]]:
    total = len(tasks)
    print(f"Launching {total} bin-contribution evaluations on {min(workers, total)} worker processes.", flush=True)
    for index, task in enumerate(tasks, 1):
        print(
            f"[bin queued {index}/{total}] condition={task['condition']} "
            f"v={float(task['velocity_cm_s']):.6e} cm/s rhores={task['rhores']} "
            f"dphires={task['angle_evals']} impact_grid={task['impact_grid']}",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, total)) as executor:
        futures = [executor.submit(compute_bin_contribution_rows, task) for task in tasks]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task_rows = future.result()
            rows.extend(task_rows)
            if task_rows:
                print(
                    f"[bin done {completed}/{total}] rhores={task_rows[0]['rhores']} "
                    f"bins={len(task_rows)} total_force={float(task_rows[0]['total_force_N']):.6e} N "
                    f"elapsed_s={float(task_rows[0]['elapsed_s']):.1f}",
                    flush=True,
                )
    return rows


def summarize(rows: list[dict[str, object]]) -> None:
    print(f"Computed {len(rows)} drag values.", flush=True)
    for scan_type in sorted({str(row["scan_type"]) for row in rows}):
        violations = [row for row in rows if row["scan_type"] == scan_type and row["monotonicity_violation"]]
        print(f"{scan_type}: found {len(violations)} monotonicity violations.", flush=True)
        for row in violations:
            print(
                "violation: "
                f"scan={row['scan_type']} condition={row['condition']} "
                f"v={float(row['velocity_cm_s']):.6e} cm/s "
                f"line_evals={row['line_evals']} "
                f"{float(row['previous_bmax_over_aH']):g}->{float(row['bmax_over_aH']):g} "
                f"drag {float(row['previous_absolute_drag_N']):.6e}->{float(row['absolute_drag_N']):.6e} "
                f"drop={float(row['fractional_drop_from_previous_bmax']):.3e}",
                flush=True,
            )


def run_tasks(tasks: list[dict[str, object]], workers: int) -> list[dict[str, object]]:
    total = len(tasks)
    print(f"Launching {total} drag evaluations on {min(workers, total)} worker processes.", flush=True)
    for index, task in enumerate(tasks, 1):
        res = resolution(vres=int(task["vres"]), rhores=int(task["rhores"]), angle_evals=int(task["angle_evals"]))
        bmax_values = tuple(float(value) for value in task["bmax_over_aH_values"])
        print(
            f"[queued {index}/{total}] scan={task['scan_type']} "
            f"condition={task['condition']} v={float(task['velocity_cm_s']):.6e} cm/s "
            f"bmax/aH={min(bmax_values):g}..{max(bmax_values):g} line_evals={task['line_evals']} "
            f"vres={res['vres']} rhores={res['rhores']} ures={res['ures']} dphires={res['dphires']} "
            f"impact_grid={task['impact_grid']}",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, total)) as executor:
        futures = [executor.submit(compute_cumulative_rows, task) for task in tasks]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task_rows = future.result()
            rows.extend(task_rows)
            row = task_rows[-1]
            print(
                f"[done {completed}/{total}] scan={row['scan_type']} "
                f"bmax/aH={float(task_rows[0]['bmax_over_aH']):g}..{float(row['bmax_over_aH']):g} "
                f"line_evals={row['line_evals']} drag_at_max_bmax={float(row['absolute_drag_N']):.6e} N "
                f"elapsed_s={float(row['elapsed_s']):.1f}",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate high-velocity drag sensitivity to rhores and angle quadrature."
    )
    parser.add_argument("--condition", type=int, default=3)
    parser.add_argument("--velocities-cm-s", type=positive_float_list, default=DEFAULT_VELOCITIES_CM_S)
    parser.add_argument("--bmax-over-aH", type=positive_float_list, default=DEFAULT_BMAX_OVER_AH)
    parser.add_argument("--evals", type=positive_int_list, default=DEFAULT_EVALS)
    parser.add_argument("--bin-rhores", type=positive_int_list, default=DEFAULT_BIN_RHORES)
    parser.add_argument("--vres", type=int, default=25)
    parser.add_argument("--fixed-rhores", type=int, default=10000)
    parser.add_argument("--fixed-angle-evals", type=int, default=10)
    parser.add_argument("--include-angle-scan", action="store_true")
    parser.add_argument("--impact-grid", choices=(IMPACT_GRID_LOG, IMPACT_GRID_EQUAL_AREA), default=IMPACT_GRID_LOG)
    parser.add_argument("--min-impact-over-max", type=float, default=1.0e-6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--output-csv", type=Path, default=OUTDIR / "monotonicity_resolution_scan.csv")
    parser.add_argument("--rhores-plot", type=Path, default=OUTDIR / "condition_3_rhores_only_scan.png")
    parser.add_argument("--angle-plot", type=Path, default=OUTDIR / "condition_3_angle_only_scan.png")
    parser.add_argument("--rhores-convergence-plot", type=Path, default=OUTDIR / "condition_3_rhores_convergence.png")
    parser.add_argument("--angle-convergence-plot", type=Path, default=OUTDIR / "condition_3_angle_convergence.png")
    parser.add_argument("--bin-output-csv", type=Path, default=OUTDIR / "condition_3_bin_force_contributions.csv")
    parser.add_argument("--bin-plot", type=Path, default=OUTDIR / "condition_3_bin_force_contributions.png")
    args = parser.parse_args()

    if max(args.bmax_over_aH) > DEFAULT_CUTOFF_RADIUS_FACTOR:
        parser.error(f"--bmax-over-aH cannot exceed {DEFAULT_CUTOFF_RADIUS_FACTOR:g}")
    if args.vres < 1 or args.fixed_rhores < 2 or args.fixed_angle_evals < 2 or args.workers < 1:
        parser.error("resolution and worker counts must be positive; rhores/angle evals must be at least 2")
    if args.impact_grid == IMPACT_GRID_LOG and not 0.0 < args.min_impact_over_max < 1.0:
        parser.error("--min-impact-over-max must satisfy 0 < value < 1 for log impact grids")

    start = time.perf_counter()
    tasks = build_tasks(args)
    rows = run_tasks(tasks, workers=int(args.workers))
    annotate_monotonicity(rows, tolerance=float(args.tolerance))
    rows.sort(
        key=lambda row: (
            str(row["scan_type"]),
            int(row["condition"]),
            float(row["velocity_cm_s"]),
            int(row["line_evals"]),
            float(row["bmax_over_aH"]),
        )
    )

    write_rows(args.output_csv, rows)
    print(f"Wrote {args.output_csv}", flush=True)
    plot_scan(rows, SCAN_RHORES_ONLY, args.rhores_plot)
    print(f"Wrote {args.rhores_plot}", flush=True)
    plot_convergence(rows, SCAN_RHORES_ONLY, args.rhores_convergence_plot)
    print(f"Wrote {args.rhores_convergence_plot}", flush=True)
    if args.include_angle_scan:
        plot_scan(rows, SCAN_ANGLE_ONLY, args.angle_plot)
        print(f"Wrote {args.angle_plot}", flush=True)
        plot_convergence(rows, SCAN_ANGLE_ONLY, args.angle_convergence_plot)
        print(f"Wrote {args.angle_convergence_plot}", flush=True)
    bin_rows = run_bin_tasks(build_bin_tasks(args), workers=int(args.workers))
    bin_rows.sort(key=lambda row: (int(row["rhores"]), float(row["rho_launch_over_aH"])))
    write_rows(args.bin_output_csv, bin_rows)
    print(f"Wrote {args.bin_output_csv}", flush=True)
    plot_bin_contributions(bin_rows, tuple(float(value) for value in args.bmax_over_aH), args.bin_plot)
    print(f"Wrote {args.bin_plot}", flush=True)
    summarize(rows)
    print(f"Total elapsed_s={time.perf_counter() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
