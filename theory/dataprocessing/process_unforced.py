"""RUN FROM THE REPOSITORY ROOT WITH 8 CPU CORES:

python .\\theory\\dataprocessing\\process_unforced.py --workers 8

Reliable reduction of the unforced LAMMPS velocity-decay campaign.

This replaces the stateful fitting cells in ``datareduction.ipynb``.  The
input ``force_*.np`` files contain one row per saved time and one column per
Si atom, followed by the physical time in the final column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import freeze_support
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

FILE_RE = re.compile(r"^force_v(?P<velocity>[0-9.]+e[+-]\d+)_c(?P<condition>\d+)\.np$")
RAW_FILE_RE = re.compile(r"^trajvel_v(?P<velocity>[0-9.]+e[+-]\d+)_c(?P<condition>\d+)\.txt$")
RESULT_FIELDS = ("amplitude", "amplitude_sigma", "tau", "tau_sigma", "start_time", "end_time")


@dataclass(frozen=True)
class FitConfig:
    skip_rows: int = 21
    minimum_window_duration_fraction: float = 0.10
    minimum_tau_s: float = 1.0e-20
    window_length_score_power: float = 1.0
    fit_window_stride: int = 10
    max_optimizer_evaluations: int = 1000
    sigma_floor_fraction: float = 1.0e-6
    max_relative_tau_sigma: float = 1.0
    max_reduced_chi2: float = 100.0
    min_r_squared: float = 0.5


@dataclass
class FitResult:
    condition: int
    nominal_velocity_cm_s: float
    source_file: str
    source_sha256: str
    status: str
    quality_flags: str
    n_times_total: int
    n_atoms: int
    n_fit_points: int
    start_index: int
    end_index_exclusive: int
    amplitude: float = math.nan
    amplitude_sigma: float = math.nan
    tau: float = math.nan
    tau_sigma: float = math.nan
    start_time: float = math.nan
    end_time: float = math.nan
    reduced_chi2: float = math.nan
    r_squared: float = math.nan
    residual_lag1_correlation: float = math.nan
    initial_mean_velocity: float = math.nan
    observed_decay_fraction: float = math.nan
    start_mean_velocity: float = math.nan
    start_velocity_sem: float = math.nan
    end_mean_velocity: float = math.nan
    end_velocity_sem: float = math.nan
    peak_time: float = math.nan
    thermalization_end_time: float = math.nan
    target_fit_start_time: float = math.nan
    window_selection_score: float = math.nan
    candidate_window_count: int = 0
    message: str = ""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_inputs(input_dir: Path) -> list[tuple[Path, float, int]]:
    found = []
    for path in sorted(input_dir.glob("force_*.np")):
        match = FILE_RE.match(path.name)
        if match:
            found.append((path, float(match.group("velocity")), int(match.group("condition"))))
    return sorted(found, key=lambda item: (item[2], item[1]))


def discover_raw_inputs(raw_dir: Path) -> list[tuple[Path, float, int]]:
    found = []
    for path in sorted(raw_dir.glob("trajvel_*.txt")):
        match = RAW_FILE_RE.match(path.name)
        if match:
            found.append((path, float(match.group("velocity")), int(match.group("condition"))))
    return sorted(found, key=lambda item: (item[2], item[1]))


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    array = np.loadtxt(path, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"expected a 2-D array with >=2 atoms and a time column; got {array.shape}")
    velocities, times = array[:, :-1], array[:, -1]
    if not np.isfinite(array).all():
        raise ValueError("contains NaN or infinite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("time column is not strictly increasing")
    return velocities, times


def load_trajectory_stats(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    velocities, times = load_trajectory(path)
    n_atoms = velocities.shape[1]
    mean = np.mean(velocities, axis=1)
    sem = np.std(velocities, axis=1, ddof=1) / math.sqrt(n_atoms)
    return mean, sem, times, n_atoms


def timestep_seconds_from_log(path: Path) -> float:
    """Read timestep size from the first two numeric thermo rows."""
    numeric_rows: list[tuple[float, float]] = []
    in_thermo = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "Step" and fields[1] == "Time":
                in_thermo = True
                numeric_rows = []
                continue
            if not in_thermo:
                continue
            try:
                numeric_rows.append((float(fields[0]), float(fields[1])))
            except (IndexError, ValueError):
                if numeric_rows:
                    break
            if len(numeric_rows) == 2:
                step_delta = numeric_rows[1][0] - numeric_rows[0][0]
                time_delta = numeric_rows[1][1] - numeric_rows[0][1]
                if step_delta <= 0 or time_delta <= 0:
                    raise ValueError(f"invalid thermo increments in {path.name}")
                return time_delta / step_delta
    raise ValueError(f"could not obtain two thermo rows from {path.name}")


def resolve_lammps_log_path(raw_trajectory_path: Path, raw_dir: Path) -> Path:
    """Find the matching LAMMPS thermo output for a raw trajectory dump."""
    log_stem = raw_trajectory_path.stem.replace("trajvel_", "unforcedvel_", 1)
    candidates = [
        raw_dir / f"{log_stem}.log",
        raw_dir / f"{log_stem}.lammps.log",
        raw_trajectory_path.with_name(f"{log_stem}.log"),
        raw_trajectory_path.with_name(f"{log_stem}.lammps.log"),
    ]
    candidates.extend(sorted(raw_dir.glob(f"{log_stem}_*.lammps.log")))
    candidates.extend(sorted(raw_dir.glob(f"{log_stem}_*.out")))
    candidates.extend(sorted(raw_trajectory_path.parent.glob(f"{log_stem}_*.lammps.log")))
    candidates.extend(sorted(raw_trajectory_path.parent.glob(f"{log_stem}_*.out")))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    searched = ", ".join(candidate.name for candidate in candidates)
    raise FileNotFoundError(f"no LAMMPS thermo log found for {raw_trajectory_path.name}; searched {searched}")


def count_raw_lammps_snapshots(path: Path, progress_interval: int = 5000) -> int:
    """Count custom-dump snapshots without parsing per-atom floating point fields."""
    count = 0
    with path.open(encoding="utf-8", errors="strict") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"unexpected dump marker {marker.strip()!r}")
            handle.readline()
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("missing NUMBER OF ATOMS marker")
            atom_count = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("missing BOX BOUNDS marker")
            for _ in range(3):
                handle.readline()
            if not handle.readline().startswith("ITEM: ATOMS"):
                raise ValueError("missing ATOMS marker")
            for _ in range(atom_count):
                handle.readline()
            count += 1
            if progress_interval > 0 and count % progress_interval == 0:
                print(f"  counted {count} snapshots in {path.name}", flush=True)
    return count


def load_raw_lammps_trajectory_stats(
    path: Path,
    log_path: Path,
    angle_radians: float = 0.3,
    progress_interval: int = 500,
    target_frame_count: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Parse a LAMMPS custom dump with strict snapshot/atom-count checks."""
    dt = timestep_seconds_from_log(log_path)
    sin_angle = math.sin(angle_radians)
    cos_angle = math.cos(angle_radians)
    means: list[float] = []
    sems: list[float] = []
    timesteps: list[float] = []
    expected_atom_count: int | None = None
    snapshot = np.empty(0, dtype=np.float64)
    total_snapshots = count_raw_lammps_snapshots(path, progress_interval=max(progress_interval, 5000))
    if total_snapshots == 0:
        raise ValueError("dump contains no complete snapshots")
    if target_frame_count > 0 and total_snapshots > target_frame_count:
        selected_indices = set(np.rint(np.linspace(0, total_snapshots - 1, target_frame_count)).astype(int).tolist())
    else:
        selected_indices = set(range(total_snapshots))
    print(
        f"  loading {len(selected_indices)}/{total_snapshots} evenly spaced raw snapshots "
        f"from {path.name} using log {log_path.name}",
        flush=True,
    )
    with path.open(encoding="utf-8", errors="strict") as handle:
        snapshot_index = 0
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"unexpected dump marker {marker.strip()!r}")
            timestep = float(handle.readline())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("missing NUMBER OF ATOMS marker")
            atom_count = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("missing BOX BOUNDS marker")
            for _ in range(3):
                if len(handle.readline().split()) < 2:
                    raise ValueError("incomplete box bounds")
            if expected_atom_count is None:
                expected_atom_count = atom_count
                snapshot = np.empty(atom_count, dtype=np.float64)
            elif atom_count != expected_atom_count:
                raise ValueError(f"atom count changed between snapshots: {expected_atom_count} then {atom_count}")
            header = handle.readline().split()
            if header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("missing ATOMS marker")
            columns = header[2:]
            try:
                vy_index, vz_index = columns.index("vy"), columns.index("vz")
            except ValueError as exc:
                raise ValueError(f"velocity columns absent: {columns}") from exc
            if snapshot_index not in selected_indices:
                for _ in range(atom_count):
                    handle.readline()
                snapshot_index += 1
                continue
            for atom_index in range(atom_count):
                fields = handle.readline().split()
                if len(fields) != len(columns):
                    raise ValueError(f"incomplete atom row {atom_index} at timestep {timestep:g}")
                snapshot[atom_index] = (
                    sin_angle * float(fields[vy_index])
                    + cos_angle * float(fields[vz_index])
                )
            means.append(float(np.mean(snapshot)))
            sems.append(float(np.std(snapshot, ddof=1) / math.sqrt(atom_count)))
            timesteps.append(timestep)
            if progress_interval > 0 and len(means) % progress_interval == 0:
                print(f"  parsed {len(means)}/{len(selected_indices)} selected snapshots from {path.name}", flush=True)
            snapshot_index += 1
    if not means or expected_atom_count is None:
        raise ValueError("dump contains no complete snapshots")
    mean = np.asarray(means, dtype=np.float64)
    sem = np.asarray(sems, dtype=np.float64)
    times = np.asarray(timesteps, dtype=np.float64) * dt
    if np.any(np.diff(times) <= 0):
        raise ValueError("dump timesteps are not strictly increasing")
    print(f"  parsed {len(means)} snapshots x {expected_atom_count} atoms from {path.name}", flush=True)
    return mean, sem, times, expected_atom_count


def intervals_overlap(mean_a: float, sem_a: float, mean_b: float, sem_b: float) -> bool:
    return max(mean_a - sem_a, mean_b - sem_b) <= min(mean_a + sem_a, mean_b + sem_b)


def exp_model(time: np.ndarray, log_amplitude: float, log_tau: float) -> np.ndarray:
    return np.exp(log_amplitude - time / np.exp(log_tau))


def fit_decay(
    mean: np.ndarray,
    sem: np.ndarray,
    times: np.ndarray,
    n_atoms: int,
    condition: int,
    nominal_velocity: float,
    source: Path,
    source_hash: str,
    config: FitConfig,
    progress_interval: int = 500,
) -> FitResult:
    total_rows = len(times)
    result = FitResult(
        condition=condition,
        nominal_velocity_cm_s=nominal_velocity,
        source_file=source.name,
        source_sha256=source_hash,
        status="failed",
        quality_flags="",
        n_times_total=total_rows,
        n_atoms=n_atoms,
        n_fit_points=0,
        start_index=config.skip_rows,
        end_index_exclusive=config.skip_rows,
    )
    if config.skip_rows >= total_rows:
        result.quality_flags = "too_few_post_thermalization_rows"
        result.message = "skip_rows leaves too few observations"
        return result

    # skip_rows is the first retained row, so the thermalization endpoint is
    # the preceding recorded sample (not the first post-thermalization sample).
    thermalization_end_index = max(0, config.skip_rows - 1)
    thermalization_end_time = float(times[thermalization_end_index])
    post_thermalization = mean[config.skip_rows :]
    peak_index = config.skip_rows + int(np.argmax(post_thermalization))
    peak_time = float(times[peak_index])
    target_fit_start_time = thermalization_end_time + 2.0 * (peak_time - thermalization_end_time)
    fit_start = int(np.searchsorted(times, target_fit_start_time, side="left"))
    maximum_fit_end = total_rows
    result.peak_time = peak_time
    result.thermalization_end_time = thermalization_end_time
    result.target_fit_start_time = target_fit_start_time
    result.start_index = fit_start
    result.end_index_exclusive = maximum_fit_end
    if fit_start >= maximum_fit_end - 2:
        result.status = "ignored"
        result.quality_flags = "too_few_fit_rows_after_doubled_peak_delay"
        result.message = "physical-time fit-start rule leaves fewer than three observations"
        return result

    # Cheap sanity checks always use the maximum possible window.
    result.start_time = float(times[fit_start])
    result.end_time = float(times[-1])
    result.initial_mean_velocity = float(mean[fit_start])
    result.observed_decay_fraction = 1.0 - float(mean[-1] / result.initial_mean_velocity)
    result.start_mean_velocity = float(mean[fit_start])
    result.start_velocity_sem = float(sem[fit_start])
    result.end_mean_velocity = float(mean[-1])
    result.end_velocity_sem = float(sem[-1])

    ignore_flags: list[str] = []
    if result.start_velocity_sem >= abs(result.start_mean_velocity):
        ignore_flags.append("start_velocity_uncertainty_at_least_100_percent")
    if result.end_velocity_sem >= abs(result.end_mean_velocity):
        ignore_flags.append("end_velocity_uncertainty_at_least_100_percent")
    if intervals_overlap(
        result.start_mean_velocity,
        result.start_velocity_sem,
        result.end_mean_velocity,
        result.end_velocity_sem,
    ):
        ignore_flags.append("start_end_velocity_intervals_overlap")
    if ignore_flags:
        result.status = "ignored"
        result.quality_flags = ";".join(ignore_flags)
        result.message = "excluded before fitting by velocity-identifiability rules"
        return result

    if result.start_mean_velocity <= 0:
        result.status = "ignored"
        result.quality_flags = "nonpositive_mean_at_fit_start"
        result.message = "positive exponential requires a positive starting mean"
        return result

    sigma_floor = config.sigma_floor_fraction * max(abs(result.initial_mean_velocity), 1.0)
    try:
        full_simulation_duration = float(times[-1] - times[0])
        minimum_window_duration = config.minimum_window_duration_fraction * full_simulation_duration
        earliest_end_time = float(times[fit_start]) + minimum_window_duration
        first_end_exclusive = max(fit_start + 3, int(np.searchsorted(times, earliest_end_time, side="left")) + 1)
        candidate_ends = list(range(first_end_exclusive, maximum_fit_end + 1, config.fit_window_stride))
        if not candidate_ends or candidate_ends[-1] != maximum_fit_end:
            candidate_ends.append(maximum_fit_end)
        best: tuple[float, object, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        initial_parameters: np.ndarray | None = None

        print(
            f"  fitting {source.name}: {len(candidate_ends)} candidate windows "
            f"(stride={config.fit_window_stride})",
            flush=True,
        )
        for candidate_number, candidate_end in enumerate(candidate_ends, start=1):
            candidate_time = times[fit_start:candidate_end]
            candidate_mean = mean[fit_start:candidate_end]
            candidate_sem = np.maximum(sem[fit_start:candidate_end], sigma_floor)
            relative_time = candidate_time - candidate_time[0]
            duration = float(relative_time[-1])
            positive_end = max(float(candidate_mean[-1]), sigma_floor)
            decay_ratio = max(float(candidate_mean[0] / positive_end), 1.0 + 1.0e-9)
            tau_guess = max(duration / math.log(decay_ratio), duration, config.minimum_tau_s)
            if initial_parameters is None:
                initial_parameters = np.log([max(float(candidate_mean[0]), sigma_floor), tau_guess])

            def residual(parameters: np.ndarray) -> np.ndarray:
                model_values = np.exp(parameters[0] - relative_time / np.exp(parameters[1]))
                return (model_values - candidate_mean) / candidate_sem

            fit = least_squares(
                residual,
                x0=initial_parameters,
                bounds=(
                    np.array([math.log(np.finfo(float).tiny), math.log(config.minimum_tau_s)]),
                    np.array([math.inf, math.inf]),
                ),
                loss="linear",
                x_scale="jac",
                max_nfev=config.max_optimizer_evaluations,
            )
            normalized_residuals = residual(fit.x)
            score = float(
                np.sum(np.square(normalized_residuals))
                / np.power(len(candidate_time), config.window_length_score_power)
            )
            result.candidate_window_count += 1
            if best is None or score < best[0]:
                best = (
                    score,
                    fit,
                    candidate_end,
                    candidate_time,
                    candidate_mean,
                    candidate_sem,
                    normalized_residuals,
                )
            initial_parameters = fit.x
            if progress_interval > 0 and candidate_number % progress_interval == 0:
                print(
                    f"  fitted {candidate_number}/{len(candidate_ends)} candidate windows for {source.name}",
                    flush=True,
                )

        if best is None:
            result.status = "ignored"
            result.quality_flags = "no_window_meets_minimum_duration"
            result.message = "no candidate fit window spans 10% of total simulation duration"
            return result

        score, fit, fit_end, fit_time, fit_mean, fit_sem, normalized_residuals = best
        result.end_index_exclusive = fit_end
        result.n_fit_points = len(fit_time)
        result.end_time = float(fit_time[-1])
        result.window_selection_score = score
        log_amplitude_at_start, log_tau = fit.x
        tau = float(math.exp(log_tau))
        log_amplitude = float(log_amplitude_at_start + fit_time[0] / tau)
        amplitude = float(math.exp(log_amplitude)) if log_amplitude < math.log(np.finfo(float).max) else math.inf
        model = np.exp(log_amplitude_at_start - (fit_time - fit_time[0]) / tau)
        raw_residual = fit_mean - model
        dof = max(1, len(fit_time) - 2)
        reduced_chi2 = float(np.sum(np.square(normalized_residuals)) / dof)
        ss_total = float(np.sum(np.square(fit_mean - np.mean(fit_mean))))
        r_squared = 1.0 - float(np.sum(np.square(raw_residual))) / ss_total if ss_total > 0 else math.nan

        covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * reduced_chi2
        log_sigmas = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        tau_sigma = tau * log_sigmas[1]
        amplitude_gradient = np.array([1.0, -fit_time[0] / tau])
        amplitude_log_variance = float(amplitude_gradient @ covariance @ amplitude_gradient)
        amplitude_sigma = amplitude * math.sqrt(max(amplitude_log_variance, 0.0))
        lag1 = (
            float(np.corrcoef(raw_residual[:-1], raw_residual[1:])[0, 1])
            if len(raw_residual) >= 3 and np.std(raw_residual) > 0
            else math.nan
        )

        if not fit.success:
            flags = ["optimizer_not_converged"]
        else:
            flags = []
        if tau <= config.minimum_tau_s * (1.0 + 1.0e-6):
            flags.append("tau_at_lower_bound")
        if not np.isfinite(amplitude):
            flags.append("amplitude_overflow")
        if tau_sigma / tau > config.max_relative_tau_sigma:
            flags.append("tau_poorly_constrained")
        if reduced_chi2 > config.max_reduced_chi2:
            flags.append("large_reduced_chi2")
        if np.isfinite(r_squared) and r_squared < config.min_r_squared:
            flags.append("low_r_squared")
        if np.isfinite(lag1) and abs(lag1) > 0.5:
            flags.append("correlated_residuals")

        result.amplitude = float(amplitude)
        result.amplitude_sigma = float(amplitude_sigma)
        result.tau = float(tau)
        result.tau_sigma = float(tau_sigma)
        result.reduced_chi2 = reduced_chi2
        result.r_squared = r_squared
        result.residual_lag1_correlation = lag1
        result.quality_flags = ";".join(sorted(set(flags)))
        result.status = "ok" if not flags else "review"
        result.message = fit.message
    except Exception as exc:
        result.quality_flags = "fit_exception"
        result.message = repr(exc)
    return result


def write_csv(path: Path, rows: list[FitResult]) -> None:
    fields = list(asdict(rows[0]).keys()) if rows else list(FitResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_compatible_results(path: Path, rows: list[FitResult], velocities: np.ndarray) -> None:
    output = np.full((4, len(velocities), 6), np.nan, dtype=np.float64)
    lookup = {(condition, float(velocity)): index for condition in range(4) for index, velocity in enumerate(velocities)}
    for row in rows:
        index = lookup.get((row.condition, row.nominal_velocity_cm_s))
        if index is not None and row.status in {"ok", "review"}:
            output[row.condition, index] = [getattr(row, field) for field in RESULT_FIELDS]
    np.save(path, output)


def plot_fit_diagnostic(
    path: Path,
    mean: np.ndarray,
    sem: np.ndarray,
    times: np.ndarray,
    result: FitResult,
    config: FitConfig,
) -> None:
    """Write a two-panel, review-oriented diagnostic for one campaign."""
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_scale = max(float(np.max(np.abs(times))), 1.0e-300)
    exponent = int(math.floor(math.log10(time_scale)))
    scaled_time = times / (10.0**exponent)
    start = result.start_index
    end = result.end_index_exclusive
    status_color = {"ok": "#2e7d32", "review": "#ed6c02", "failed": "#c62828"}[result.status]

    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(scaled_time, mean, color="#1565c0", linewidth=1.2, label="mean projected Si velocity")
    ax.fill_between(scaled_time, mean - sem, mean + sem, color="#1565c0", alpha=0.2, label="mean ± SEM")
    if config.skip_rows > 0:
        ax.axvspan(
            scaled_time[0],
            scaled_time[config.skip_rows - 1],
            color="0.5",
            alpha=0.12,
            label="excluded thermalization",
        )
    if start > config.skip_rows:
        ax.axvspan(
            scaled_time[config.skip_rows],
            scaled_time[start - 1],
            color="#00897b",
            alpha=0.08,
            label="post-peak delay before fit",
        )
    if np.isfinite(result.peak_time):
        ax.axvline(
            result.peak_time / (10.0**exponent),
            color="#00897b",
            linestyle="--",
            linewidth=1.0,
            label="post-thermalization peak",
        )
    if np.isfinite(result.target_fit_start_time):
        ax.axvline(
            result.target_fit_start_time / (10.0**exponent),
            color="#7b1fa2",
            linestyle="--",
            linewidth=1.0,
            label="calculated fit start",
        )
    if end > start and end <= len(times):
        ax.axvspan(scaled_time[start], scaled_time[end - 1], color="#7b1fa2", alpha=0.1, label="selected fit window")

    residual_ax.axhline(0.0, color="black", linewidth=0.8)
    if result.status != "failed" and np.isfinite(result.amplitude) and np.isfinite(result.tau):
        fit_times = times[start:end]
        model = result.amplitude * np.exp(-fit_times / result.tau)
        ax.plot(scaled_time[start:end], model, color="#d32f2f", linewidth=2.0, label="robust exponential fit")
        safe_sem = np.maximum(sem, config.sigma_floor_fraction * max(abs(result.initial_mean_velocity), 1.0))
        normalized = (mean[start:end] - model) / safe_sem[start:end]
        residual_ax.plot(scaled_time[start:end], normalized, color="#455a64", marker=".", markersize=3, linewidth=0.8)
        residual_ax.axhline(2.0, color="0.6", linestyle=":", linewidth=0.8)
        residual_ax.axhline(-2.0, color="0.6", linestyle=":", linewidth=0.8)
    else:
        residual_ax.text(0.5, 0.5, "No valid exponential fit", transform=residual_ax.transAxes, ha="center", va="center")

    flags = result.quality_flags or "none"
    summary = (
        f"status={result.status}  A={result.amplitude:.4g} cm/s  τ={result.tau:.4g} s  "
        f"τ uncertainty={result.tau_sigma:.3g} s\n"
        f"R²={result.r_squared:.3g}  reduced χ²={result.reduced_chi2:.3g}  "
        f"window score={result.window_selection_score:.3g}  candidates={result.candidate_window_count}\n"
        f"lag-1 residual correlation={result.residual_lag1_correlation:.3g}\nflags: {flags}"
    )
    ax.set_title(
        f"Condition {result.condition}, nominal velocity {result.nominal_velocity_cm_s:.3g} cm/s\n"
        f"{result.source_file}",
        color=status_color,
    )
    ax.text(
        0.01,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": status_color},
    )
    ax.set_ylabel("projected velocity (cm/s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    residual_ax.set_xlabel(f"time / 10$^{{{exponent}}}$ s")
    residual_ax.set_ylabel("residual / SEM")
    residual_ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_diagnostic_index(path: Path, rows: list[FitResult], plot_names: dict[str, str]) -> None:
    table_rows = []
    for row in sorted(rows, key=lambda item: (item.condition, item.nominal_velocity_cm_s)):
        plot = plot_names.get(row.source_file)
        if not plot:
            continue
        link = f'<a href="{html.escape(plot)}"><img src="{html.escape(plot)}" loading="lazy"></a>'
        table_rows.append(
            "<tr>"
            f'<td class="{row.status}">{html.escape(row.status)}</td>'
            f"<td>{row.condition}</td><td>{row.nominal_velocity_cm_s:.6g}</td>"
            f"<td>{html.escape(row.quality_flags or 'none')}</td><td>{link}</td></tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Unforced campaign fit diagnostics</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.4rem;vertical-align:top}} th{{position:sticky;top:0;background:white}}
img{{width:360px;height:auto}} .ok{{color:#2e7d32}} .review{{color:#ed6c02}} .failed{{color:#c62828}}
</style></head><body><h1>Unforced campaign fit diagnostics</h1>
<p>Click any preview for the full-resolution figure. Review orange and red entries before downstream use.</p>
<table><thead><tr><th>Status</th><th>Condition</th><th>Nominal velocity (cm/s)</th><th>Flags</th><th>Diagnostic</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></body></html>"""
    path.write_text(document, encoding="utf-8")


def clear_generated_diagnostics(path: Path) -> None:
    """Remove only files generated by this script, never arbitrary user files."""
    if not path.exists():
        return
    for generated in path.glob("condition_*_velocity_*.png"):
        generated.unlink()
    index = path / "index.html"
    if index.exists():
        index.unlink()


def process_campaign(
    task: tuple[Path, float, int, str, Path, Path, FitConfig, bool, int, bool, int],
) -> tuple[FitResult, str | None]:
    """Worker entry point: load, validate, fit, hash, and optionally plot one campaign."""
    (
        path,
        velocity,
        condition,
        source_kind,
        raw_dir,
        diagnostics_dir,
        config,
        make_plot,
        progress_interval,
        hash_files,
        raw_target_frames,
    ) = task
    start_time = perf_counter()
    print(f"Starting {path.name}", flush=True)
    try:
        if source_kind == "raw":
            mean, sem, times, n_atoms = load_raw_lammps_trajectory_stats(
                path,
                resolve_lammps_log_path(path, raw_dir),
                progress_interval=progress_interval,
                target_frame_count=raw_target_frames,
            )
        else:
            mean, sem, times, n_atoms = load_trajectory_stats(path)
        if hash_files:
            print(f"  hashing {path.name}", flush=True)
            source_hash = sha256_file(path)
        else:
            source_hash = "not_computed"
        print(f"  fitting decay for {path.name}", flush=True)
        result = fit_decay(mean, sem, times, n_atoms, condition, velocity, path, source_hash, config, progress_interval)
        plot_name = None
        if make_plot and result.status != "ignored":
            print(f"  writing diagnostic plot for {path.name}", flush=True)
            plot_name = f"condition_{condition}_velocity_{velocity:.6e}.png"
            plot_fit_diagnostic(diagnostics_dir / plot_name, mean, sem, times, result, config)
        print(f"Finished {path.name} in {perf_counter() - start_time:.1f}s", flush=True)
        return result, plot_name
    except Exception as exc:
        source_hash = sha256_file(path) if hash_files else "not_computed"
        result = FitResult(
            condition=condition,
            nominal_velocity_cm_s=velocity,
            source_file=path.name,
            source_sha256=source_hash,
            status="failed",
            quality_flags="input_validation_failed",
            n_times_total=0,
            n_atoms=0,
            n_fit_points=0,
            start_index=config.skip_rows,
            end_index_exclusive=config.skip_rows,
            message=repr(exc),
        )
        return result, None


def audit_nominal_raw_dir(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"path": None, "status": "not_checked"}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    trajectory_files = [item for item in files if item.name.startswith("trajvel_")]
    run_logs = [item for item in files if item.name.startswith("unforcedvel_")]
    return {
        "path": str(path.resolve()),
        "status": "complete_enough_to_rebuild" if trajectory_files and run_logs else "incomplete",
        "file_count": len(files),
        "trajectory_file_count": len(trajectory_files),
        "run_log_count": len(run_logs),
        "files": [item.name for item in files],
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=repo_root / "unforced/dataarchive/nprun4_29")
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "unforced/daisresults/dais")
    parser.add_argument("--source", choices=["raw", "intermediate"], default="raw")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--skip-rows", type=int, default=21)
    parser.add_argument("--limit", type=int, help="Process only the first N discovered files (smoke tests).")
    parser.add_argument("--condition", type=int, choices=range(4))
    parser.add_argument("--velocity", type=float, help="Process the discovered nominal velocity nearest this value.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel campaign workers (default: 8).")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-campaign PNG diagnostics and HTML index.")
    parser.add_argument(
        "--fit-window-stride",
        type=int,
        default=10,
        help="Try every Nth candidate fit endpoint (1 reproduces the old exhaustive search; default: 10).",
    )
    parser.add_argument(
        "--max-optimizer-evaluations",
        type=int,
        default=1000,
        help="Maximum least-squares function evaluations per candidate window (default: 1000).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Print raw-snapshot and fit-window progress every N items; 0 disables inner progress.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 hashes of large source files for faster exploratory runs.",
    )
    parser.add_argument(
        "--raw-target-frames",
        type=int,
        default=1000,
        help="For raw dumps, parse this many evenly spaced snapshots instead of every frame; 0 parses all.",
    )
    args = parser.parse_args()

    if args.fit_window_stride < 1:
        raise SystemExit("--fit-window-stride must be at least 1")
    if args.max_optimizer_evaluations < 1:
        raise SystemExit("--max-optimizer-evaluations must be at least 1")
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval must be nonnegative")
    if args.raw_target_frames < 0:
        raise SystemExit("--raw-target-frames must be nonnegative")

    config = FitConfig(
        skip_rows=args.skip_rows,
        fit_window_stride=args.fit_window_stride,
        max_optimizer_evaluations=args.max_optimizer_evaluations,
    )
    inputs = discover_raw_inputs(args.raw_dir) if args.source == "raw" else discover_inputs(args.input_dir)
    if args.condition is not None:
        inputs = [item for item in inputs if item[2] == args.condition]
    if args.velocity is not None and inputs:
        nearest = min({item[1] for item in inputs}, key=lambda value: abs(math.log(value / args.velocity)))
        inputs = [item for item in inputs if item[1] == nearest]
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        input_pattern = "trajvel_*.txt" if args.source == "raw" else "force_*.np"
        searched_dir = args.raw_dir if args.source == "raw" else args.input_dir
        raise SystemExit(f"No {input_pattern} inputs found in {searched_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.output_dir / "diagnostics"
    if not args.no_plots:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        clear_generated_diagnostics(diagnostics_dir)
    rows: list[FitResult] = []
    plot_names: dict[str, str] = {}
    tasks = [
        (
            path,
            velocity,
            condition,
            args.source,
            args.raw_dir,
            diagnostics_dir,
            config,
            not args.no_plots,
            args.progress_interval,
            not args.no_hash,
            args.raw_target_frames,
        )
        for path, velocity, condition in inputs
    ]
    print(f"Processing {len(tasks)} campaigns with {args.workers} workers", flush=True)
    if args.workers == 1:
        for completed, task in enumerate(tasks, start=1):
            result, plot_name = process_campaign(task)
            rows.append(result)
            if plot_name:
                plot_names[result.source_file] = plot_name
            print(
                f"[{completed}/{len(tasks)}] {result.source_file}: {result.status}"
                + (f" ({result.quality_flags})" if result.quality_flags else ""),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_campaign, task): task[0].name for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                result, plot_name = future.result()
                rows.append(result)
                if plot_name:
                    plot_names[result.source_file] = plot_name
                print(
                    f"[{completed}/{len(tasks)}] {result.source_file}: {result.status}"
                    + (f" ({result.quality_flags})" if result.quality_flags else ""),
                    flush=True,
                )
    rows.sort(key=lambda item: (item.condition, item.nominal_velocity_cm_s))

    write_csv(args.output_dir / "fit_results.csv", rows)
    if not args.no_plots:
        write_diagnostic_index(diagnostics_dir / "index.html", rows, plot_names)
    all_discovered = discover_raw_inputs(args.raw_dir) if args.source == "raw" else discover_inputs(args.input_dir)
    unique_velocities = np.array(sorted({velocity for _, velocity, _ in all_discovered}))
    write_compatible_results(args.output_dir / "results.npy", rows, unique_velocities)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "input_dir": str(args.input_dir.resolve()),
        "source": args.source,
        "output_dir": str(args.output_dir.resolve()),
        "configuration": asdict(config),
        "discovered_inputs": len(all_discovered),
        "processed_inputs": len(rows),
        "workers": args.workers,
        "source_hashes": "computed" if not args.no_hash else "not_computed",
        "raw_target_frames": args.raw_target_frames,
        "status_counts": {
            status: sum(row.status == status for row in rows)
            for status in ("ok", "review", "ignored", "failed")
        },
        "raw_source_audit": audit_nominal_raw_dir(args.raw_dir),
        "compatible_results_note": "Missing/failed entries are NaN, not the legacy -1 sentinel.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["status_counts"], indent=2))


if __name__ == "__main__":
    freeze_support()
    main()
