# Run from the repository root with eight CPU cores:
# python ./theory/dataprocessing/process_unforced.py --workers 8
#
# Reliable reduction of the DAIS production unforced LAMMPS velocity-decay
# campaign. This replaces the stateful fitting cells in datareduction.ipynb.

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
RAW_FILE_RE = re.compile(r"^traj_unforcedprod_v(?P<velocity>[0-9.]+e[+-]\d+)_c(?P<condition>\d+)\.txt$")
RESULT_FIELDS = ("amplitude", "amplitude_sigma", "tau", "tau_sigma", "start_time", "end_time")


@dataclass(frozen=True)
class FitConfig:
    fit_start_peak_fraction: float = 0.999
    fit_end_peak_fraction: float = 0.80
    minimum_tau_s: float = 1.0e-20
    max_optimizer_evaluations: int = 1000
    sigma_floor_fraction: float = 1.0e-6
    max_relative_tau_sigma: float = 1.0
    max_reduced_chi2: float = 100.0
    min_r_squared: float = 0.5
    # Exponential is the simpler default.  The extra flexibility of the
    # power-law fit must reduce RMSE by more than 20% before it is selected.
    minimum_power_rmse_improvement_fraction: float = 0.20


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
    rmse: float = math.nan
    chi2: float = math.nan
    reduced_chi2: float = math.nan
    r_squared: float = math.nan
    residual_lag1_correlation: float = math.nan
    exponential_amplitude: float = math.nan
    exponential_amplitude_sigma: float = math.nan
    exponential_tau: float = math.nan
    exponential_tau_sigma: float = math.nan
    exponential_rmse: float = math.nan
    exponential_chi2: float = math.nan
    exponential_reduced_chi2: float = math.nan
    exponential_r_squared: float = math.nan
    power_B: float = math.nan
    power_B_sigma: float = math.nan
    power_amplitude_at_start: float = math.nan
    power_amplitude_at_start_sigma: float = math.nan
    power_time_offset: float = math.nan
    power_time_offset_sigma: float = math.nan
    power_alpha: float = math.nan
    power_alpha_sigma: float = math.nan
    power_beta: float = math.nan
    power_beta_sigma: float = math.nan
    power_rmse: float = math.nan
    power_chi2: float = math.nan
    power_reduced_chi2: float = math.nan
    power_r_squared: float = math.nan
    best_model: str = ""
    model_selection_reason: str = ""
    initial_mean_velocity: float = math.nan
    observed_decay_fraction: float = math.nan
    start_mean_velocity: float = math.nan
    start_velocity_sem: float = math.nan
    end_mean_velocity: float = math.nan
    end_velocity_sem: float = math.nan
    peak_time: float = math.nan
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
    for path in sorted(raw_dir.glob("traj_unforcedprod_*.txt")):
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
    log_stem = raw_trajectory_path.stem.removeprefix("traj_")
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
        # Unforced decay and its 99.9%-to-80% fit window occur near the start.
        # Quadratic spacing retains far more early-time frames than uniform
        # thinning while still covering the complete trajectory.
        sampling_coordinate = np.linspace(0.0, 1.0, target_frame_count)
        selected_indices = set(
            np.rint(np.square(sampling_coordinate) * (total_snapshots - 1)).astype(int).tolist()
        )
    else:
        selected_indices = set(range(total_snapshots))
    print(
        f"  loading {len(selected_indices)}/{total_snapshots} early-time-weighted raw snapshots "
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


def power_model(time: np.ndarray, amplitude_at_start: float, time_offset: float, alpha: float) -> np.ndarray:
    """Stable form of v = B (t0-t)**(-beta), beta=1/(alpha-1)."""
    beta = 1.0 / (alpha - 1.0)
    return amplitude_at_start * np.power((time_offset - time) / time_offset, -beta)


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
        start_index=0,
        end_index_exclusive=0,
    )
    if total_rows < 3:
        result.quality_flags = "too_few_rows"
        result.message = "fewer than three observations"
        return result

    # Locate the peak over the complete trajectory, then fit its descending branch from
    # the first sample at or below 99.9% of the peak through the first sample at
    # or below 80%.  If 80% is never reached, fit through the final sample.
    peak_index = int(np.argmax(mean))
    peak_velocity = float(mean[peak_index])
    peak_time = float(times[peak_index])
    result.peak_time = peak_time

    if peak_velocity <= 0.0:
        result.status = "ignored"
        result.quality_flags = "nonpositive_peak"
        result.message = "positive exponential requires a positive peak"
        return result

    descending_mean = mean[peak_index:]
    start_crossings = np.flatnonzero(
        descending_mean <= config.fit_start_peak_fraction * peak_velocity
    )
    if start_crossings.size == 0:
        result.status = "ignored"
        result.quality_flags = "never_reaches_99_9_percent_of_peak"
        result.message = "trajectory never decays to 99.9% of its peak"
        return result
    fit_start = peak_index + int(start_crossings[0])
    end_crossings = np.flatnonzero(
        mean[fit_start:] <= config.fit_end_peak_fraction * peak_velocity
    )
    maximum_fit_end = fit_start + int(end_crossings[0]) + 1 if end_crossings.size else total_rows
    result.target_fit_start_time = float(times[fit_start])
    result.start_index = fit_start
    result.end_index_exclusive = maximum_fit_end
    if fit_start >= maximum_fit_end - 2:
        result.status = "ignored"
        result.quality_flags = "too_few_rows_between_99_9_and_80_percent_of_peak"
        result.message = "peak-threshold fit window contains fewer than three observations"
        return result

    result.start_time = float(times[fit_start])
    result.end_time = float(times[maximum_fit_end - 1])
    result.initial_mean_velocity = float(mean[fit_start])
    result.observed_decay_fraction = 1.0 - float(mean[maximum_fit_end - 1] / result.initial_mean_velocity)
    result.start_mean_velocity = float(mean[fit_start])
    result.start_velocity_sem = float(sem[fit_start])
    result.end_mean_velocity = float(mean[maximum_fit_end - 1])
    result.end_velocity_sem = float(sem[maximum_fit_end - 1])

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
        print(
            f"  fitting {source.name}: fixed 99.9%-to-80%-of-peak window "
            f"({maximum_fit_end - fit_start} samples)",
            flush=True,
        )
        fit_time = times[fit_start:maximum_fit_end]
        fit_mean = mean[fit_start:maximum_fit_end]
        fit_sem = np.maximum(sem[fit_start:maximum_fit_end], sigma_floor)
        relative_time = fit_time - fit_time[0]
        duration = float(relative_time[-1])
        positive_end = max(float(fit_mean[-1]), sigma_floor)
        decay_ratio = max(float(fit_mean[0] / positive_end), 1.0 + 1.0e-9)
        tau_guess = max(duration / math.log(decay_ratio), duration, config.minimum_tau_s)

        def exp_residual(parameters: np.ndarray) -> np.ndarray:
            return (exp_model(relative_time, *parameters) - fit_mean) / fit_sem

        exp_fit = least_squares(
            exp_residual,
            x0=np.log([max(float(fit_mean[0]), sigma_floor), tau_guess]),
            bounds=([math.log(np.finfo(float).tiny), math.log(config.minimum_tau_s)], [math.inf, math.inf]),
            loss="linear", x_scale="jac", max_nfev=config.max_optimizer_evaluations,
        )

        # Fit v=A*((q-x)/q)**p with x=t/duration, q>1, p=1/(1-alpha)>0.
        # This is a well-scaled parameterization of v=B*(t0-t)**p, t0>duration.
        def power_residual(parameters: np.ndarray) -> np.ndarray:
            log_A, log_q_margin, log_p = parameters
            q = 1.0 + math.exp(log_q_margin)
            p = math.exp(log_p)
            x = relative_time / duration
            values = math.exp(log_A) * np.power((q - x) / q, p)
            return (values - fit_mean) / fit_sem

        power_fit = least_squares(
            power_residual,
            x0=np.array([math.log(max(float(fit_mean[0]), sigma_floor)), 0.0, 0.0]),
            bounds=(
                [math.log(np.finfo(float).tiny), math.log(1.0e-8), math.log(1.0e-6)],
                [math.inf, math.log(1.0e8), math.log(100.0)],
            ),
            loss="linear", x_scale="jac", max_nfev=config.max_optimizer_evaluations,
        )
        result.candidate_window_count = 1
        result.end_index_exclusive = maximum_fit_end
        result.n_fit_points = len(fit_time)
        result.end_time = float(fit_time[-1])
        log_amplitude_at_start, log_tau = exp_fit.x
        tau = float(math.exp(log_tau))
        amplitude = float(math.exp(log_amplitude_at_start))
        exp_values = exp_model(relative_time, *exp_fit.x)
        log_power_A, log_q_margin, log_p = power_fit.x
        power_A = math.exp(log_power_A)
        q = 1.0 + math.exp(log_q_margin)
        p = math.exp(log_p)
        alpha = 1.0 - 1.0 / p
        beta = -p
        time_offset = q * duration
        log_power_B = log_power_A - p * math.log(time_offset)
        power_B = math.exp(log_power_B) if log_power_B < math.log(np.finfo(float).max) else math.inf
        power_values = power_model(relative_time, power_A, time_offset, alpha)
        exp_raw = fit_mean - exp_values
        power_raw = fit_mean - power_values
        ss_total = float(np.sum(np.square(fit_mean - np.mean(fit_mean))))
        exp_chi2_total = float(np.sum(np.square(exp_residual(exp_fit.x))))
        power_chi2_total = float(np.sum(np.square(power_residual(power_fit.x))))
        exp_chi2 = exp_chi2_total / max(1, len(fit_time) - 2)
        power_chi2 = power_chi2_total / max(1, len(fit_time) - 3)
        exp_rmse = float(np.sqrt(np.mean(np.square(exp_raw))))
        power_rmse = float(np.sqrt(np.mean(np.square(power_raw))))
        exp_r2 = 1.0 - float(np.sum(np.square(exp_raw))) / ss_total if ss_total > 0 else math.nan
        power_r2 = 1.0 - float(np.sum(np.square(power_raw))) / ss_total if ss_total > 0 else math.nan
        exp_covariance = np.linalg.pinv(exp_fit.jac.T @ exp_fit.jac) * exp_chi2
        power_covariance = np.linalg.pinv(power_fit.jac.T @ power_fit.jac) * power_chi2
        log_sigmas = np.sqrt(np.maximum(np.diag(exp_covariance), 0.0))
        tau_sigma = tau * log_sigmas[1]
        amplitude_sigma = amplitude * log_sigmas[0]
        power_sigmas = np.sqrt(np.maximum(np.diag(power_covariance), 0.0))
        p_sigma = p * power_sigmas[2]
        alpha_sigma = p_sigma / (p * p)
        dlog_t0_dlog_margin = (q - 1.0) / q
        time_offset_gradient = np.array([0.0, time_offset * dlog_t0_dlog_margin, 0.0])
        time_offset_sigma = math.sqrt(max(float(time_offset_gradient @ power_covariance @ time_offset_gradient), 0.0))
        log_B_gradient = np.array([1.0, -p * dlog_t0_dlog_margin, -p * math.log(time_offset)])
        power_B_sigma = abs(power_B) * math.sqrt(max(float(log_B_gradient @ power_covariance @ log_B_gradient), 0.0))
        power_A_sigma = power_A * power_sigmas[0]
        power_well_constrained = (
            power_fit.success
            and len(fit_time) >= 5
            and all(np.isfinite(value) for value in (
                power_rmse, alpha, alpha_sigma, time_offset, time_offset_sigma, power_A, power_A_sigma,
            ))
            and alpha < 0.98
            and time_offset_sigma / time_offset < 1.0
        )
        power_improvement = 1.0 - power_rmse / exp_rmse if exp_rmse > 0.0 else -math.inf
        required_power_rmse = exp_rmse * (1.0 - config.minimum_power_rmse_improvement_fraction)
        if power_well_constrained and power_rmse < required_power_rmse:
            best_model = "power"
            selection_reason = (
                f"power RMSE is {power_improvement:.1%} lower and power parameters are constrained"
            )
        else:
            best_model = "exponential"
            if not power_well_constrained:
                selection_reason = "power fit rejected as invalid, boundary-limited, undersampled, or poorly constrained"
            else:
                selection_reason = (
                    f"power RMSE improvement {power_improvement:.1%} is below the "
                    f"{config.minimum_power_rmse_improvement_fraction:.0%} acceptance threshold"
                )
        selected_raw = exp_raw if best_model == "exponential" else power_raw
        selected_rmse = exp_rmse if best_model == "exponential" else power_rmse
        selected_chi2_total = exp_chi2_total if best_model == "exponential" else power_chi2_total
        selected_reduced_chi2 = exp_chi2 if best_model == "exponential" else power_chi2
        selected_r2 = exp_r2 if best_model == "exponential" else power_r2
        selected_fit = exp_fit if best_model == "exponential" else power_fit
        lag1 = (
            float(np.corrcoef(selected_raw[:-1], selected_raw[1:])[0, 1])
            if len(selected_raw) >= 3 and np.std(selected_raw) > 0
            else math.nan
        )

        flags = []
        if not selected_fit.success:
            flags.append(f"{best_model}_optimizer_not_converged")
        if best_model == "exponential" and tau <= config.minimum_tau_s * (1.0 + 1.0e-6):
            flags.append("tau_at_lower_bound")
        if best_model == "exponential" and not np.isfinite(amplitude):
            flags.append("amplitude_overflow")
        if best_model == "exponential" and tau_sigma / tau > config.max_relative_tau_sigma:
            flags.append("tau_poorly_constrained")
        if selected_reduced_chi2 > config.max_reduced_chi2:
            flags.append("large_reduced_chi2")
        if np.isfinite(selected_r2) and selected_r2 < config.min_r_squared:
            flags.append("low_r_squared")
        if np.isfinite(lag1) and abs(lag1) > 0.5:
            flags.append("correlated_residuals")

        result.amplitude = float(amplitude)
        result.amplitude_sigma = float(amplitude_sigma)
        result.tau = float(tau)
        result.tau_sigma = float(tau_sigma)
        result.exponential_amplitude = amplitude
        result.exponential_amplitude_sigma = amplitude_sigma
        result.exponential_tau = tau
        result.exponential_tau_sigma = tau_sigma
        result.exponential_rmse = exp_rmse
        result.exponential_chi2 = exp_chi2_total
        result.exponential_reduced_chi2 = exp_chi2
        result.exponential_r_squared = exp_r2
        result.power_B = float(power_B)
        result.power_B_sigma = float(power_B_sigma)
        result.power_amplitude_at_start = float(power_A)
        result.power_amplitude_at_start_sigma = float(power_A_sigma)
        result.power_time_offset = float(time_offset)
        result.power_time_offset_sigma = float(time_offset_sigma)
        result.power_alpha = alpha
        result.power_alpha_sigma = alpha_sigma
        result.power_beta = beta
        result.power_beta_sigma = p_sigma
        result.power_rmse = power_rmse
        result.power_chi2 = power_chi2_total
        result.power_reduced_chi2 = power_chi2
        result.power_r_squared = power_r2
        result.best_model = best_model
        result.model_selection_reason = selection_reason
        result.rmse = selected_rmse
        result.chi2 = selected_chi2_total
        result.window_selection_score = selected_rmse
        result.reduced_chi2 = selected_reduced_chi2
        result.r_squared = selected_r2
        result.residual_lag1_correlation = lag1
        result.quality_flags = ";".join(sorted(set(flags)))
        result.status = "ok" if not flags else "review"
        result.message = f"exponential: {exp_fit.message}; power: {power_fit.message}"
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

    start = result.start_index
    end = result.end_index_exclusive
    plot_end = min(max(end, start + 1), len(times))
    plot_slice = slice(start, plot_end)
    time_scale = max(float(np.max(np.abs(times[plot_slice]))), 1.0e-300)
    exponent = int(math.floor(math.log10(time_scale)))
    scaled_time = times / (10.0**exponent)
    status_color = {"ok": "#2e7d32", "review": "#ed6c02", "failed": "#c62828"}[result.status]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(scaled_time[plot_slice], mean[plot_slice], color="#1565c0", linewidth=1.2, label="mean projected Si velocity")
    ax.fill_between(
        scaled_time[plot_slice], mean[plot_slice] - sem[plot_slice], mean[plot_slice] + sem[plot_slice],
        color="#1565c0", alpha=0.2, label="mean ± SEM",
    )

    residual_ax.axhline(0.0, color="black", linewidth=0.8)
    if result.status != "failed" and np.isfinite(result.exponential_amplitude) and np.isfinite(result.exponential_tau):
        fit_times = times[start:end]
        elapsed = fit_times - fit_times[0]
        exp_values = result.exponential_amplitude * np.exp(-elapsed / result.exponential_tau)
        power_values = power_model(
            elapsed, result.power_amplitude_at_start, result.power_time_offset, result.power_alpha,
        )
        ax.plot(scaled_time[start:end], exp_values, color="#d32f2f", linewidth=2.0, label="exponential fit")
        ax.plot(scaled_time[start:end], power_values, color="#6a1b9a", linewidth=2.0, label="power fit")
        safe_sem = np.maximum(sem, config.sigma_floor_fraction * max(abs(result.initial_mean_velocity), 1.0))
        selected = exp_values if result.best_model == "exponential" else power_values
        normalized = (mean[start:end] - selected) / safe_sem[start:end]
        residual_ax.plot(scaled_time[start:end], normalized, color="#455a64", marker=".", markersize=3, linewidth=0.8)
        residual_ax.axhline(2.0, color="0.6", linestyle=":", linewidth=0.8)
        residual_ax.axhline(-2.0, color="0.6", linestyle=":", linewidth=0.8)
        metrics = (
            f"exponential: RMSE={result.exponential_rmse:.4g}, "
            f"$\\chi^2$={result.exponential_chi2:.4g}\n"
            f"power law: RMSE={result.power_rmse:.4g}, "
            f"$\\chi^2$={result.power_chi2:.4g}\n"
            f"accepted: {result.best_model}"
        )
        ax.text(
            0.98, 0.04, metrics, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
        )
    else:
        residual_ax.text(0.5, 0.5, "No valid fit", transform=residual_ax.transAxes, ha="center", va="center")

    ax.set_title(
        f"Condition {result.condition}, nominal velocity {result.nominal_velocity_cm_s:.3g} cm/s",
        color=status_color,
    )
    ax.set_ylabel("projected velocity (cm/s)")
    if plot_end > start:
        ax.set_xlim(scaled_time[start], scaled_time[plot_end - 1])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False, borderaxespad=0.0)
    residual_ax.set_xlabel(f"time / 10$^{{{exponent}}}$ s")
    residual_ax.set_ylabel(f"{result.best_model} residual / SEM")
    residual_ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
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
            f"<td>{html.escape(row.best_model or 'none')}</td><td>{row.rmse:.6g}</td>"
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
<table><thead><tr><th>Status</th><th>Condition</th><th>Nominal velocity (cm/s)</th><th>Best model</th><th>RMSE</th><th>Flags</th><th>Diagnostic</th></tr></thead>
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
            start_index=0,
            end_index_exclusive=0,
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
    parser = argparse.ArgumentParser(
        description="Reduce the DAIS production unforced LAMMPS velocity-decay campaign."
    )
    parser.add_argument("--input-dir", type=Path, default=repo_root / "unforced/dataarchive/nprun4_29")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root / "unforced/daisresults/daisproduction",
        help="Directory containing traj_unforcedprod_*.txt dumps and matching LAMMPS logs.",
    )
    parser.add_argument("--source", choices=["raw", "intermediate"], default="raw")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--limit", type=int, help="Process only the first N discovered files (smoke tests).")
    parser.add_argument("--condition", type=int, choices=range(4))
    parser.add_argument("--velocity", type=float, help="Process the discovered nominal velocity nearest this value.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel campaign workers (default: 8).")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-campaign PNG diagnostics and HTML index.")
    parser.add_argument(
        "--max-optimizer-evaluations",
        type=int,
        default=1000,
        help="Maximum least-squares function evaluations for each fixed fit window (default: 1000).",
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
        help="For raw dumps, parse this many early-time-weighted snapshots; 0 parses all.",
    )
    args = parser.parse_args()
    run_start_time = perf_counter()

    if args.max_optimizer_evaluations < 1:
        raise SystemExit("--max-optimizer-evaluations must be at least 1")
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval must be nonnegative")
    if args.raw_target_frames < 0:
        raise SystemExit("--raw-target-frames must be nonnegative")

    config = FitConfig(
        max_optimizer_evaluations=args.max_optimizer_evaluations,
    )
    print("Discovering input campaigns...", flush=True)
    inputs = discover_raw_inputs(args.raw_dir) if args.source == "raw" else discover_inputs(args.input_dir)
    discovered_input_count = len(inputs)
    if args.condition is not None:
        inputs = [item for item in inputs if item[2] == args.condition]
    if args.velocity is not None and inputs:
        nearest = min({item[1] for item in inputs}, key=lambda value: abs(math.log(value / args.velocity)))
        inputs = [item for item in inputs if item[1] == nearest]
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        input_pattern = "traj_unforcedprod_*.txt" if args.source == "raw" else "force_*.np"
        searched_dir = args.raw_dir if args.source == "raw" else args.input_dir
        raise SystemExit(f"No {input_pattern} inputs found in {searched_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    print(
        f"Configuration: source={args.source}, workers={args.workers}, "
        f"target_frames={args.raw_target_frames}, plots={not args.no_plots}, "
        f"hashes={not args.no_hash}",
        flush=True,
    )
    print(f"Selected {len(inputs)} of {discovered_input_count} discovered campaigns", flush=True)

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

    print("Writing fit_results.csv...", flush=True)
    write_csv(args.output_dir / "fit_results.csv", rows)
    if not args.no_plots:
        print("Writing diagnostic index...", flush=True)
        write_diagnostic_index(diagnostics_dir / "index.html", rows, plot_names)
    all_discovered = discover_raw_inputs(args.raw_dir) if args.source == "raw" else discover_inputs(args.input_dir)
    unique_velocities = np.array(sorted({velocity for _, velocity, _ in all_discovered}))
    print("Writing compatible results.npy...", flush=True)
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
    print("Writing manifest.json...", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["status_counts"], indent=2))
    print(f"Completed in {perf_counter() - run_start_time:.1f}s", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
