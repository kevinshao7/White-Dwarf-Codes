"""Process the DAIS unforced convergence sweep.

Run from the repository root, for example:

python .\\unforced\\daisprocesscodes\\process_dais_convergence_sweep.py

To regenerate only c1 figures from existing processed CSVs, fitting only the
first 10% velocity drop from the fitted starting speed:

python .\\unforced\\daisprocesscodes\\process_dais_convergence_sweep.py --figures-from-processed

To overplot the c1 velocity-time traces directly from existing processed CSVs:

python .\\unforced\\daisprocesscodes\\process_dais_convergence_sweep.py --velocity-overlays-from-processed

The defaults stream
``unforced/daisresults/daisoconvergencesweepresults/traj_dais_*.txt``,
keep every 100th saved dump snapshot, use 8 worker threads, write reduced
data to ``unforced/daisprocesscodes/output/processed_data``, and write
convergence figures to ``unforced/daisprocesscodes/output/figures``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

TRAJ_RE = re.compile(
    r"^traj_dais_(?P<family>box|cutoff|ratio)_(?P<label>[^_]+)_c(?P<condition>\d+)\.txt$"
)
RESULT_FIELDS = [
    "family",
    "label",
    "condition",
    "parameter_value",
    "n_si",
    "cutoff_lS",
    "n_h",
    "source_file",
    "status",
    "quality_flags",
    "n_snapshots_seen",
    "n_snapshots_kept",
    "n_fit_points",
    "fit_start_index",
    "fit_end_index",
    "fit_start_velocity_cm_s",
    "fit_end_velocity_cm_s",
    "fit_decay_fraction_of_start_velocity",
    "start_time_s",
    "end_time_s",
    "initial_mean_velocity_cm_s",
    "final_mean_velocity_cm_s",
    "final_velocity_sem_cm_s",
    "observed_decay_fraction",
    "amplitude_cm_s",
    "amplitude_sigma_cm_s",
    "tau_s",
    "tau_sigma_s",
    "reduced_chi2",
    "r_squared",
    "wall_time_s",
    "message",
]


@dataclass(frozen=True)
class SweepMeta:
    family: str
    label: str
    condition: int
    parameter_value: float
    n_si: int
    cutoff_lS: float
    n_h: int


@dataclass
class RunResult:
    family: str
    label: str
    condition: int
    parameter_value: float
    n_si: int
    cutoff_lS: float
    n_h: int
    source_file: str
    status: str
    quality_flags: str
    n_snapshots_seen: int
    n_snapshots_kept: int
    n_fit_points: int
    fit_start_index: int = 0
    fit_end_index: int = 0
    fit_start_velocity_cm_s: float = math.nan
    fit_end_velocity_cm_s: float = math.nan
    fit_decay_fraction_of_start_velocity: float = math.nan
    start_time_s: float = math.nan
    end_time_s: float = math.nan
    initial_mean_velocity_cm_s: float = math.nan
    final_mean_velocity_cm_s: float = math.nan
    final_velocity_sem_cm_s: float = math.nan
    observed_decay_fraction: float = math.nan
    amplitude_cm_s: float = math.nan
    amplitude_sigma_cm_s: float = math.nan
    tau_s: float = math.nan
    tau_sigma_s: float = math.nan
    reduced_chi2: float = math.nan
    r_squared: float = math.nan
    wall_time_s: float = math.nan
    message: str = ""


def parse_meta(path: Path) -> SweepMeta | None:
    match = TRAJ_RE.match(path.name)
    if not match:
        return None
    family = match.group("family")
    label = match.group("label")
    condition = int(match.group("condition"))
    if label.startswith("NSi"):
        n_si = int(label.removeprefix("NSi"))
        cutoff = 2.0
        n_h = 100 * n_si if family == "box" else 100_000
        parameter = float(n_si)
    elif label.endswith("lS"):
        cutoff = float(label.removesuffix("lS"))
        n_si = 1_000
        n_h = 100_000
        parameter = cutoff
    else:
        raise ValueError(f"cannot parse sweep label from {path.name}")
    return SweepMeta(family, label, condition, parameter, n_si, cutoff, n_h)


def discover_trajectories(input_dir: Path) -> list[tuple[Path, SweepMeta]]:
    found: list[tuple[Path, SweepMeta]] = []
    for path in sorted(input_dir.glob("traj_dais_*.txt")):
        meta = parse_meta(path)
        if meta is not None:
            found.append((path, meta))
    return sorted(found, key=lambda item: (item[1].family, item[1].condition, item[1].parameter_value))


def timestep_seconds_from_log(input_dir: Path, meta: SweepMeta) -> float:
    name = f"dais_{meta.family}_{meta.label}_c{meta.condition}"
    candidates = [input_dir / f"{name}.log"]
    candidates.extend(sorted(input_dir.glob(f"{name}_*.lammps.log")))
    candidates.extend(sorted(input_dir.glob(f"{name}_*.out")))
    for path in candidates:
        if not path.is_file():
            continue
        dt = parse_timestep_seconds(path)
        if dt is not None:
            return dt
    searched = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"could not determine timestep for {name}; searched {searched}")


def parse_timestep_seconds(path: Path) -> float | None:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.search(r"Time step\s+:\s+([0-9.eE+-]+)", line)
            if match:
                return float(match.group(1))
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
                    in_thermo = False
                continue
            if len(numeric_rows) == 2:
                step_delta = numeric_rows[1][0] - numeric_rows[0][0]
                time_delta = numeric_rows[1][1] - numeric_rows[0][1]
                if step_delta > 0 and time_delta > 0:
                    return time_delta / step_delta
    return None


def stream_projected_velocity(
    path: Path,
    dt_s: float,
    stride: int,
    production_start_step: int,
    angle_radians: float,
    progress_interval: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    sin_angle = math.sin(angle_radians)
    cos_angle = math.cos(angle_radians)
    means: list[float] = []
    sems: list[float] = []
    times: list[float] = []
    snapshot_values = np.empty(0, dtype=np.float64)
    expected_atoms: int | None = None
    snapshots_seen = 0
    snapshots_kept = 0
    production_snapshots_seen = 0

    with path.open(encoding="utf-8", errors="strict") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"unexpected dump marker {marker.strip()!r}")
            timestep = int(float(handle.readline()))
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("missing NUMBER OF ATOMS marker")
            atom_count = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("missing BOX BOUNDS marker")
            for _ in range(3):
                handle.readline()
            header = handle.readline().split()
            if header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("missing ATOMS marker")
            columns = header[2:]
            try:
                vy_index = columns.index("vy")
                vz_index = columns.index("vz")
            except ValueError as exc:
                raise ValueError(f"velocity columns absent in {path.name}: {columns}") from exc
            if expected_atoms is None:
                expected_atoms = atom_count
                snapshot_values = np.empty(atom_count, dtype=np.float64)
            elif atom_count != expected_atoms:
                raise ValueError(f"atom count changed from {expected_atoms} to {atom_count}")

            # LAMMPS writes a dump at the end of the thermostat run before the
            # subsequent velocity kick.  Require a strictly later timestep so
            # the reduced trajectory starts in the kicked production segment.
            in_production = timestep > production_start_step
            keep = in_production and production_snapshots_seen % stride == 0
            if keep:
                for atom_index in range(atom_count):
                    fields = handle.readline().split()
                    snapshot_values[atom_index] = (
                        sin_angle * float(fields[vy_index])
                        + cos_angle * float(fields[vz_index])
                    )
                means.append(float(np.mean(snapshot_values)))
                sems.append(float(np.std(snapshot_values, ddof=1) / math.sqrt(atom_count)))
                times.append((timestep - production_start_step) * dt_s)
                snapshots_kept += 1
            else:
                for _ in range(atom_count):
                    handle.readline()
            if in_production:
                production_snapshots_seen += 1
            snapshots_seen += 1
            if progress_interval > 0 and snapshots_seen % progress_interval == 0:
                print(f"  scanned {snapshots_seen} snapshots in {path.name}", flush=True)

    if expected_atoms is None:
        raise ValueError("dump contains no complete snapshots")
    time_array = np.asarray(times, dtype=np.float64)
    if len(time_array) >= 2 and np.any(np.diff(time_array) <= 0):
        raise ValueError("kept trajectory times are not strictly increasing")
    return (
        np.asarray(means, dtype=np.float64),
        np.asarray(sems, dtype=np.float64),
        time_array,
        expected_atoms,
        snapshots_seen,
        snapshots_kept,
    )


def fit_decay(
    times: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    max_nfev: int,
    fit_skip_rows: int,
    max_decay_fraction: float | None = None,
) -> dict[str, float | int | str]:
    fit_start_index = min(max(fit_skip_rows, 0), len(times))
    fit_end_index = len(times)
    window_flags: list[str] = []
    if max_decay_fraction is not None and fit_start_index < len(times):
        start_mean = float(mean[fit_start_index])
        decay_threshold = start_mean * (1.0 - max_decay_fraction)
        below_threshold = np.flatnonzero(mean[fit_start_index:] <= decay_threshold)
        if len(below_threshold):
            fit_end_index = fit_start_index + int(below_threshold[0]) + 1
        else:
            window_flags.append("requested_decay_not_reached")
    fit_times = times[fit_start_index:fit_end_index]
    fit_mean = mean[fit_start_index:fit_end_index]
    fit_sem = sem[fit_start_index:fit_end_index]
    if len(fit_times) < 4:
        return {"status": "ignored", "quality_flags": "too_few_points", "message": "fewer than four retained snapshots"}
    if fit_mean[0] <= 0:
        return {"status": "ignored", "quality_flags": "nonpositive_initial_mean", "message": "cannot fit positive exponential"}
    sigma_floor = 1.0e-6 * max(abs(float(fit_mean[0])), 1.0)
    safe_sem = np.maximum(fit_sem, sigma_floor)
    shifted_time = fit_times - fit_times[0]
    positive_end = max(float(fit_mean[-1]), sigma_floor)
    decay_ratio = max(float(fit_mean[0] / positive_end), 1.0 + 1.0e-9)
    duration = max(float(shifted_time[-1]), 1.0e-300)
    tau_guess = max(duration / math.log(decay_ratio), duration, 1.0e-20)
    x0 = np.log([max(float(fit_mean[0]), sigma_floor), tau_guess])

    def residual(parameters: np.ndarray) -> np.ndarray:
        model = np.exp(parameters[0] - shifted_time / np.exp(parameters[1]))
        return (model - fit_mean) / safe_sem

    fit = least_squares(
        residual,
        x0=x0,
        bounds=(
            np.array([math.log(np.finfo(float).tiny), math.log(1.0e-20)]),
            np.array([math.inf, math.inf]),
        ),
        loss="linear",
        x_scale="jac",
        max_nfev=max_nfev,
    )
    normalized = residual(fit.x)
    log_a0, log_tau = fit.x
    tau = float(math.exp(log_tau))
    amplitude = float(math.exp(log_a0 + fit_times[0] / tau))
    model = np.exp(log_a0 - shifted_time / tau)
    raw_residual = fit_mean - model
    dof = max(1, len(fit_times) - 2)
    reduced_chi2 = float(np.sum(np.square(normalized)) / dof)
    ss_total = float(np.sum(np.square(fit_mean - np.mean(fit_mean))))
    r_squared = 1.0 - float(np.sum(np.square(raw_residual))) / ss_total if ss_total > 0 else math.nan
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * reduced_chi2
    log_sigmas = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    tau_sigma = float(tau * log_sigmas[1])
    amplitude_sigma = float(amplitude * log_sigmas[0])
    flags: list[str] = window_flags
    if not fit.success:
        flags.append("optimizer_not_converged")
    if tau_sigma / tau > 1.0:
        flags.append("tau_poorly_constrained")
    if np.isfinite(r_squared) and r_squared < 0.5:
        flags.append("low_r_squared")
    return {
        "status": "ok" if not flags else "review",
        "quality_flags": ";".join(flags),
        "message": fit.message,
        "fit_start_index": fit_start_index,
        "fit_end_index": fit_end_index,
        "fit_start_velocity_cm_s": float(fit_mean[0]),
        "fit_end_velocity_cm_s": float(fit_mean[-1]),
        "fit_decay_fraction_of_start_velocity": (
            float((fit_mean[0] - fit_mean[-1]) / fit_mean[0])
            if fit_mean[0] != 0.0
            else math.nan
        ),
        "amplitude_cm_s": amplitude,
        "amplitude_sigma_cm_s": amplitude_sigma,
        "tau_s": tau,
        "tau_sigma_s": tau_sigma,
        "reduced_chi2": reduced_chi2,
        "r_squared": float(r_squared),
    }


def write_time_series(path: Path, times: np.ndarray, mean: np.ndarray, sem: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "mean_projected_velocity_cm_s", "sem_projected_velocity_cm_s"])
        writer.writerows(zip(times, mean, sem, strict=True))


def process_one(task: tuple[Path, SweepMeta, Path, Path, int, int, float, int, int, int]) -> RunResult:
    path, meta, input_dir, processed_dir, stride, production_start_step, angle, progress_interval, max_nfev, fit_skip_rows = task
    started = perf_counter()
    print(f"Starting {path.name}", flush=True)
    base = f"dais_{meta.family}_{meta.label}_c{meta.condition}"
    try:
        dt_s = timestep_seconds_from_log(input_dir, meta)
        mean, sem, times, n_atoms, seen, kept = stream_projected_velocity(
            path, dt_s, stride, production_start_step, angle, progress_interval
        )
        write_time_series(processed_dir / f"{base}_timeseries_stride{stride}.csv", times, mean, sem)
        fit = fit_decay(times, mean, sem, max_nfev, fit_skip_rows)
        fit_start_index = int(fit.get("fit_start_index", min(max(fit_skip_rows, 0), len(times))))
        fit_end_index = int(fit.get("fit_end_index", len(times)))
        result = RunResult(
            **asdict(meta),
            source_file=path.name,
            status=str(fit["status"]),
            quality_flags=str(fit["quality_flags"]),
            n_snapshots_seen=seen,
            n_snapshots_kept=kept,
            n_fit_points=max(0, fit_end_index - fit_start_index) if fit["status"] != "ignored" else 0,
            fit_start_index=fit_start_index,
            fit_end_index=fit_end_index,
            fit_start_velocity_cm_s=float(fit.get("fit_start_velocity_cm_s", math.nan)),
            fit_end_velocity_cm_s=float(fit.get("fit_end_velocity_cm_s", math.nan)),
            fit_decay_fraction_of_start_velocity=float(fit.get("fit_decay_fraction_of_start_velocity", math.nan)),
            start_time_s=float(times[fit_start_index]) if fit_start_index < len(times) else math.nan,
            end_time_s=float(times[fit_end_index - 1]) if fit_end_index > fit_start_index else math.nan,
            initial_mean_velocity_cm_s=float(mean[fit_start_index]) if fit_start_index < len(mean) else math.nan,
            final_mean_velocity_cm_s=float(mean[-1]) if len(mean) else math.nan,
            final_velocity_sem_cm_s=float(sem[-1]) if len(sem) else math.nan,
            observed_decay_fraction=(
                1.0 - float(mean[-1] / mean[fit_start_index])
                if fit_start_index < len(mean) and mean[fit_start_index] != 0
                else math.nan
            ),
            wall_time_s=perf_counter() - started,
            message=str(fit["message"]),
        )
        for key in ("amplitude_cm_s", "amplitude_sigma_cm_s", "tau_s", "tau_sigma_s", "reduced_chi2", "r_squared"):
            setattr(result, key, float(fit.get(key, math.nan)))
        print(f"Finished {path.name}: {result.status}", flush=True)
        return result
    except Exception as exc:
        print(f"Failed {path.name}: {exc!r}", flush=True)
        return RunResult(
            **asdict(meta),
            source_file=path.name,
            status="failed",
            quality_flags="processing_exception",
            n_snapshots_seen=0,
            n_snapshots_kept=0,
            n_fit_points=0,
            wall_time_s=perf_counter() - started,
            message=repr(exc),
        )


def write_summary(path: Path, rows: list[RunResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({field: getattr(row, field) for field in RESULT_FIELDS} for row in rows)


def plot_convergence(figures_dir: Path, rows: list[RunResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok_rows = [row for row in rows if row.status in {"ok", "review"} and np.isfinite(row.tau_s)]
    families = sorted({row.family for row in ok_rows})
    for family in families:
        subset = [row for row in ok_rows if row.family == family]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        for condition in sorted({row.condition for row in subset}):
            condition_rows = sorted(
                [row for row in subset if row.condition == condition],
                key=lambda row: row.parameter_value,
            )
            x = np.array([row.parameter_value for row in condition_rows], dtype=float)
            tau = np.array([row.tau_s for row in condition_rows], dtype=float)
            tau_sigma = np.array([row.tau_sigma_s for row in condition_rows], dtype=float)
            final_velocity = np.array([row.final_mean_velocity_cm_s for row in condition_rows], dtype=float)
            initial_velocity = np.array([row.initial_mean_velocity_cm_s for row in condition_rows], dtype=float)
            axes[0].errorbar(x, tau, yerr=tau_sigma, marker="o", capsize=3, label=f"c{condition}")
            axes[1].plot(x, final_velocity / initial_velocity, marker="o", label=f"c{condition}")
        x_label = {"box": "NSi with NH = 100 * NSi", "ratio": "NSi with NH = 100000", "cutoff": "cutoff / lS"}[family]
        for ax in axes:
            ax.set_xlabel(x_label)
            ax.grid(alpha=0.25)
            if family in {"box", "ratio"}:
                ax.set_xscale("log")
        axes[0].set_ylabel("exponential decay tau (s)")
        axes[0].set_yscale("log")
        axes[1].set_ylabel("final / initial projected Si velocity")
        axes[0].legend(frameon=False)
        axes[1].legend(frameon=False)
        fig.suptitle(f"DAIS {family} convergence")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{family}_convergence.png", dpi=180)
        plt.close(fig)

    for family in families:
        subset = [row for row in ok_rows if row.family == family]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for condition in sorted({row.condition for row in subset}):
            condition_rows = sorted(
                [row for row in subset if row.condition == condition],
                key=lambda row: row.parameter_value,
            )
            x = np.array([row.parameter_value for row in condition_rows], dtype=float)
            tau = np.array([row.tau_s for row in condition_rows], dtype=float)
            reference = tau[-1]
            ax.plot(x, tau / reference, marker="o", label=f"c{condition}")
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xlabel({"box": "NSi", "ratio": "NSi", "cutoff": "cutoff / lS"}[family])
        ax.set_ylabel("tau / finest-setting tau")
        if family in {"box", "ratio"}:
            ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{family}_relative_tau_convergence.png", dpi=180)
        plt.close(fig)


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_time_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array[:, 0], array[:, 1], array[:, 2]


def row_to_result(row: dict[str, str]) -> RunResult:
    return RunResult(
        family=row["family"],
        label=row["label"],
        condition=int(row["condition"]),
        parameter_value=safe_float(row["parameter_value"]),
        n_si=int(safe_float(row["n_si"])),
        cutoff_lS=safe_float(row["cutoff_lS"]),
        n_h=int(safe_float(row["n_h"])),
        source_file=row["source_file"],
        status=row["status"],
        quality_flags=row["quality_flags"],
        n_snapshots_seen=int(safe_float(row["n_snapshots_seen"])),
        n_snapshots_kept=int(safe_float(row["n_snapshots_kept"])),
        n_fit_points=int(safe_float(row["n_fit_points"])),
        fit_start_index=int(safe_float(row.get("fit_start_index", "0"))),
        fit_end_index=int(safe_float(row.get("fit_end_index", row["n_snapshots_kept"]))),
        fit_start_velocity_cm_s=safe_float(row.get("fit_start_velocity_cm_s", "nan")),
        fit_end_velocity_cm_s=safe_float(row.get("fit_end_velocity_cm_s", "nan")),
        fit_decay_fraction_of_start_velocity=safe_float(row.get("fit_decay_fraction_of_start_velocity", "nan")),
        start_time_s=safe_float(row["start_time_s"]),
        end_time_s=safe_float(row["end_time_s"]),
        initial_mean_velocity_cm_s=safe_float(row["initial_mean_velocity_cm_s"]),
        final_mean_velocity_cm_s=safe_float(row["final_mean_velocity_cm_s"]),
        final_velocity_sem_cm_s=safe_float(row["final_velocity_sem_cm_s"]),
        observed_decay_fraction=safe_float(row["observed_decay_fraction"]),
        amplitude_cm_s=safe_float(row["amplitude_cm_s"]),
        amplitude_sigma_cm_s=safe_float(row["amplitude_sigma_cm_s"]),
        tau_s=safe_float(row["tau_s"]),
        tau_sigma_s=safe_float(row["tau_sigma_s"]),
        reduced_chi2=safe_float(row["reduced_chi2"]),
        r_squared=safe_float(row["r_squared"]),
        wall_time_s=safe_float(row["wall_time_s"]),
        message=row["message"],
    )


def plot_run_diagnostic(path: Path, row: dict[str, str], times: np.ndarray, mean: np.ndarray, sem: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau = safe_float(row["tau_s"])
    tau_sigma = safe_float(row["tau_sigma_s"])
    amplitude = safe_float(row["amplitude_cm_s"])
    fitted_decay_fraction = safe_float(row.get("fit_decay_fraction_of_start_velocity", "nan"))
    fit_start_velocity = safe_float(row.get("fit_start_velocity_cm_s", "nan"))
    fit_end_velocity = safe_float(row.get("fit_end_velocity_cm_s", "nan"))
    fit_start_index = int(safe_float(row.get("fit_start_index", "0"))) if np.isfinite(safe_float(row.get("fit_start_index", "0"))) else 0
    fit_start_index = min(max(fit_start_index, 0), len(times))
    fit_end_index = int(safe_float(row.get("fit_end_index", str(len(times))))) if np.isfinite(safe_float(row.get("fit_end_index", str(len(times))))) else len(times)
    fit_end_index = min(max(fit_end_index, fit_start_index), len(times))
    status = row["status"]
    flags = row["quality_flags"] or "none"
    color = {"ok": "#2e7d32", "review": "#ed6c02", "ignored": "#6d6d6d", "failed": "#c62828"}.get(status, "#333333")
    scale = max(float(np.nanmax(np.abs(times))) if len(times) else 1.0, 1.0e-300)
    exponent = int(math.floor(math.log10(scale)))
    scaled_time = times / (10.0**exponent)

    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.plot(scaled_time, mean, color="#1565c0", linewidth=1.2, label="mean projected Si velocity")
    ax.fill_between(scaled_time, mean - sem, mean + sem, color="#1565c0", alpha=0.18, label="mean +/- SEM")
    if fit_start_index > 0 and len(times):
        ax.axvspan(
            scaled_time[0],
            scaled_time[fit_start_index - 1],
            color="0.45",
            alpha=0.14,
            label="excluded initial transient",
        )
    if np.isfinite(fit_start_velocity):
        cutoff_velocity = fit_start_velocity * 0.8
        ax.axhline(
            cutoff_velocity,
            color="#8d6e63",
            linestyle=":",
            linewidth=1.0,
            label="10% drop cutoff",
        )
    residual_ax.axhline(0.0, color="black", linewidth=0.8)

    if status in {"ok", "review"} and np.isfinite(tau) and np.isfinite(amplitude):
        fit_times = times[fit_start_index:fit_end_index]
        fit_mean = mean[fit_start_index:fit_end_index]
        fit_sem = sem[fit_start_index:fit_end_index]
        model = amplitude * np.exp(-fit_times / tau)
        ax.plot(scaled_time[fit_start_index:fit_end_index], model, color="#d32f2f", linewidth=2.0, label="exponential fit")
        if fit_end_index < len(times):
            ax.axvspan(
                scaled_time[fit_end_index],
                scaled_time[-1],
                color="#8d6e63",
                alpha=0.10,
                label="excluded after 10% decay",
            )
        safe_sem = np.maximum(fit_sem, 1.0e-6 * max(abs(float(fit_mean[0])), 1.0))
        residual_ax.plot(
            scaled_time[fit_start_index:fit_end_index],
            (fit_mean - model) / safe_sem,
            color="#455a64",
            marker=".",
            markersize=3,
            linewidth=0.8,
        )
        residual_ax.axhline(2.0, color="0.6", linestyle=":", linewidth=0.8)
        residual_ax.axhline(-2.0, color="0.6", linestyle=":", linewidth=0.8)
    else:
        residual_ax.text(0.5, 0.5, "No fitted exponential", transform=residual_ax.transAxes, ha="center", va="center")

    title = (
        f"{row['family']} {row['label']} c{row['condition']} - {status}; "
        f"tau={tau:.3e} +/- {tau_sigma:.1e} s; "
        f"fit {fit_start_velocity:.3e}->{fit_end_velocity:.3e} cm/s; "
        f"drop/start={fitted_decay_fraction:.3f}; flags={flags}"
    )
    ax.set_title(title, color=color)
    ax.set_ylabel("projected velocity (cm/s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False, borderaxespad=0.0)
    residual_ax.set_xlabel(f"time / 10$^{{{exponent}}}$ s")
    residual_ax.set_ylabel("residual / SEM")
    residual_ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_diagnostic_index(path: Path, rows: list[dict[str, str]], plot_names: dict[str, str]) -> None:
    table_rows = []
    for row in sorted(rows, key=lambda item: (item["family"], int(item["condition"]), safe_float(item["parameter_value"]))):
        name = plot_names.get(row["source_file"])
        if not name:
            continue
        status = html.escape(row["status"])
        flags = html.escape(row["quality_flags"] or "none")
        label = html.escape(f"{row['family']} {row['label']} c{row['condition']}")
        table_rows.append(
            "<tr>"
            f'<td class="{status}">{status}</td>'
            f"<td>{label}</td>"
            f"<td>{html.escape(row['tau_s'])}</td>"
            f"<td>{html.escape(row.get('fit_decay_fraction_of_start_velocity', ''))}</td>"
            f"<td>{flags}</td>"
            f'<td><a href="{html.escape(name)}"><img src="{html.escape(name)}" loading="lazy"></a></td>'
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DAIS per-run fit diagnostics</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.4rem;vertical-align:top}} th{{position:sticky;top:0;background:white}}
img{{width:360px;height:auto}} .ok{{color:#2e7d32}} .review{{color:#ed6c02}} .ignored{{color:#6d6d6d}} .failed{{color:#c62828}}
</style></head><body><h1>DAIS per-run fit diagnostics</h1>
<p>Click any preview for the full-resolution figure.</p>
<table><thead><tr><th>Status</th><th>Run</th><th>Tau (s)</th><th>Fit Drop / Start</th><th>Flags</th><th>Diagnostic</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></body></html>"""
    path.write_text(document, encoding="utf-8")


def plot_run_diagnostics_from_summary(processed_dir: Path, diagnostics_dir: Path) -> list[str]:
    summary_path = processed_dir / "dais_convergence_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary file: {summary_path}")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summary(summary_path)
    plot_names: dict[str, str] = {}
    for row in rows:
        source_stem = Path(row["source_file"]).stem.removeprefix("traj_")
        matches = sorted(processed_dir.glob(f"{source_stem}_timeseries_stride*.csv"))
        if not matches:
            continue
        times, mean, sem = load_time_series(matches[-1])
        plot_name = f"{source_stem}_diagnostic.png"
        plot_run_diagnostic(diagnostics_dir / plot_name, row, times, mean, sem)
        plot_names[row["source_file"]] = plot_name
    write_diagnostic_index(diagnostics_dir / "index.html", rows, plot_names)
    return sorted(plot_names.values())


def refit_c1_from_processed(
    processed_dir: Path,
    output_dir: Path,
    fit_skip_rows: int,
    max_decay_fraction: float,
    max_nfev: int,
) -> None:
    figures_dir = output_dir / "figures"
    diagnostics_dir = output_dir / "diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for generated in figures_dir.glob("*.png"):
        generated.unlink()
    for generated in diagnostics_dir.glob("*.png"):
        generated.unlink()
    index = diagnostics_dir / "index.html"
    if index.exists():
        index.unlink()
    rows: list[RunResult] = []
    diagnostic_rows: list[dict[str, str]] = []
    plot_names: dict[str, str] = {}

    for series_path in sorted(processed_dir.glob("dais_*_c1_timeseries_stride*.csv")):
        source_stem = re.sub(r"_timeseries_stride\d+$", "", series_path.stem)
        meta = parse_meta(Path(f"traj_{source_stem}.txt"))
        if meta is None or meta.condition != 1:
            continue
        times, mean, sem = load_time_series(series_path)
        fit = fit_decay(
            times,
            mean,
            sem,
            max_nfev=max_nfev,
            fit_skip_rows=fit_skip_rows,
            max_decay_fraction=max_decay_fraction,
        )
        fit_start_index = int(fit.get("fit_start_index", min(max(fit_skip_rows, 0), len(times))))
        fit_end_index = int(fit.get("fit_end_index", len(times)))
        row = RunResult(
            **asdict(meta),
            source_file=f"traj_{source_stem}.txt",
            status=str(fit["status"]),
            quality_flags=str(fit["quality_flags"]),
            n_snapshots_seen=len(times),
            n_snapshots_kept=len(times),
            n_fit_points=max(0, fit_end_index - fit_start_index) if fit["status"] != "ignored" else 0,
            fit_start_index=fit_start_index,
            fit_end_index=fit_end_index,
            fit_start_velocity_cm_s=float(fit.get("fit_start_velocity_cm_s", math.nan)),
            fit_end_velocity_cm_s=float(fit.get("fit_end_velocity_cm_s", math.nan)),
            fit_decay_fraction_of_start_velocity=float(fit.get("fit_decay_fraction_of_start_velocity", math.nan)),
            start_time_s=float(times[fit_start_index]) if fit_start_index < len(times) else math.nan,
            end_time_s=float(times[fit_end_index - 1]) if fit_end_index > fit_start_index else math.nan,
            initial_mean_velocity_cm_s=float(mean[fit_start_index]) if fit_start_index < len(mean) else math.nan,
            final_mean_velocity_cm_s=float(mean[fit_end_index - 1]) if fit_end_index > fit_start_index else math.nan,
            final_velocity_sem_cm_s=float(sem[fit_end_index - 1]) if fit_end_index > fit_start_index else math.nan,
            observed_decay_fraction=(
                1.0 - float(mean[fit_end_index - 1] / mean[fit_start_index])
                if fit_end_index > fit_start_index and mean[fit_start_index] != 0
                else math.nan
            ),
            message=str(fit["message"]),
        )
        for key in ("amplitude_cm_s", "amplitude_sigma_cm_s", "tau_s", "tau_sigma_s", "reduced_chi2", "r_squared"):
            setattr(row, key, float(fit.get(key, math.nan)))
        rows.append(row)

        row_dict = {field: str(getattr(row, field)) for field in RESULT_FIELDS}
        diagnostic_rows.append(row_dict)
        plot_name = f"{source_stem}_diagnostic.png"
        plot_run_diagnostic(diagnostics_dir / plot_name, row_dict, times, mean, sem)
        plot_names[row.source_file] = plot_name

    rows.sort(key=lambda row: (row.family, row.parameter_value))
    summary_path = output_dir / "processed_data" / "dais_c1_first10pct_refit_summary.csv"
    write_summary(summary_path, rows)
    plot_convergence(figures_dir, rows)
    write_diagnostic_index(diagnostics_dir / "index.html", diagnostic_rows, plot_names)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "processed_data_time_series",
        "condition": 1,
        "fit_skip_rows": fit_skip_rows,
        "max_decay_fraction_of_start_velocity": max_decay_fraction,
        "note": "This mode replaces output/figures and output/diagnostics with c1-only first-10%-drop plots.",
        "summary_csv": str(summary_path),
        "figures_dir": str(figures_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "processed_count": len(rows),
        "status_counts": {
            status: sum(row.status == status for row in rows)
            for status in ("ok", "review", "ignored", "failed")
        },
    }
    (output_dir / "manifest_c1_first10pct.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote c1 first-10%-decay figures for {len(rows)} processed time series", flush=True)


def plot_c1_velocity_overlays_from_processed(processed_dir: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay_dir = output_dir / "velocity_overlays_c1"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for generated in overlay_dir.glob("*.png"):
        generated.unlink()

    grouped: dict[str, list[tuple[SweepMeta, Path, np.ndarray, np.ndarray, np.ndarray]]] = {
        "box": [],
        "cutoff": [],
        "ratio": [],
    }
    for series_path in sorted(processed_dir.glob("dais_*_c1_timeseries_stride*.csv")):
        source_stem = re.sub(r"_timeseries_stride\d+$", "", series_path.stem)
        meta = parse_meta(Path(f"traj_{source_stem}.txt"))
        if meta is None or meta.condition != 1:
            continue
        times, mean, sem = load_time_series(series_path)
        grouped[meta.family].append((meta, series_path, times, mean, sem))

    written: list[str] = []
    for family, entries in grouped.items():
        if not entries:
            continue
        entries.sort(key=lambda item: item[0].parameter_value)
        time_scale = max(max(float(np.nanmax(item[2])) for item in entries if len(item[2])), 1.0e-300)
        exponent = int(math.floor(math.log10(time_scale)))
        x_label = f"time / 10$^{{{exponent}}}$ s"
        parameter_label = {"box": "NSi", "ratio": "NSi", "cutoff": "cutoff/lS"}[family]

        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for meta, _, times, mean, _ in entries:
            label = f"{parameter_label}={meta.parameter_value:g}"
            ax.plot(times / (10.0**exponent), mean, linewidth=1.4, label=label)
        ax.set_title(f"DAIS {family} c1 velocity-time overlay")
        ax.set_xlabel(x_label)
        ax.set_ylabel("mean projected Si velocity (cm/s)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = overlay_dir / f"{family}_c1_velocity_time_overlay.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path.name)

        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for meta, _, times, mean, _ in entries:
            start = float(mean[0]) if len(mean) else math.nan
            normalized = mean / start if np.isfinite(start) and start != 0.0 else np.full_like(mean, math.nan)
            label = f"{parameter_label}={meta.parameter_value:g}"
            ax.plot(times / (10.0**exponent), normalized, linewidth=1.4, label=label)
        ax.set_title(f"DAIS {family} c1 normalized velocity-time overlay")
        ax.set_xlabel(x_label)
        ax.set_ylabel("mean projected Si velocity / initial value")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = overlay_dir / f"{family}_c1_normalized_velocity_time_overlay.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path.name)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "processed_data_time_series",
        "condition": 1,
        "output_dir": str(overlay_dir),
        "figures": written,
    }
    (output_dir / "manifest_c1_velocity_overlays.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(written)} c1 velocity overlay figures to {overlay_dir}", flush=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "unforced" / "daisresults" / "daisoconvergencesweepresults"
    default_output = Path(__file__).resolve().parent / "output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--stride", type=int, default=100, help="Keep every Nth saved dump snapshot; default: 100.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads; default: 8.")
    parser.add_argument("--production-start-step", type=int, default=2000)
    parser.add_argument(
        "--fit-skip-rows",
        type=int,
        default=10,
        help="Exclude this many retained post-kick rows from exponential fits; default: 10.",
    )
    parser.add_argument("--angle-radians", type=float, default=0.3)
    parser.add_argument("--max-optimizer-evaluations", type=int, default=1000)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="Process only the first N trajectories for a smoke test.")
    parser.add_argument("--family", choices=["box", "cutoff", "ratio"])
    parser.add_argument("--condition", type=int, choices=[0, 1])
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Build per-run fit diagnostic plots from existing processed_data CSVs without reading trajectory dumps.",
    )
    parser.add_argument(
        "--figures-from-processed",
        action="store_true",
        help="Refit c1 processed time-series CSVs and replace output/figures and output/diagnostics.",
    )
    parser.add_argument(
        "--velocity-overlays-from-processed",
        action="store_true",
        help="Write c1 direct velocity-time overlay plots from processed_data CSVs without fitting.",
    )
    parser.add_argument(
        "--max-decay-fraction",
        type=float,
        default=0.10,
        help="For --figures-from-processed, fit until velocity has dropped by this fraction of fit-start velocity; default: 0.10.",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.fit_skip_rows < 0:
        raise SystemExit("--fit-skip-rows must be nonnegative")
    if not 0.0 < args.max_decay_fraction < 1.0:
        raise SystemExit("--max-decay-fraction must be between 0 and 1")
    processed_dir = args.output_dir / "processed_data"
    figures_dir = args.output_dir / "figures"
    diagnostics_dir = args.output_dir / "diagnostics"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.diagnostics_only:
        written = plot_run_diagnostics_from_summary(processed_dir, diagnostics_dir)
        print(f"Wrote {len(written)} per-run diagnostics to {diagnostics_dir}", flush=True)
        return
    if args.figures_from_processed:
        refit_c1_from_processed(
            processed_dir=processed_dir,
            output_dir=args.output_dir,
            fit_skip_rows=args.fit_skip_rows,
            max_decay_fraction=args.max_decay_fraction,
            max_nfev=args.max_optimizer_evaluations,
        )
        return
    if args.velocity_overlays_from_processed:
        plot_c1_velocity_overlays_from_processed(processed_dir, args.output_dir)
        return

    discovered = discover_trajectories(args.input_dir)
    if args.family is not None:
        discovered = [item for item in discovered if item[1].family == args.family]
    if args.condition is not None:
        discovered = [item for item in discovered if item[1].condition == args.condition]
    if args.limit is not None:
        discovered = discovered[: args.limit]
    if not discovered:
        raise SystemExit(f"No traj_dais_*.txt inputs found in {args.input_dir}")

    tasks = [
        (
            path,
            meta,
            args.input_dir,
            processed_dir,
            args.stride,
            args.production_start_step,
            args.angle_radians,
            args.progress_interval,
            args.max_optimizer_evaluations,
            args.fit_skip_rows,
        )
        for path, meta in discovered
    ]
    print(f"Processing {len(tasks)} trajectories with {args.workers} threads, stride={args.stride}", flush=True)
    rows: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{completed}/{len(tasks)}] {row.source_file}: {row.status}", flush=True)

    rows.sort(key=lambda row: (row.family, row.condition, row.parameter_value))
    summary_path = processed_dir / "dais_convergence_summary.csv"
    write_summary(summary_path, rows)
    plot_convergence(figures_dir, rows)
    run_diagnostics = plot_run_diagnostics_from_summary(processed_dir, diagnostics_dir)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "stride": args.stride,
        "workers": args.workers,
        "production_start_step": args.production_start_step,
        "fit_skip_rows": args.fit_skip_rows,
        "angle_radians": args.angle_radians,
        "processed_count": len(rows),
        "status_counts": {
            status: sum(row.status == status for row in rows)
            for status in ("ok", "review", "ignored", "failed")
        },
        "summary_csv": str(summary_path),
        "figures": sorted(path.name for path in figures_dir.glob("*.png")),
        "diagnostics": run_diagnostics,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["status_counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
