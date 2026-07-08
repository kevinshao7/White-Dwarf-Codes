from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path

# This script parallelizes across drag cases; keep numerical libraries from
# multiplying that by starting their own thread pools in each process.
for thread_env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(thread_env_var, "1")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_args, condition_label, make_drag, quiet_drag, velocity_cases, write_csv

OUTDIR = Path(__file__).resolve().parent
DEFAULT_CONDITIONS = [0, 2]
DEFAULT_CUTOFF_RADIUS_DEBYE_FACTORS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
DEFAULT_VELOCITY_SIGMA_WIDTHS = [2.0, 3.0, 4.0, 5.0, 6.0]
CUTOFF_SCANS = {
    "velocity_cutoff": {
        "label": "velocity half-width [thermal sigma]; default=4",
        "values": DEFAULT_VELOCITY_SIGMA_WIDTHS,
    },
    "cutoff_radius": {
        "label": "cutoff radius / Debye radius",
        "values": DEFAULT_CUTOFF_RADIUS_DEBYE_FACTORS,
    },
}


def yukawa_energy_at_radius(drag, radius_m: float) -> float:
    ustart = 1.0 / radius_m
    return drag.A * math.exp(-drag.k0 / ustart) * ustart


def set_outer_radius(drag, radius_m: float) -> None:
    drag.ustart = 1.0 / radius_m
    drag.E0Y = yukawa_energy_at_radius(drag, radius_m)


def effective_diagnostics(drag, default_radius_m: float, default_e0y_j: float) -> dict[str, float]:
    outer_radius_m = 1.0 / drag.ustart
    return {
        "effective_outer_radius_m": outer_radius_m,
        "effective_outer_radius_factor": outer_radius_m / default_radius_m,
        "effective_outer_radius_over_debye": outer_radius_m / drag.lD,
        "effective_screening_k0R": drag.k0 * outer_radius_m,
        "effective_E0Y_J": drag.E0Y,
        "effective_E0Y_factor": drag.E0Y / default_e0y_j if default_e0y_j != 0.0 else math.nan,
    }


def default_metadata(drag) -> dict[str, float | str]:
    interparticle_spacing_m = 1.0 / drag.ustart
    return {
        "default_outer_radius_definition": "hydrogen interparticle spacing",
        "default_outer_radius_m": interparticle_spacing_m,
        "hydrogen_interparticle_spacing_m": interparticle_spacing_m,
        "electron_debye_radius_m": drag.lD,
        "default_vrel_sigma_width": drag.vrel_sigma_width,
    }


def make_scan_drag(condition: int, scan: str, value: float, vres: int, rhores: int, ures: int, dphires: int):
    drag = make_drag(condition, vres=vres, rhores=rhores, ures=ures, dphires=dphires)
    default_radius_m = 1.0 / drag.ustart
    default_e0y_j = drag.E0Y
    defaults = default_metadata(drag)

    if scan == "velocity_cutoff":
        drag.vrel_sigma_width = value
    elif scan == "cutoff_radius":
        set_outer_radius(drag, value * drag.lD)
    else:
        raise ValueError(f"unknown scan {scan!r}")

    return drag, defaults, effective_diagnostics(drag, default_radius_m, default_e0y_j)


def run_case(task: tuple[int, str, float, str, float, int, int, int, int]) -> dict[str, float | int | str]:
    condition, velocity_name, velocity_m_s, scan, value, vres, rhores, ures, dphires = task
    drag, defaults, diagnostics = make_scan_drag(condition, scan, value, vres, rhores, ures, dphires)
    row = {
        "condition": condition,
        "velocity": velocity_name,
        "velocity_m_s": velocity_m_s,
        "scan": scan,
        "scan_value": value,
        "status": "ok",
        "error": "",
        **defaults,
        **diagnostics,
    }
    try:
        row["drag_N"] = quiet_drag(drag, velocity_m_s)
    except Exception as exc:
        row["drag_N"] = math.nan
        row["status"] = "failed"
        row["error"] = repr(exc)
    return row


def run_tasks(tasks: list[tuple[int, str, float, str, float, int, int, int, int]], workers: int) -> list[dict[str, float | int | str]]:
    rows = []
    completed = 0
    total = len(tasks)
    print(f"Progress: 0/{total} (0.0%)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            print(f"Progress: {completed}/{total} ({100.0 * completed / total:.1f}%)", flush=True)
    return rows


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.set_defaults(conditions=DEFAULT_CONDITIONS)
    args = parser.parse_args()

    tasks = []
    for condition in args.conditions:
        probe = make_drag(condition, args.vres, args.rhores, args.ures, args.dphires)
        for velocity_name, velocity_m_s in velocity_cases(probe).items():
            for scan, spec in CUTOFF_SCANS.items():
                for value in spec["values"]:
                    tasks.append((condition, velocity_name, velocity_m_s, scan, value, args.vres, args.rhores, args.ures, args.dphires))

    if not tasks:
        raise SystemExit("No validation tasks were selected.")

    workers = min(max(1, args.workers), len(tasks), os.cpu_count() or args.workers)

    print(f"Running {len(tasks)} drag evaluations with {workers} worker processes.", flush=True)
    rows = run_tasks(tasks, workers)

    rows.sort(key=lambda r: (r["condition"], r["scan"], r["velocity"], r["scan_value"]))

    write_csv(OUTDIR / "cutoff_radius_convergence.csv", rows)

    plot_defaults = {condition: default_metadata(make_drag(condition)) for condition in args.conditions}
    plot_labels = {condition: condition_label(condition) for condition in args.conditions}

    for scan, spec in CUTOFF_SCANS.items():
        fig, axes = plt.subplots(len(args.conditions), 1, figsize=(7, 3 * len(args.conditions)), squeeze=False)
        for ax, condition in zip(axes[:, 0], args.conditions):
            subset = [row for row in rows if row["condition"] == condition and row["scan"] == scan]
            for velocity_name in sorted({row["velocity"] for row in subset}):
                group = [row for row in subset if row["velocity"] == velocity_name and row["status"] == "ok"]
                ax.plot(
                    [row["scan_value"] for row in group],
                    [abs(row["drag_N"]) for row in group],
                    marker="o",
                    label=velocity_name,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(spec["label"])
            ax.set_ylabel("|drag| [N]")
            defaults = plot_defaults[condition]
            ax.set_title(
                f"{plot_labels[condition]}; default R={defaults['default_outer_radius_m']:.2e} m "
                f"(H interparticle spacing), lD={defaults['electron_debye_radius_m']:.2e} m"
            )
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{scan}_convergence.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(len(args.conditions), 1, figsize=(7, 3 * len(args.conditions)), squeeze=False)
        for ax, condition in zip(axes[:, 0], args.conditions):
            subset = [row for row in rows if row["condition"] == condition and row["scan"] == scan]
            for velocity_name in sorted({row["velocity"] for row in subset}):
                group = [row for row in subset if row["velocity"] == velocity_name and row["status"] == "ok"]
                group.sort(key=lambda row: row["scan_value"])
                if not group:
                    continue
                reference = abs(group[-1]["drag_N"])
                if reference == 0.0:
                    continue
                ax.plot(
                    [row["scan_value"] for row in group],
                    [max(abs(abs(row["drag_N"]) - reference) / reference, 1.0e-16) for row in group],
                    marker="o",
                    label=velocity_name,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(spec["label"])
            ax.set_ylabel("relative change vs largest cutoff")
            defaults = plot_defaults[condition]
            ax.set_title(
                f"{plot_labels[condition]}; default R={defaults['default_outer_radius_m']:.2e} m "
                f"(H interparticle spacing), lD={defaults['electron_debye_radius_m']:.2e} m"
            )
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{scan}_relative_convergence.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    freeze_support()
    main()
