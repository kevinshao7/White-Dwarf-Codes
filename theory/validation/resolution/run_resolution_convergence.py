from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import condition_label, make_drag, quiet_drag, relative_to_reference, velocity_cases, write_csv

OUTDIR = Path(__file__).resolve().parent
BASE_RESOLUTION = {"vres": 50, "rhores": 180, "ures": 180, "dphires": 180}
RESOLUTION_VALUES = {
    "vres": [25, 35, 50, 70, 100],
    "rhores": [80, 120, 180, 240, 320],
    "ures": [80, 120, 180, 240, 320],
    "dphires": [80, 120, 180, 240, 320],
}
PARAMETER_LABELS = {
    "vres": "velocity samples",
    "rhores": "impact-parameter samples",
    "ures": "inverse-radius samples, infinity to start",
    "dphires": "scattering-angle inverse-radius samples",
}


def run_case(task: tuple[int, str, float, str, int]) -> dict[str, float | int | str]:
    condition, velocity_name, velocity_m_s, parameter, value = task
    resolution = dict(BASE_RESOLUTION)
    resolution[parameter] = value
    drag = make_drag(condition, resolution["vres"], resolution["rhores"], resolution["ures"], resolution["dphires"])
    return {
        "condition": condition,
        "velocity": velocity_name,
        "velocity_m_s": velocity_m_s,
        "parameter": parameter,
        "parameter_value": value,
        **resolution,
        "drag_N": quiet_drag(drag, velocity_m_s),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = []
    for condition in args.conditions:
        probe = make_drag(condition)
        for velocity_name, velocity_m_s in velocity_cases(probe).items():
            for parameter, values in RESOLUTION_VALUES.items():
                for value in values:
                    tasks.append((condition, velocity_name, velocity_m_s, parameter, value))

    total = len(tasks)
    print(
        f"Starting {total} resolution convergence simulations for conditions {args.conditions} "
        f"using up to {args.workers} worker processes.",
        flush=True,
    )
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            elapsed = time.perf_counter() - start
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total - completed) / rate if rate > 0 else float("nan")
            print(
                f"[{completed:3d}/{total}] "
                f"condition={row['condition']} velocity={row['velocity']} "
                f"{row['parameter']}={row['parameter_value']} "
                f"drag={row['drag_N']:.6e} N "
                f"elapsed={elapsed/60:.1f} min eta={remaining/60:.1f} min",
                flush=True,
            )

    rows.sort(key=lambda r: (r["condition"], r["parameter"], r["velocity"], r["parameter_value"]))

    for condition in args.conditions:
        for parameter in RESOLUTION_VALUES:
            subset = [row for row in rows if row["condition"] == condition and row["parameter"] == parameter]
            for velocity_name in sorted({row["velocity"] for row in subset}):
                group = [row for row in subset if row["velocity"] == velocity_name]
                rel_errors = relative_to_reference([float(row["drag_N"]) for row in group])
                reference = float(group[-1]["drag_N"])
                for row, rel_error in zip(group, rel_errors):
                    row["reference_parameter_value"] = group[-1]["parameter_value"]
                    row["reference_drag_N"] = reference
                    row["rel_error_vs_highest_resolution"] = rel_error

    write_csv(OUTDIR / "resolution_convergence.csv", rows)

    for parameter in RESOLUTION_VALUES:
        fig, axes = plt.subplots(
            len(args.conditions),
            2,
            figsize=(12, 3.2 * len(args.conditions)),
            squeeze=False,
            sharex="col",
        )
        for row_axes, condition in zip(axes, args.conditions):
            drag_ax, error_ax = row_axes
            subset = [row for row in rows if row["condition"] == condition and row["parameter"] == parameter]
            for velocity_name in sorted({row["velocity"] for row in subset}):
                group = [row for row in subset if row["velocity"] == velocity_name]
                x = [row["parameter_value"] for row in group]
                drag_ax.plot(
                    x,
                    [abs(row["drag_N"]) for row in group],
                    marker="o",
                    label=velocity_name,
                )
                error_ax.plot(
                    x,
                    [row["rel_error_vs_highest_resolution"] for row in group],
                    marker="o",
                    label=velocity_name,
                )
            drag_ax.set_yscale("log")
            drag_ax.set_xlabel(PARAMETER_LABELS[parameter])
            drag_ax.set_ylabel("|drag| [N]")
            drag_ax.set_title(condition_label(condition))
            drag_ax.grid(True, which="both", alpha=0.3)
            drag_ax.legend()
            error_ax.set_yscale("log")
            error_ax.set_xlabel(PARAMETER_LABELS[parameter])
            error_ax.set_ylabel("relative error vs highest resolution")
            error_ax.set_title(f"{condition_label(condition)} convergence")
            error_ax.grid(True, which="both", alpha=0.3)
            error_ax.legend()
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{parameter}_convergence.png", dpi=200)
        plt.close(fig)

    summary_fig, summary_axes = plt.subplots(
        len(args.conditions),
        1,
        figsize=(8, 3.2 * len(args.conditions)),
        squeeze=False,
    )
    for ax, condition in zip(summary_axes[:, 0], args.conditions):
        for parameter in RESOLUTION_VALUES:
            subset = [
                row
                for row in rows
                if row["condition"] == condition
                and row["parameter"] == parameter
                and row["parameter_value"] != row["reference_parameter_value"]
            ]
            by_value = sorted({row["parameter_value"] for row in subset})
            ax.plot(
                by_value,
                [
                    max(
                        row["rel_error_vs_highest_resolution"]
                        for row in subset
                        if row["parameter_value"] == value
                    )
                    for value in by_value
                ],
                marker="o",
                label=PARAMETER_LABELS[parameter],
            )
        ax.set_yscale("log")
        ax.set_xlabel("sample count")
        ax.set_ylabel("max relative error across velocity cases")
        ax.set_title(condition_label(condition))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    summary_fig.tight_layout()
    summary_fig.savefig(OUTDIR / "resolution_convergence.png", dpi=200)
    plt.close(summary_fig)

    elapsed = time.perf_counter() - start
    print(f"Finished {total} simulations in {elapsed/60:.1f} min.", flush=True)

    for parameter in RESOLUTION_VALUES:
        print(f"{PARAMETER_LABELS[parameter]} convergence:", flush=True)
        for condition in args.conditions:
            subset = [row for row in rows if row["condition"] == condition and row["parameter"] == parameter]
            for velocity_name in sorted({row["velocity"] for row in subset}):
                group = [row for row in subset if row["velocity"] == velocity_name]
                worst_non_reference = max(
                    (
                        row["rel_error_vs_highest_resolution"]
                        for row in group
                        if row["parameter_value"] != row["reference_parameter_value"]
                    ),
                    default=0.0,
                )
                print(
                    f"  condition={condition} velocity={velocity_name}: "
                    f"reference {group[-1]['parameter_value']} -> {group[-1]['drag_N']:.6e} N, "
                    f"worst non-reference relative error {worst_non_reference:.3e}",
                    flush=True,
                )


if __name__ == "__main__":
    freeze_support()
    main()
