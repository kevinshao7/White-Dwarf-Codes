"""Plot and tabulate finite-launch drag against analytic weak-Yukawa theories.

Run from the repository root, for example:

    python theory/finite/literaturecomparison/compare_weak_yukawa.py --conditions 0 1 2 3

The numerical result is FiniteLaunchDrag.  The two reference curves are
parameter-free analytic weak-coupling models from ``weak_yukawa.py``; fitted
models (e.g. Grabowski/Stanton fits and effective-potential theories) are
intentionally not included.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Each finite-launch solve is independent.  Pin numerical libraries to one
# thread *before* NumPy/SciPy are imported, so 24 worker processes use 24 CPU
# cores rather than each trying to start its own BLAS thread pool.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
THEORY_DIR = HERE.parents[1]
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))

from finite.finite_launch import FiniteLaunchDrag  # noqa: E402
# This script is intentionally runnable by absolute path, without requiring
# ``theory/finite`` to be installed as a package.  Python puts this script's
# directory on sys.path in that case, so import its sibling directly.
from weak_yukawa import (  # noqa: E402
    MODEL_LABELS,
    drag_force,
    generalized_coulomb_logarithm,
    weak_coupling_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--points", type=int, default=24, help="number of log-spaced velocities")
    parser.add_argument("--vmin-cm-s", type=float, default=1.0e4)
    parser.add_argument("--vmax-cm-s", type=float, default=1.0e8)
    parser.add_argument("--vres", type=int, default=30, help="finite-theory velocity quadrature")
    parser.add_argument("--rhores", type=int, default=100, help="finite-theory impact-parameter quadrature")
    parser.add_argument("--dphires", type=int, default=1000, help="finite-theory orbit quadrature")
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="number of independent finite-launch worker processes (default: 24)",
    )
    parser.add_argument("--output", type=Path, default=HERE / "weak_yukawa_comparison")
    return parser.parse_args()


def finite_force_worker(task: tuple[int, float, int, int, int]) -> float:
    """Evaluate one independent finite-launch velocity point in one process."""
    condition, velocity_m_s, vres, rhores, dphires = task
    numerical = FiniteLaunchDrag(
        condition, method="vectorized", vres=vres, rhores=rhores, dphires=dphires
    )
    return abs(float(numerical.drag(velocity_m_s)))


def finite_force_parallel(condition: int, velocities_m_s: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Return finite-launch forces, distributed across up to 24 CPU cores.

    A velocity point has no dependency on any other point, so process-level
    parallelism is exact and leaves the finite solver unchanged.  ``map``
    preserves the input velocity ordering, which keeps the resulting CSV and
    plot deterministic.
    """
    tasks = [
        (condition, float(velocity), args.vres, args.rhores, args.dphires)
        for velocity in velocities_m_s
    ]
    worker_count = min(args.workers, len(tasks))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return np.fromiter(executor.map(finite_force_worker, tasks, chunksize=1), dtype=np.float64, count=len(tasks))


def main() -> None:
    args = parse_args()
    if args.points < 2 or args.vmin_cm_s <= 0.0 or args.vmax_cm_s <= args.vmin_cm_s:
        raise ValueError("require points >= 2 and 0 < vmin-cm-s < vmax-cm-s")
    if any(condition not in range(4) for condition in args.conditions):
        raise ValueError("conditions must be drawn from 0, 1, 2, 3")
    if not 1 <= args.workers <= 24:
        raise ValueError("workers must lie in [1, 24]")

    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    velocities_cm_s = np.logspace(math.log10(args.vmin_cm_s), math.log10(args.vmax_cm_s), args.points)
    velocities_m_s = velocities_cm_s / 100.0
    n_conditions = len(args.conditions)
    n_columns = 2
    n_rows = math.ceil(n_conditions / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12, 4.5 * n_rows), squeeze=False)
    rows: list[dict[str, float | int | str]] = []

    for panel_index, condition in enumerate(args.conditions):
        numerical = FiniteLaunchDrag(
            condition, method="vectorized", vres=args.vres, rhores=args.rhores, dphires=args.dphires
        )
        finite_force = finite_force_parallel(condition, velocities_m_s, args)
        analytic = {name: drag_force(numerical, velocities_m_s, name) for name in MODEL_LABELS}
        ratio = weak_coupling_ratio(numerical, velocities_m_s)
        logs = {name: generalized_coulomb_logarithm(numerical, velocities_m_s, name) for name in MODEL_LABELS}

        force_ax = axes.flat[panel_index]
        force_ax.loglog(
            velocities_cm_s,
            finite_force,
            "o-",
            ms=3,
            lw=1.5,
            color="black",
            label="This work: finite-launch Yukawa",
        )
        for name, label in MODEL_LABELS.items():
            force_ax.loglog(velocities_cm_s, analytic[name], lw=1.7, label=label)
        force_ax.set_ylabel("drag magnitude (N)")
        force_ax.set_title(f"Condition {condition}: T={numerical.T:.0e} K, density={numerical.gcc:.0e} g cm$^{{-3}}$")
        force_ax.grid(True, which="both", alpha=0.25)
        force_ax.legend(fontsize=8)

        for index, velocity in enumerate(velocities_m_s):
            rows.append({
                "condition": condition,
                "velocity_cm_s": velocities_cm_s[index],
                "finite_launch_force_N": finite_force[index],
                "landau_force_N": analytic["landau"][index],
                "born_transport_force_N": analytic["born_transport"][index],
                "finite_over_born": finite_force[index] / analytic["born_transport"][index],
                "b90_over_lambda_s": ratio[index],
                "landau_coulomb_log": logs["landau"][index],
                "born_transport_coulomb_log": logs["born_transport"][index],
            })

    for axis in axes.flat[:n_conditions]:
        axis.set_xlabel("bulk velocity (cm s$^{-1}$)")
    for axis in axes.flat[n_conditions:]:
        axis.remove()
    fig.suptitle("This work: finite-launch Yukawa drag vs analytic literature theories", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure_path = args.output / "comparison.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    csv_path = args.output / "comparison.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {figure_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
