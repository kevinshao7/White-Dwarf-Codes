# BlueHive Theory Shape Sweep

This folder contains a 120-job BlueHive version of the impact-parameter shape
sweep. Each job requests one CPU and computes one `(condition, velocity)` pair.
The impact integral is evaluated once on a shared log-spaced launch-impact grid
out to `bmax/aH = 10`, then partial sums are used for smaller cutoffs.

The cutoff list is:

```text
0.1, 0.2, 0.5, 1, 2, 5, 10
```

The shared launch-impact grid uses:

```text
rhores = 10000
impact_grid = log
min_impact_over_max = 1e-6
max_bmax/aH = 10
ures = dphires = 100
```

This replaces the older independent per-cutoff equal-area resolution scaling:

```text
rhores = 10 * (bmax/0.1)^2
ures = dphires = 100 * (bmax/0.1)
```

Because every cutoff now uses the same bin contributions, the reported lower
`bmax/aH` values are cumulative subsets of the `bmax/aH = 10` calculation.

The job layout is:

```text
4 conditions x 30 log-spaced velocities from 1e4 to 1e8 cm/s = 120 jobs
```

## Python Environment

The compute jobs need:

```bash
module load python3/3.11.10
python3 -m venv /home/kshao4/env
source /home/kshao4/env/bin/activate
python3 -m pip install -r requirements_hpc.txt
```

Equivalent explicit install:

```bash
python3 -m pip install numpy scipy matplotlib
```

`numpy` and `scipy` are required for the calculations.  `matplotlib` is only
required for `aggregate_results.py` to make the final plots.

## BlueHive Run

From this directory:

```bash
bash run_task.sh
```

With no arguments, `run_task.sh` regenerates `bluehive_shape_tasks.csv`, regenerates
the SLURM scripts, and submits all 120 jobs with `sbatch`.

The older submit entrypoint is still available and delegates to `run_task.sh`:

```bash
bash submit_120.sh
```

After all 120 jobs finish:

```bash
python3 aggregate_results.py
```

Individual tasks are also launched through `run_task.sh`. When called with
`--task-id` or with `TASK_ID` set by SLURM, it loads the BlueHive Python module,
activates `/home/kshao4/env`, checks `numpy` and `scipy`, and then calls
`run_task.py`. The generated SLURM files only set `TASK_ID` and call the bash
wrapper.

Outputs:

- `bluehive_impact_parameter_shape_curves.csv`
- `condition_0_bluehive_impact_parameter_shape.png`
- `condition_1_bluehive_impact_parameter_shape.png`
- `condition_2_bluehive_impact_parameter_shape.png`
- `condition_3_bluehive_impact_parameter_shape.png`
