# BlueHive Theory Shape Sweep

This folder contains a 120-job BlueHive version of the impact-parameter shape
sweep.  Each job requests one CPU and computes all seven `bmax/aH` cutoffs for
one `(condition, velocity)` pair.

The cutoff list is:

```text
0.1, 0.2, 0.5, 1, 2, 5, 10
```

At `bmax/aH = 0.1`, the impact grid uses `rhores = 10`, while the radial and
scattering-angle quadratures use `ures = dphires = 100`.

Resolution scaling:

```text
rhores = 10 * (bmax/0.1)^2
ures = dphires = 100 * (bmax/0.1)
```

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
module load python3/3.11.10
source /home/kshao4/env/bin/activate
python3 generate_tasks.py
python3 generate_slurm_files.py
mkdir -p logs task_results
for job in slurm/shape_*.slurm; do sbatch "$job"; done
```

Or use:

```bash
bash submit_120.sh
```

After all 120 jobs finish:

```bash
python3 aggregate_results.py
```

Outputs:

- `bluehive_impact_parameter_shape_curves.csv`
- `condition_0_bluehive_impact_parameter_shape.png`
- `condition_1_bluehive_impact_parameter_shape.png`
- `condition_2_bluehive_impact_parameter_shape.png`
- `condition_3_bluehive_impact_parameter_shape.png`
