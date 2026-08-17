# Dungeon Theory Shape Sweep

Direct-run version of `bluehivetheoryshape` for a machine with 20 CPU cores and
up to 2 GPUs. The BlueHive folder is unchanged.

This version uses:

```text
bmax/aH = 0.1, 0.3, 1, 3, 10
velocity = 25 log-spaced points from 1e5 to 1e8 cm/s
conditions = 0, 1, 2, 3
```

The calculation still uses the validated shared log-spaced launch-impact grid
out to `bmax/aH = 10`, then cumulative partial sums for lower bmax values. This
preserves monotonic behavior versus increasing bmax.

The low-level scattering solver in `dragbase2.py` is NumPy/SciPy CPU code. If
CuPy is installed, this folder can use the selected GPUs for the per-speed
`sin(theta)^2` and cumulative-sum reduction after the CPU scattering angles have
been computed. If CuPy is unavailable, it falls back to CPU automatically.

## Run

From this directory:

```bash
bash run_task.sh
```

Equivalent explicit command:

```bash
python3 generate_tasks.py
python3 run_all.py --workers 20 --gpus 2
python3 aggregate_results.py
```

Outputs:

- `dungeon_shape_tasks.csv`
- `task_results/task_*.csv`
- `dungeon_impact_parameter_shape_curves.csv`
- `condition_0_dungeon_impact_parameter_shape.png`
- `condition_1_dungeon_impact_parameter_shape.png`
- `condition_2_dungeon_impact_parameter_shape.png`
- `condition_3_dungeon_impact_parameter_shape.png`
