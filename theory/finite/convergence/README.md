# Convergence scans

This directory contains vectorized-quadrature convergence checks for
`FiniteLaunchDrag` (`theory/finite/finite_launch.py`). It has two drivers:

- `run_resolution_convergence.py` checks the three numerical grids and a
  finite-cutoff scan for condition 0.
- `run_bmax_convergence.py` checks the large-`b_max` limit over all four
  conditions.

Both depend only on `theory/finite/` and `theory/dragbase2.py`.

## `run_resolution_convergence.py`

```powershell
python .\theory\finite\convergence\run_resolution_convergence.py
```

The driver defaults to 24 workers and produces `condition_0_convergence.png`:
velocity-resolution (`vres`), impact-parameter resolution (`rhores`),
scattering-angle resolution (`dphires`), and drag-force shape while varying the
cutoff. The production settings are `vres=100`, `rhores=300`, and
`dphires=300`; each resolution scan varies one grid and uses the largest tested
resolution (default `1000`) as its reference.

The cutoff scan evaluates `bmax/lambda_S = 0.1, 1, 10, 100, 1000, 10000`.
`condition_0_convergence.csv` records each force evaluation, its settings, the
launch-radius scale, and relative error.

## `run_bmax_convergence.py`

```powershell
python .\theory\finite\convergence\run_bmax_convergence.py --workers 8
```

This sweeps `b_max/a_H = r_i/a_H = rhomax_fraction` over the same cutoff list
for conditions `0 1 2 3`. The largest tested cutoff is the practical
large-`b_max` reference. Defaults use 16 log-spaced velocities from `1e5` to
`1e8 cm/s`, with base `vres=101` and `rhores=dphires=360` at `b_max/a_H=0.1`.
The total integration-point count grows by `1.5` per factor of ten in cutoff.

Outputs:

- `bmax_convergence_scan.csv` records each condition, cutoff, velocity, force,
  relative error, and actual scaled grid settings.
- `bmax_convergence_force_vs_velocity.png` shows the four condition panels.
- `bmax_convergence_relative_error_vs_velocity.png` shows error against the
  largest tested cutoff.
