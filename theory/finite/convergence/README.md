# Finite-launch convergence

This directory contains the vectorized-quadrature convergence driver for
`FiniteLaunchDrag`, condition 0.

## Run

```powershell
python .\theory\finite\convergence\run_resolution_convergence.py
```

The driver defaults to 24 workers. It creates one 2-by-2 figure,
`condition_0_convergence.png`, over a log-spaced bulk-velocity grid:

- velocity-resolution (`vres`) convergence;
- impact-parameter-resolution (`rhores`) convergence;
- scattering-angle-resolution (`dphires`) convergence;
- the drag-force shape as the impact-parameter cutoff (`bmax`) changes.

The production/default resolution is `vres=100`, `rhores=300`, and
`dphires=300`. Each resolution scan varies only its named grid and holds the
other two at those defaults. Its reference is the largest value in
`--resolutions` (default: `1000`).

The cutoff scan evaluates `bmax/lambda_S = 0.1, 1, 10, 100, 1000, 10000`.
The fourth plot panel shows only the resulting drag-force shapes; the `1000`
result remains the reference used to compute the relative errors of every
other cutoff value (including `10000`) in the CSV. `bmax` is converted to the
corresponding finite launch radius internally, since `FiniteLaunchDrag` sets
the cutoff equal to that radius.

`condition_0_convergence.csv` records each force evaluation, all actual
resolution settings, the launch-radius scale, and its relative error.
