# 8/14 Monotonicity Checks

This directory checks whether the very-high-velocity decrease of total drag
with increasing `bmax/aH` is caused by either the impact-parameter quadrature
or the scattering-angle quadrature.

The current solver path is still `validation/common.py` and `dragbase2.py`; it
does not use `FiniteLaunchDrag`.  The angle integrals in `dragbase2.py` use
Gauss-Legendre quadrature with an endpoint-clustering transform, and the
diagnostic computes all `bmax/aH` values cumulatively on one shared grid up to
the largest requested cutoff.

The default run uses only condition 3 at `v = 1e8 cm/s` and scans only
`rhores`:

```text
bmax/aH = 0.5, 1, 2, 5
eval counts = 10, 100, 1000, 10000
workers = 8
```

Default scan:

```text
rhores = 10,100,1000,10000 while ures=dphires=10
impact grid = log-spaced in launch impact parameter from 1e-6*bmax_max to bmax_max
```

Run:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe `
  .\theory\validation\8_14_monotonicity\run_monotonicity_checks.py
```

The script prints queued tasks, worker starts, completed task count, `bmax/aH`,
quadrature sizes, drag value, and elapsed seconds for each calculation.

Outputs:

- `monotonicity_resolution_scan.csv`
- `condition_3_rhores_only_scan.png`
- `condition_3_rhores_convergence.png`
- `condition_3_bin_force_contributions.csv`
- `condition_3_bin_force_contributions.png`

The bin-contribution plot compares `rhores=100` and `rhores=1000` on the same
current-model cumulative log grid up to `bmax/aH=5`, with `ures=dphires=10`.
It plots both per-bin `Delta F` and the corresponding force density.

To also rerun the angle scan:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe `
  .\theory\validation\8_14_monotonicity\run_monotonicity_checks.py `
  --include-angle-scan
```

The opt-in angle scan varies `ures=dphires = 10,100,1000,10000` while holding
`rhores=10000`.
