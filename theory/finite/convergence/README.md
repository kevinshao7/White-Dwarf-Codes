# Resolution convergence

Demonstrates that the two nested integrals inside
`FiniteLaunchDrag.impact_parameter_integral` (impact parameter `b`, scattering
angle `t`) converge as their respective grid resolutions increase, and
benchmarks which of the four `method=` quadrature schemes
(`theory/finite/finite_launch.py`) is fastest at the resolution needed to hit
a target accuracy.

Depends only on `theory/finite/` and `theory/dragbase2.py` -- nothing under
`theory/validation/`.

## What it measures

Two of the four schemes use adaptive `scipy.integrate.quad` on one integral
and a fixed grid on the other, which isolates that grid's error with no
contamination from the other integral:

- `quad_angle` (angle: quad, impact: log-spaced midpoint) -- sweeping
  `rhores` here measures the impact-parameter grid's error alone.
- `quad_impact` (angle: midpoint, impact: quad) -- sweeping `dphires` here
  measures the scattering-angle grid's error alone.

Both are compared against `quad_quad` (adaptive quadrature on both, tight
tolerance) as the reference value. `vectorized` (midpoint on both, the
fastest scheme) is swept at `rhores = dphires = n` to show the combined error
of the scheme that actually matters for production use.

The script then finds the smallest tested `n` for which `vectorized` stays
under `--target-relative-error` for every tested `(condition, speed)` pair,
times a full `drag()` call for every method at that `n`, and prints/saves
which method is fastest. Treat that printed conclusion as the one to trust --
it is computed from the timing table each run, not asserted from memory.

## Run

```powershell
python .\theory\finite\convergence\run_resolution_convergence.py --workers 8
```

Runtime scales as `len(conditions) * len(speeds) * 3 scan-types *
len(resolutions)` quad-based evaluations, plus
`len(METHODS) * len(conditions)` timed `drag()` calls at the recommended
resolution. With the defaults (4 conditions, 3 speeds, 7 resolutions) this is
252 scan evaluations; expect it to take several minutes on 8 cores, dominated
by the `quad_quad` and `quad_impact` evaluations (nested adaptive quadrature
is the slow path -- see `timing_benchmark.csv` for exact numbers each run).

The script runs in two phases -- reference values first, then the scan --
each printed on completion of every task and, per `--heartbeat-seconds`
(default 12s), on a wall-clock timer even if nothing has finished yet, so a
single slow quad evaluation cannot look like a hang. The timing benchmark
prints one line per `(method, condition)` cell as it completes, not one
summary line after all conditions for that method finish. The
`(condition, speed)` reference value used to judge every resolution and scan
type is computed once per pair up front rather than inside every scan task
(it used to be recomputed 21x per pair -- 3 scan types x 7 resolutions --
since it doesn't depend on either).

**Two tolerances, not one.** `--reference-quad-epsrel` (default 1e-11) is
used only for that once-per-pair ground-truth value. The swept
`quad_angle`/`quad_impact` scan tasks use a separate `--scan-quad-epsrel`
(default 1e-8): a scan task issues up to `n` nested adaptive-quadrature calls,
so forcing all of them to 1e-11 when the discretization error being measured
is O(1e-3) made the largest-`n` cells (1440, 2880) pathologically slow for no
accuracy benefit -- a real bug found the first time this was run at full
scale (throughput collapsed well before task 60/252). Fixed by decoupling the
two tolerances; a full `n=2880` scan batch (21 tasks) now completes in a few
seconds rather than not finishing at all.

The scan also runs in batches of one `(condition, speed)` pair at a time (3
scan types x `len(resolutions)` tasks per batch, printed as
`pair k/N: condition=... speed=...`) rather than one flat submission of every
task. Pass `--max-scan-seconds` to stop starting new pairs once that much
wall-clock time has elapsed in the scan phase -- the in-progress pair still
finishes, so the CSV/plots only ever contain whole, complete pairs, never an
arbitrary partial mix of scan types and resolutions. Skipped pairs are
reported explicitly, and the accuracy recommendation is judged only against
the pairs that finished.

## Outputs

- `resolution_convergence.csv` -- one row per `(condition, speed, scan_type,
  n)`: the integral value, the `quad_quad` reference, and the relative error.
- `condition_<n>_resolution_convergence.png` -- log-log relative error vs.
  resolution, one panel per tested speed, with an `n^-2` reference line (the
  midpoint rule's expected order).
- `timing_benchmark.csv` -- wall-clock `drag()` time per method per
  condition at the recommended resolution, best-of-`--timing-repeats`.
- Console output ends with the recommended method + resolution and whether
  the target accuracy was actually confirmed (a resolution list that never
  reaches the target prints a warning rather than silently reporting the
  largest value as "converged").
