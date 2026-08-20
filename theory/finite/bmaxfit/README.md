# Velocity-dependent b_max fits

`theory/finite/lammps_fit/fit_bmax_to_lammps.py` fits one `b_max/a_H`
(`rhomax_fraction` in `FiniteLaunchDrag`, `theory/finite/finite_launch.py`)
per condition, held constant across the whole velocity range, against
LAMMPS/DAIS drag simulations. The two scripts here both ask whether that
constant should instead vary with velocity, in two different ways. Both
build directly on `fit_bmax_to_lammps.py` (via `common.py`, which flat-imports
it as `fit_bmax_to_lammps`) for data loading, point filtering/selection, and
drag-evaluation plumbing -- neither reimplements it.

## `fit_bmax_per_point.py` -- one b_max per (condition, velocity) point

For each selected LAMMPS point, solves `b_max/a_H` in isolation via
`scipy.optimize.brentq` root-finding on the log-residual, so the single free
parameter exactly reproduces that one point's acceleration. Sweeping the
result against velocity is the direct empirical answer to "how does b_max
vary with v" -- no assumption about the functional form.

Parallel across (condition, point) tasks: each task is one full,
independent root-find, submitted whole to a `ProcessPoolExecutor`.

```powershell
python .\theory\finite\bmaxfit\fit_bmax_per_point.py --workers 8
```

Outputs: `bmax_per_point_fit.csv` (one row per point: condition, velocity,
best-fit `b_max/a_H`, model/data acceleration, convergence flags),
`condition_<n>_bmax_per_point.png` (b_max/a_H vs. velocity per condition,
overlaid on the raw LAMMPS points), `bmax_per_point_all_conditions.png`
(all four conditions' trends on one axis).

Points that can't be bracketed within `[--bmax-min, --bmax-max]` (the model
can't reach that data point anywhere in the allowed range) are reported
bound-clamped with `converged=False` rather than silently returned as a
real fit -- see the `converged`/`at_lower_bound`/`at_upper_bound` columns.

## `fit_bmax_two_regimes.py` -- one b_max below, one above thermal velocity

For each condition, splits selected points at the thermal width
`v_th = sqrt(kB T / mu)` (`FiniteLaunchDrag.drag`'s `sigma_v`) into a
below-`v_th` and an above-`v_th` group, then runs
`fit_bmax_to_lammps.fit_condition` -- the same `least_squares` fit used for
the single-value model, unmodified -- once per group. Produces 2 x 4 = 8
fitted values, directly comparable to `fit_bmax_to_lammps`'s 4 since the
residual definition and bounds are identical.

Parallelism is inherited from `fit_condition`: each regime fit's
`least_squares` iterations submit one drag evaluation per fit point to a
shared `ProcessPoolExecutor`.

```powershell
python .\theory\finite\bmaxfit\fit_bmax_two_regimes.py --workers 8
```

Outputs: `bmax_two_regimes_fit_summary.csv` (one row per condition x regime:
best-fit `b_max/a_H`, its 1-sigma uncertainty, reduced chi^2, thermal
velocity used for the split, plus `best_bmax_over_debye_length` for the
weakly-coupled conditions 0/2 -- same fields as `fit_bmax_to_lammps`'s
summary), `bmax_two_regimes_fit_predictions.csv` (per-point predictions,
tagged with `regime`), `condition_<n>_bmax_two_regimes_overlay.png` (both
regimes' fit points/model curves on one log-log plot, with a dashed line at
`v_th`).

If a condition's filtered points fall entirely on one side of `v_th`, that
regime is skipped for that condition (printed, not silently dropped) rather
than fit against zero points.

## Shared: `common.py`

Not runnable on its own. `thermal_velocity_cm_s(condition)` computes
`v_th`; `build_common_parser` / `load_and_filter_points` hold the CLI
options and data loading/filtering both scripts share (results paths,
velocity range, `--max-relative-sigma`, `b_max` bounds, `--method`,
`--resolution`, `--vres`, `--workers`, `--heartbeat-seconds`). Each script
adds its own point-*selection* arguments on top
(`--points-per-condition`/`--points-per-regime`), since how many points to
select means something different for a per-point solve than for a
per-regime `least_squares` fit.
