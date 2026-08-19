# LAMMPS impact-parameter cutoff fit

Fits the single free parameter of the finite-launch model,
`b_max/a_H = rhomax_fraction`, per condition, against LAMMPS (and DAIS, for
conditions 0 and 2) molecular-dynamics drag simulations. Launch radius `r_i`
is held fixed at the hydrogen interparticle spacing `a_H`.

Depends only on `theory/finite/`, `theory/dragbase2.py`, and
`theory/dataprocessing/output{,_dais}/results.npy` -- nothing under
`theory/validation/`. Adapted from
`theory/validation/impactparameterfit/fit_bmax_to_lammps.py` with two
deliberate simplifications (see the module docstring in
`fit_bmax_to_lammps.py` for the reasoning):

1. **Single parameter, not two.** The old model had a separate
   `cutoff_radius_factor` (launch radius, `r_i = 50 a_H`) and impact-parameter
   cutoff; here `r_i = a_H` is fixed, so `b_max/a_H` is bounded to `(0, 1]`
   rather than `(0, 50]`. A fit that lands on the `1.0` boundary means the
   model's LAMMPS-implied cutoff wants to exceed the launch radius itself --
   that is a real finding about the model, not a fit failure, and the script
   prints a warning rather than silently reporting the boundary value as a
   converged answer.
2. **Simplified point selection.** Points are ranked by velocity, split into
   `--points-per-condition` quantile groups, and the lowest-relative-sigma
   point is kept per group. The old script's per-campaign regex grouping and
   condition-3-specific low/high velocity split are not reproduced.

## Run

```powershell
python .\theory\finite\lammps_fit\fit_bmax_to_lammps.py --workers 8
```

Cost is dominated by `least_squares` iterations x `--points-per-condition`
parallel drag evaluations per iteration. With defaults (8 points/condition,
`rhores=dphires=720`, `method=vectorized`) each drag evaluation is fast (the
`vectorized` scheme was benchmarked at ~40-90x faster than `quad_quad` during
development -- see `theory/finite/convergence/`); expect the full fit across
4 conditions to take a few minutes on 8 cores. Re-run
`theory/finite/convergence/run_resolution_convergence.py` before trusting the
`--resolution 720` default at a materially different velocity range or
condition set, and pass a different `--resolution` if that scan recommends
one.

Every point evaluation inside every `least_squares` iteration is printed on
completion, plus a wall-clock heartbeat (`--heartbeat-seconds`, default 12s)
if nothing has completed recently -- with a slower `--method` or higher
`--resolution`, a single iteration can otherwise run silently for a long
time before the old one-line-per-iteration summary printed at all.

## Outputs

- `bmax_fit_summary.csv` -- one row per condition: best-fit `b_max/a_H`,
  its 1-sigma uncertainty (from the least-squares Jacobian covariance),
  reduced chi^2, convergence flag, and whether the fit sits on the upper
  bound.
- `bmax_fit_predictions.csv` -- per fit-point model vs. data acceleration,
  log residual, and weighted log residual.
- `condition_<n>_bmax_fit_overlay.png` -- all LAMMPS points (gray), the
  points actually used in the fit (red, with error bars), and the fitted
  model curve (blue), log-log velocity vs. acceleration.

## Caveats

- `--bmax-max` cannot exceed `1.0`; the script rejects it at startup.
- The default initial guess is `1.0` only when it is strictly interior to
  `(--bmax-min, --bmax-max)`; with the default bounds `(0.01, 1.0)`, `1.0` is
  the upper bound itself, so the geometric mean `sqrt(bmax_min * bmax_max)`
  is used instead. Starting a gradient-based optimizer exactly on a bound can
  pin it there before it explores the interior -- confirmed during
  development that the geometric-mean start does explore properly.
- If a condition's data pushes the fit to `1.0` (the `r_i` boundary), the
  single-parameter model cannot represent that condition's LAMMPS drag;
  letting `r_i` vary as a second fit parameter would be the next step, not
  reflected in this script.
