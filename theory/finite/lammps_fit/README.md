# LAMMPS impact-parameter cutoff fit

Fits the single free parameter of the finite-launch model,
`b_max/a_H = rhomax_fraction`, per condition, against LAMMPS (and DAIS, for
conditions 0 and 2) molecular-dynamics drag simulations. Launch radius `r_i`
moves with the fit: `r_i = rhomax_fraction * a_H` (`a_H` the hydrogen
interparticle spacing), and `b_max` is always forced equal to `r_i`
(`FiniteLaunchDrag.launch_pmax`, `theory/finite/finite_launch.py`), so the
two are tied together rather than `b_max` being an independently bounded
fraction of a fixed `r_i`.

Depends only on `theory/finite/`, `theory/dragbase2.py`, and
`theory/dataprocessing/output{,_dais}/results.npy` -- nothing under
`theory/validation/`. Adapted from
`theory/validation/impactparameterfit/fit_bmax_to_lammps.py` with two
deliberate simplifications (see the module docstring in
`fit_bmax_to_lammps.py` for the reasoning):

1. **Single parameter, not two.** The old model had a separate
   `cutoff_radius_factor` (launch radius, `r_i = 50 a_H`) and independently
   bounded impact-parameter cutoff; here `b_max` is always forced equal to
   `r_i` (a tangent launch), so the one remaining free parameter,
   `rhomax_fraction = r_i/a_H = b_max/a_H`, has no `<= 1` ceiling to enforce
   -- growing it just moves the whole launch sphere (and its tangent
   `b_max`) outward. The fit's `--bmax-max` therefore defaults to infinity. A
   fit that still runs away unbounded means even an arbitrarily distant
   launch sphere can't reach that condition's LAMMPS drag (the Yukawa
   screening makes the drag saturate as `r_i -> infinity`, so this is a real
   finding about the model, not a fit failure) -- the script prints a
   warning rather than silently reporting a huge value as a converged
   answer.
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

## GPU dispatch (`--gpu-devices`)

`--gpu-devices 0,1` batches every point evaluation for a trial `b_max` (one
`least_squares` iteration's worth, or the final/`+-1 sigma` passes) through
`FiniteLaunchDrag.drag_batch(xp=cupy)` (`theory/finite/finite_launch.py`) on
those CUDA devices instead of submitting one CPU task per point to
`--workers`; see `run_fit_points_gpu`'s docstring for how points are split
across devices. Requires `cupy`; the CPU path (the default, no `--gpu-devices`)
needs nothing beyond `numpy`/`scipy`.

**Verify before trusting fitted results from `--gpu-devices`:** `drag_batch`
was validated only on its `numpy` backend (reproduces the scalar `drag()` to
machine precision across all four conditions and several `rhomax_fraction`
values) in an environment with no GPU. The `cupy` code path itself has not
been run on real hardware. Before trusting a fit produced with
`--gpu-devices`, run the same condition/points with and without it and
confirm `best_bmax_over_aH` agrees to several significant figures.

## Outputs

- `bmax_fit_summary.csv` -- one row per condition: best-fit `b_max/a_H`,
  its 1-sigma uncertainty (from the least-squares Jacobian covariance),
  reduced chi^2, convergence flag, and whether the fit sits on the upper
  bound. For the weakly-coupled conditions (`gcc=1e-5`: 0 and 2), also
  `best_bmax_over_debye_length` and its 1-sigma uncertainty -- `b_max`
  expressed as a fraction of the electron Debye length `lambda_De`
  (`DragFourth.lD` in `theory/dragbase2.py`), plus the raw
  `hydrogen_spacing_m`/`debye_length_m` used to convert between the two
  ratios. NaN for the strongly-coupled conditions (1, 3), where the Debye
  length is not the relevant screening scale.
- `bmax_fit_predictions.csv` -- per fit-point model vs. data acceleration,
  log residual, and weighted log residual.
- `condition_<n>_bmax_fit_overlay.png` -- all LAMMPS points (gray), the
  points actually used in the fit (red, with error bars), and the fitted
  model curve (blue), log-log velocity vs. acceleration. A lightly shaded
  blue band shows the model curve at `b_max/a_H` +/-1 sigma. For conditions
  0 and 2, the title/legend also report `b_max/lambda_De`.

## Caveats

- `--bmax-max` defaults to infinity (`DEFAULT_BMAX_MAX`); `scipy.optimize.
  least_squares` handles an unbounded upper bound directly, so no bracket
  growth is needed here (contrast `theory/finite/bmaxfit/fit_bmax_per_point.py`,
  whose `brentq` root-find does need a finite bracket and grows one
  geometrically instead).
- The default initial guess is `1.0` only when it is strictly interior to
  `(--bmax-min, --bmax-max)`; with the default bounds `(0.01, inf)` it is, so
  this reduces to the plain `1.0` start.
- If a condition's data still isn't reached at the `least_squares` optimum,
  the model genuinely cannot represent that condition's LAMMPS drag no matter
  how far the launch sphere moves out -- not a fit failure, but a real
  finding about the model's physics (Yukawa screening bounds how much drag
  is reachable even as `r_i -> infinity`).
