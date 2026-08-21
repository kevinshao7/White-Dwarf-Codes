# Resolution convergence

Demonstrates how the predicted drag force from `FiniteLaunchDrag`
(`theory/finite/finite_launch.py`, `method="vectorized"`) depends on each of
its three independent resolution knobs, for condition 0 only.

Depends only on `theory/finite/` and `theory/dragbase2.py` -- nothing under
`theory/validation/`.

## What it measures

Three grids can each be refined independently:

- `vres` -- midpoint rule over the relative-velocity Maxwellian inside `drag`.
- `rhores` -- log-spaced midpoint rule over the impact parameter `b` inside
  `impact_parameter_integral`.
- `dphires` -- midpoint rule over the regularised scattering-angle integral
  inside `orbit_angle`.

The script sweeps each one individually while holding the other two fixed at
the largest value in `--resolutions`, and evaluates `drag(vb)` (condition 0,
`method="vectorized"` only -- no `scipy.integrate.quad` call anywhere in this
script) across a log-spaced velocity grid for every `(scan type, resolution,
velocity)` combination.

Relative error is measured against each scan's own finest tested resolution
(the largest value in `--resolutions`), not a separate quad-based ground
truth -- that resolution's row is already part of the sweep, so no extra
evaluations are needed.

## Run

```powershell
python .\theory\finite\convergence\run_resolution_convergence.py --workers 8
```

Runtime scales as `3 scan types * len(resolutions) * len(velocities)`
`drag()` evaluations. With the defaults (4 resolutions, 16 velocities) this is
192 evaluations, all `method="vectorized"` -- fast relative to the old
quad-based version of this script.

## Outputs

- `condition_0_resolution_scan.csv` -- one row per `(scan_type, resolution,
  velocity)`: `vres`/`rhores`/`dphires` actually used, the drag force, and the
  relative error vs. that scan's finest tested resolution.
- `condition_0_drag_vs_velocity.png` -- drag force (N) vs. bulk velocity
  (cm/s), log-log. Colour encodes resolution (light -> dark as resolution
  increases); linestyle encodes scan type (solid = `vres`, dotted = `rhores`,
  dashed = `dphires`).
- `condition_0_relative_error_vs_velocity.png` -- same colour/linestyle
  scheme, relative error (log scale) vs. the same velocity axis. The
  reference resolution for each scan is omitted (its error is identically
  zero, which can't be shown on a log axis).

Both plots title with condition 0's temperature and density, read directly
off the `FiniteLaunchDrag` instance (`T`, `gcc`) so the label can't drift from
`dragbase2.py`.
