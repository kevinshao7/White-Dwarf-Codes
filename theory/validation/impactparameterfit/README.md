# Impact-Parameter Fit Validation

This directory studies how the finite impact-parameter cutoff changes the
drag model in `theory/dragbase2.py`.

For scripts that sweep `bmax/aH`, `--rhores` and `--dphires` are interpreted as
the impact-bin and scattering-angle resolutions at `bmax/aH = 1`.  The
finite-launch impact grid is equal-area, so maintaining a consistent physical
impact-bin width requires `rhores_effective = ceil(rhores * (bmax/aH)^2)`.
The scattering-angle quadrature uses `dphires_effective = ceil(dphires *
bmax/aH)`.

## Main Scripts

- `run_impact_parameter_fit.py`
  Fits one parameter, `bmax/aH`, for each condition.  The launch radius is held
  fixed at `50 aH`, so the fitted value is converted to the solver input as
  `rhomax_fraction = (bmax/aH) / 50`.

- `run_fixed_bmax_ah_comparison.py`
  Runs the same model without optimization, using one fixed cutoff
  `bmax/aH = 1` for all conditions.

- `impactparametershape.py`
  Sweeps selected `bmax/aH` values and velocities using the old `common.py`
  path, where `DragFourth` keeps the naive finite-radius setup and the
  asymptotic angular momentum `L = mu b v_inf`.

- `impactparametershape_finite_start.py`
  Runs the same sweep using `commonfinite.py`, where particles are launched
  from a finite radius and angular momentum is computed self-consistently as
  `L = mu r_start v_start sin(theta)`.

- `plot_impact_parameter_trends.py`
  Replots the fitted cutoff from `impact_parameter_fit_summary.csv` in several
  normalizations: `aH`, ion screening length, Yukawa screening length, and
  electron Debye length.

- `plot_closest_approach_vs_bmax.py`
  Sweeps `bmax/aH` and plots the point of closest approach, `rmin`, for
  selected retained impact-parameter bins inside the drag integral.  By default
  it samples inner, middle, and outer retained bins.  This directly checks how
  the impact-parameter cutoff also moves the radial cutoff used by the finite
  scattering-angle integral.

- `plot_impact_bin_contributions.py`
  Decomposes the finite-launch drag integral into launch-impact-parameter bins.
  The x axis is `p/aH`, where `p` is the impact parameter at the finite launch
  radius, and the y axis is the drag-force contribution from that bin after
  summing over the thermal velocity grid.

## Generated Outputs

- `impact_parameter_fit_summary.csv`
  Best-fit cutoff and derived length-scale normalizations.

- `impact_parameter_fit_predictions.csv`
  Model/data comparison at the selected fit points.

- `impact_parameter_fit_curve.csv`
  Smooth model curves at the fitted cutoffs.

- `impact_parameter_shape_curves.csv`
  Drag values from the explicit `bmax/aH` sweep in `impactparametershape.py`
  using the old `common.py` path.

- `impact_parameter_shape_curves_finite_start.csv`
  Drag values from the explicit `bmax/aH` sweep in
  `impactparametershape_finite_start.py` using the new `commonfinite.py` path.

- `closest_approach_vs_bmax.csv`
  Closest-approach diagnostics from `plot_closest_approach_vs_bmax.py`.

- `impact_bin_contributions.csv`
  Per-bin drag-force contributions from `plot_impact_bin_contributions.py`.

## The `C` Reference Term In The Scattering Integral

The active solver does not have a separate Python function named `cfunction`.
The important quantity is the local Coulomb reference coefficient

```text
C = A exp(-k0 / u0)
```

where `u0 = 1 / rmin` is the inverse radius at closest approach for the current
trajectory.  The Yukawa potential strength changes with radius, which makes the
orbital-angle integral difficult near the turning point.  The solver therefore
subtracts a Coulomb problem whose strength is chosen to match the Yukawa
potential at `rmin`, then integrates the nonsingular difference
`Phi_Y - Phi_C`.

The finite-start correction then subtracts the Yukawa-minus-free angular change
outside the radial cutoff.  Because `acipc` is fixed to `1`, that angle cutoff
radius is the same physical radius as `bmax`.  Increasing `bmax` therefore does
two things at once:

- it allows larger finite-start impact parameters into the cross-section sum;
- it moves the outer limit of the finite scattering-angle correction.

For that reason, increasing `bmax` does not have to strictly increase the final
drag.  The cross-section domain grows, but the scattering angle assigned to each
trajectory is also recalculated with a different finite-angle cutoff.

## Rerun Commands

```powershell
python .\theory\validation\impactparameterfit\run_impact_parameter_fit.py --workers 8
python .\theory\validation\impactparameterfit\impactparametershape.py --workers 8
python .\theory\validation\impactparameterfit\impactparametershape_finite_start.py --workers 8
python .\theory\validation\impactparameterfit\plot_closest_approach_vs_bmax.py
python .\theory\validation\impactparameterfit\plot_impact_bin_contributions.py
```
