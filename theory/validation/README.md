# Drag validation scripts

Each validation folder has a standalone Windows-safe script using
`ProcessPoolExecutor`. Run from the repository root:

```powershell
python .\theory\validation\cutoffradius\run_cutoff_radius_convergence.py --workers 8
python .\theory\validation\impactparameterfit\run_impact_parameter_fit.py --workers 8
python .\theory\validation\resolution\run_resolution_convergence.py --workers 8
python .\theory\validation\strongcouplinglimit\run_rutherford_limit.py --workers 8
python .\theory\validation\velocitylimits\run_velocity_limits.py --workers 8
```

Useful quick-run options:

```powershell
python .\theory\validation\velocitylimits\run_velocity_limits.py --conditions 0 --workers 2 --vres 6 --rhores 12 --ures 12 --dphires 12
python .\theory\validation\impactparameterfit\run_impact_parameter_fit.py --conditions 0 --workers 2 --max-fit-evaluations 4 --curve-points 6 --fit-points-per-condition 2 --vres 6 --rhores 12 --ures 12 --dphires 12 --min-velocity-cm-s 1e6
```

Default validation resolutions are `vres=50`, `rhores=180`, `ures=180`,
and `dphires=180`. Lower these on the command line only for smoke tests.

Velocity inputs in the plots are labeled in `cm/s`; calls into
`dragbase2.py` are converted to SI `m/s`. The fixed velocity cases are:
`1 cm/s`, the code's one-dimensional thermal speed
`sqrt(kb T / mu)`, and `2e7 cm/s`.

Each script writes a CSV and PNG files in its own folder. The plotted y
axes are physical quantities: drag validations show `|drag| [N]`, and the
Rutherford-limit validation shows scattering angle in radians.

The cutoff validation writes separate plots for `vrel`, `rhomax`, and
`umax`. The resolution validation writes separate plots for `vres`,
`rhores`, `ures`, and `dphires`. The velocity-limit validation samples up
to `1e8 cm/s`.

The impact-parameter fitting validation fits the upper impact-parameter
bound against the LAMMPS exponential-fit data used for `fit.png`. By
default it reads `unforced/dataarchive/nprun4_29/results.npy`, expands each
valid row into 10 points with `v = A exp(-t/tau)` and `a = v/tau`, uses
only weakly coupled condition indexes `0` and `2`, and fits the 4 data
points with the lowest relative acceleration error in each condition
(`--fit-points-per-condition 4`, ranked by
`acceleration_sigma_cm_s2 / acceleration_cm_s2`). It fits `rhomax` as
`bmax / lD`, where `lD` is the electron Debye length, and converts theory
force to acceleration with `drag_N / DragFourth.ms * 100`. The script
converts this fitted Debye-length fraction into the
`DragFourth.rhomax_fraction` value used internally. The two weakly coupled
cases are fit in one combined least-squares solve, and each residual
evaluation maps the expensive drag calls over a `ProcessPoolExecutor`,
defaulting to 8 workers. With the default 4 fit points per condition, this
dispatches 8 drag calculations per residual evaluation. The best-fit curves
use the same pool. The plotted best-fit line uses 24 adaptively spaced
velocity points by default, with denser spacing around the data acceleration
peak.
The best-fit legend reports the coupling parameter `Gamma`, the standard
least-squares covariance estimate for `bmax / lD`, and the equivalent
`bmax / lS`, where `lS = 1 / k0` is the temperature-dependent
Thomas-Fermi/Yukawa screening length in `dragbase2.py`. It writes
`impact_parameter_fit_summary.csv`, `impact_parameter_fit_predictions.csv`,
`impact_parameter_fit_curve.csv`, and `impact_parameter_fit.png`.
Pass `--data-csv` with columns `condition`, `velocity_cm_s`,
`acceleration_cm_s2`, and optionally `acceleration_sigma_cm_s2` to fit a
curated dataset instead of the archived result arrays.

In `dragbase2.py`, the default outer-radius cutoff is the hydrogen
interparticle spacing, because `ustart = 1 / interparticlespacing`. The
electron Debye radius `lD` is computed but is not the default outer cutoff.
The default `rhomax` cutoff is `0.3` times the hydrogen interparticle
spacing, and the default relative-velocity integration half-width is
`4` thermal sigmas.

For velocity-limit fitting, the high-velocity slope is fit only on sampled
points above the detected Bragg peak, not across the peak itself.
