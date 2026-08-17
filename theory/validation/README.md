# Drag validation scripts

Each validation folder has a standalone Windows-safe script using
`ProcessPoolExecutor`. Run from the repository root:

```powershell
python .\theory\validation\acipc\run_acipc_fit.py --workers 8
python .\theory\validation\cutoffradius\run_cutoff_radius_convergence.py --workers 8
python .\theory\validation\impactparameterfit\fit_bmax_to_lammps.py --workers 8
python .\theory\validation\resolution\run_resolution_convergence.py --workers 8
python .\theory\validation\shapeconvergence\run_shape_convergence.py --workers 8
python .\theory\validation\strongcouplinglimit\run_rutherford_limit.py --workers 8
python .\theory\validation\velocitylimits\run_velocity_limits.py --workers 8
```

Useful quick-run options:

```powershell
python .\theory\validation\acipc\run_acipc_fit.py --conditions 0 --workers 2 --fit-points-per-condition 4 --max-fit-evaluations 2 --curve-points 4 --vres 6 --rhores 12 --ures 12 --dphires 12 --quiet
python .\theory\validation\velocitylimits\run_velocity_limits.py --conditions 0 --workers 2 --vres 6 --rhores 12 --ures 12 --dphires 12
python .\theory\validation\impactparameterfit\fit_bmax_to_lammps.py --conditions 0 --workers 2 --max-fit-evaluations 4 --curve-points 6 --fit-points-per-condition 2 --vres 6 --rhores 12 --ures 12 --dphires 12 --min-velocity-cm-s 1e6
```

Default validation resolutions are `vres=201`, `rhores=180`, `ures=180`,
and `dphires=180`. Lower these on the command line only for smoke tests.

Velocity inputs in the plots are labeled in `cm/s`; calls into
`dragbase2.py` are converted to SI `m/s`. The fixed velocity cases are:
`1 cm/s`, the code's one-dimensional thermal speed
`sqrt(kb T / mu)`, and `2e7 cm/s`.

Each script writes a CSV and PNG files in its own folder. The plotted y
axes are physical quantities: drag validations show `|drag| [N]`, and the
Rutherford-limit validation shows scattering angle in radians.

The acipc validation fits two physical cutoff parameters per condition:
`acipc = r_angle,max / b_max` and the impact-parameter cutoff length
`b_max` in meters. It reads the same default LAMMPS exponential-fit data as
the impact-parameter fitting validation, selecting one low-error observation
from each of several log-spaced experimental campaigns.

The cutoff validation writes separate plots for `vrel`, `rhomax`, and
`umax`. The resolution validation writes separate plots for `vres`,
`rhores`, `ures`, and `dphires`. The velocity-limit validation samples up
to `1e8 cm/s`.

The shape-convergence validation resolves the condition-0 drag curve across
the low-velocity tail, Bragg region, and high-velocity tail for several
finite-launch cutoff fractions.

The impact-parameter fitting validation fits the upper impact-parameter
bound `bmax/aH` against LAMMPS exponential-fit data. It reads
`theory/dataprocessing/output/results.npy` for strongly coupled cases and
`theory/dataprocessing/output_dais/results.npy` for DAIS cases, expands each
valid row into points with `v = A exp(-t/tau)` and `a = v/tau`, and selects
low-error points per condition. The fitted `bmax/aH` is converted to
`DragFourth.rhomax_fraction = (bmax/aH) / 50`. The best-fit plot is generated
by `impactparameterfit/fit_bmax_to_lammps.py` and written to
`bmax_fit_lammps_overlay.png`; companion tables are
`bmax_fit_summary.csv`, `bmax_fit_lammps_predictions.csv`, and
`bmax_fit_model_curves.csv`.
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
