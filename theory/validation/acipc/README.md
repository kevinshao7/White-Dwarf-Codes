# Two-parameter acipc fit

This validation fits the two physical cutoff parameters directly:

```text
acipc = r_angle,max / b_max
b_max = impact-parameter cutoff length [m]
```

The default data source matches the impact-parameter fitter:

```text
unforced/dataarchive/nprun4_29/results.npy
```

For each condition, the default fit set uses one observation from each of
eight distinct LAMMPS campaigns. Campaigns are selected near log-spaced
velocity-regime targets, and the chosen observation inside each campaign is
the point with the lowest relative acceleration error.

Run the full fit with:

```powershell
python .\theory\validation\acipc\run_acipc_fit.py --workers 8
```

For a quick smoke test:

```powershell
python .\theory\validation\acipc\run_acipc_fit.py --conditions 0 --workers 2 --fit-points-per-condition 4 --max-fit-evaluations 2 --curve-points 4 --vres 6 --rhores 12 --ures 12 --dphires 12 --quiet
```

Outputs are written next to the script:

```text
acipc_fit_summary.csv
acipc_fit_predictions.csv
acipc_fit_curve.csv
acipc_fit.png
```
