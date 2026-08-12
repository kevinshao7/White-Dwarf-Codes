# Drag-curve shape convergence

This test calculates drag curves for conditions 1 and 3 at exactly 16 focused
velocities while sweeping the impact-parameter cutoff from `0.1` to `10`
hydrogen interparticle spacings:

```powershell
python .\theory\validation\shapeconvergence\run_shape_convergence.py --workers 8
```

The command-line parameter is `--bmax-over-spacing`; values are interpreted as
`b_max / a_H`, where `a_H` is the hydrogen interparticle spacing. The solver
still launches trajectories at a larger finite radius, `50 a_H`, so internally
the script passes `rhomax_fraction = (b_max / a_H) / 50` to `DragFourth`.
This keeps the plot labels and CSV columns in physical cutoff units rather than
the finite-launch radius fraction.

The velocity grid uses 3 points on the low-velocity tail, 10 points from
`1e6` to `3e7 cm/s` around the expected Bragg peak, and 3 points on the
high-velocity tail. Each condition gets a PNG named
`condition_<condition>_rhomax_fraction_shape_convergence.png`, and all
numerical data are saved in `rhomax_fraction_shape_convergence.csv`.

Before launching the sweep, the script requires a zero-force calculation to
vanish within `1e-30 N`. Negative drag values are marked explicitly with
crosses rather than silently folded into the positive curves.
