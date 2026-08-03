# Condition-0 numerical shape convergence

This test calculates the numerically sensitive condition 0 at exactly 16 focused
velocities for the finite-launch-safe `rhomax_fraction` values `0.30`,
`0.35`, and `0.40`.
Everything else is held fixed, including `acipc=1`, so the physical
scattering-angle cutoff remains equal to the impact-parameter cutoff.

Eight worker processes are used by default:

```powershell
python .\theory\validation\shapeconvergence\run_shape_convergence.py --workers 8
```

The velocity grid uses 3 points on the low-velocity tail, 10 points from
`1e6` to `3e7 cm/s` around the expected Bragg peak, and 3 points on the
high-velocity tail. It produces only the red condition-0 plot. The curves
show how its drag shape changes with the impact-parameter cutoff. All
numerical data are saved in `rhomax_fraction_shape_convergence.csv`.
Before launching the sweep, the script requires a zero-force calculation to
vanish within `1e-30 N`. Negative drag values are marked explicitly with
crosses rather than silently folded into the positive curves.

The default relative-velocity resolution is `vres=201`. The drag solver pairs
positive and negative relative speeds before accumulating the integral, which
keeps the low-drift thermal cancellation from producing spurious sign flips.
