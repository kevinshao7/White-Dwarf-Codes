# Finite-launch versus analytic weak-Yukawa drag

`compare_weak_yukawa.py` compares the finite-launch Si--H Yukawa calculation
with two parameter-free analytic weak-coupling references across the Bragg
peak region:

- Landau/Spitzer Maxwellian dynamical friction (the form given by Gurnett &
  Bhattacharjee and Boyd & Sanderson), with a smooth Yukawa-screened Coulomb
  logarithm;
- the first-Born Yukawa momentum-transfer logarithm.

Both use the finite solver's pair potential, reduced mass, density,
temperature, and Melrose-corrected screening length.  Their lower cut-off is
the physical combination `sqrt(b_90**2 + (hbar/(2*mu*v))**2)`, so there are no
fit parameters.

Run from `White-Dwarf-Codes`:

```bash
python theory/finite/literaturecomparison/compare_weak_yukawa.py --conditions 0 1 2 3
```

It writes `comparison.png` and `comparison.csv` under
`literaturecomparison/weak_yukawa_comparison/`.  The second panel per
condition gives `b_90/lambda_s`; values well below one mark the regime where
the analytic weak-scattering theories are applicable.  The script deliberately
excludes fitted strong-coupling cross sections (Stanton/Sprenkle), Grabowski's
multi-parameter fit, and effective-potential theories.

Finite-launch velocity points are independent and are dispatched through a
24-process pool by default.  Each worker is restricted to one BLAS/OpenMP
thread, preventing oversubscription and allowing all 24 CPU cores to be used.
Use `--workers N` only when fewer cores should be allocated.
