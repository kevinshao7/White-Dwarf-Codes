# Unforced simulation data processing

This directory contains a reproducible replacement for the fitting portion of
`unforced/unforcedcodes/datareduction.ipynb`.

## Important provenance finding

`unforced/dataarchive/run4_29` contains 182 trajectory/log pairs. These files
are ignored by normal repository file enumeration, so use filesystem-aware
inventory commands when auditing them. The old reduction produced only 172
intermediate `force_*.np` files in `nprun4_29`; ten raw cases were silently
lost. The new pipeline reads the raw trajectories by default.

Every output records the raw input filename and SHA-256 digest. The manifest
also inventories the raw source. Use `--source intermediate` only to reproduce
or diagnose the old intermediate stage.

## Improvements over the notebook

- deterministic CLI rather than hidden notebook state;
- strict input shape, finiteness, and monotonic-time validation;
- explicit failures and review flags rather than bare `except: print("error")`;
- filename-based discovery rather than assuming all 240 cases exist;
- direct raw LAMMPS dump parsing with snapshot and atom-count validation;
- positive parameterization of amplitude and decay time;
- robust nonlinear least squares;
- covariance, goodness-of-fit, decay-strength, and residual-correlation checks;
- unambiguous half-open fit-window indices;
- CSV output for inspection plus a downstream-compatible NumPy array;
- `NaN` for missing values rather than the ambiguous `-1` sentinel;
- source hashes and a machine-readable run manifest.

The uncertainty remains a conditional fit uncertainty. It does not prove that
Si atoms or adjacent saved times are statistically independent. Correlated
residuals are flagged; bootstrap or replicate simulations are the next
substantive uncertainty improvement.

## Run

From the repository root:

```powershell
python .\theory\dataprocessing\process_unforced.py
```

Campaigns are processed in parallel with 8 worker processes by default. Change
this with `--workers N`.

For a quick smoke test:

```powershell
python .\theory\dataprocessing\process_unforced.py --limit 2 --output-dir .\theory\dataprocessing\smoke_output
```

Outputs:

- `fit_results.csv`: auditable result and diagnostics per trajectory;
- `results.npy`: shape `(4, 60, 6)`, preserving the legacy field order;
- `manifest.json`: configuration, environment, counts, and provenance audit.
- `diagnostics/index.html`: browsable graphical review of every campaign;
- `diagnostics/*.png`: one velocity-curve fit diagnostic per campaign.

Each diagnostic has a data/fit panel and a normalized-residual panel. The
selected fitting interval is shaded. Orange (`review`) and red (`failed`)
figures require attention before downstream use. Pass `--no-plots` only for
non-graphical batch processing.

## Current exclusion and fit-window rules

After the configured thermalization rows:

1. Find the maximum mean projected velocity.
2. Calculate the physical time interval `t` between the end of thermalization
   and that maximum.
3. Set the target fit start to the thermalization-end time plus `2*t`, then use
   the first recorded sample at or after that physical time.
4. Before fitting, use the maximum window (through the final simulation
   sample) and mark the campaign `ignored` if the velocity SEM is at least
   100% of the absolute mean at either fit endpoint, or if the start and end
   mean ± SEM intervals overlap.
5. Treat the fit-window end as a discrete fitted choice. Candidate windows
   must span at least 10% of the full simulation duration.
6. For every candidate endpoint, fit a positive exponential with
   `tau >= 1e-20 s`, then select the endpoint minimizing
   `sum(((model-data)/SEM)^2) / N`. The exponent is available as
   `FitConfig.window_length_score_power`.

Ignored campaigns remain in `fit_results.csv` with their reasons, but are not
written into `results.npy` and receive no plot. At the beginning of a plotted
run, the script deletes only its own old `condition_*_velocity_*.png` files and
diagnostic `index.html`, preventing stale plots from surviving a changed fit.

`status=ok` means no configured warning fired. `status=review` is a numerical
result that should be inspected before scientific use. `status=failed` is not
written into the compatible result array.
