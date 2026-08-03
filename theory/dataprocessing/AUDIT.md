# Audit of the existing reduction

## Inventory

- Requested raw directory: `unforced/dataarchive/run4_29`
- Expected grid from the notebook: 60 velocities x 4 conditions = 240 cases
- Raw trajectory/log pairs currently present: 182
- Intermediate `force_*.np`/`log_*.np` pairs: 172
- Legacy `results.npy` valid entries: 172

Thus 58 simulations are absent from the raw archive and a further 10 raw
simulations were lost during the old raw-to-intermediate reduction. Absence is
not evidence of simulation failure because the notebook swallowed all
exceptions.

## High-value reliability issues

1. `except:` around parsing and fitting converts every failure into the word
   `fail` or `error`, losing the filename, exception, and failure stage.
2. The log parser depends on one exact whitespace string and then removes six
   rows by position. This is fragile across LAMMPS output variants.
3. The dump parser removes the first and last snapshots using unconditional
   `pop()` calls rather than validating completeness.
4. Fit windows are chosen using the global sample nearest to 99.9% and 93% of
   the first mean. With noisy/non-monotonic traces, those points need not be
   the first threshold crossings or even be in chronological order.
5. `time[starti:endi]` excludes `endi`, but the saved end time is
   `time[endi]`, so the stored fit interval does not match the fitted data.
6. Failed/missing values use `-1`, which can be confused with a physical
   signed quantity and requires field-specific validity rules.
7. The fit assumes atom velocities are independent when calculating the
   standard error and assumes saved times are independent in the covariance.
   Neither assumption is checked.
8. Only parameter covariance is saved. No convergence state, residual
   diagnostics, goodness-of-fit, source identity, or configuration accompanies
   the result.
9. The old plot expands each fitted exponential into ten points. Those ten
   points are not ten independent simulation measurements.

## Interpretation of the new flags

Low nominal velocities can have a projected mean comparable to thermal noise.
The new reducer deliberately fails a positive exponential fit when the mean is
nonpositive and flags weak observed decay, poorly constrained decay time,
correlated residuals, poor R-squared, and excessive chi-squared. These outcomes
are information about identifiability, not software failures.

## Recommended next scientific improvements

1. Inspect diagnostic traces for every `review` result before downstream use.
2. Estimate uncertainty with a cluster/block bootstrap that preserves
   time-correlation and, ideally, independent simulation replicas.
3. Compare exponential decay against alternatives (constant/no detectable
   decay, exponential plus offset, and local-slope models) using held-out or
   information-criterion checks.
4. Record why each of the 58 expected raw cases is absent and recover the 10
   raw cases dropped by the old parser.
5. Treat points sampled from one fitted curve as correlated in downstream
   impact-parameter fitting; preferably fit the original decay likelihood
   hierarchically rather than resampling each curve into pseudo-observations.
