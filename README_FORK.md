# Fork notes — Cas A light echo 3D structure function

Fork of [schlafly/casa](https://github.com/schlafly/casa) for the Mab thermal
light echo analysis. Upstream code is unchanged except where noted below.

## Branch layout

All work is on `fixes-and-noise-term`, one commit per logical change so each is
reviewable on its own:

| commit | what |
|---|---|
| `1e0f099` | fix: truncated result files from a `data_dir`/`save_dir` mix-up |
| `0c366ca` | fix: `summarize_chunks` accepts `save_dir` so `path` is correct |
| `8a6731e` | fix: clamp `inner_uv_pixels` — unblocks all sub-400px windows |
| `bca6b42` | feature: `weibull_noise_log_s2`, fits the noise pedestal |
| `f322751` | analysis scripts + result tables |

To compare against upstream, add the remote yourself (the sandbox cannot write
`.git/config`):

```bash
git remote add upstream https://github.com/schlafly/casa.git
git fetch upstream
git log --oneline upstream/main..fixes-and-noise-term
```

The first three commits are candidates for an upstream PR; they are genuine bugs
independent of this project's science. The fourth is a modelling change and is
probably ours to keep, at least until it is validated on the full sample.

## The noise term

`weibull_noise_log_s2` adds a fourth parameter `s2_noise` to the fitted profile:

    S2(r) = var_inf * (1 - exp(-r^beta))^(alpha/beta) + s2_noise

Independent per-pixel measurement noise adds a *constant* `2*sigma_n^2` to S2 at
every non-zero lag. It is a bias, not a variance, so it does not average down
with more lag pairs. The offset-free profile has no parameter for it, so the
optimiser absorbs it into the geometry — and since the pedestal dominates where
`S2_true` is smallest, it flattens the small-lag end specifically, biasing
`alpha` low and inflating the correlation length. The radius entering the
profile is the ellipsoid-normalised radius, so that radial bias propagates
directly into the fitted axis lengths and ratios.

It is added in *linear* S2 space, before the log, because that is where the
noise variance actually adds. It is *fitted* rather than subtracted (even though
`analysis/noise_audit.py` measures a per-window floor that could be subtracted)
so that the floor's own uncertainty propagates into the parameter errors.

Added as a new profile rather than by editing `weibull_log_s2`, so earlier runs
stay reproducible and the two models are a controlled comparison differing by
exactly one parameter. Pass it explicitly:

```python
fit = sf.fit_s2(s2, profile=sf.weibull_noise_log_s2, weighting='1/r')
```

## Data

Bulk arrays are **not** in the repo. They are symlinked into `data/` from
`~/Dropbox/TLE_Data`:

    resampled_epochs_noclip.npy  U_grid.npy  V_grid.npy
    epoch_mean_w.npy  edge_mask_r50.npy

`data/chunk_windows.csv` *is* tracked (it comes from upstream). Per-window fit
outputs (`data/sf_fits/`, `data/jk_q4/`, `data/jk_sub200/`) are gitignored — they
are large and regenerable.

## Analysis scripts

- `analysis/noise_audit.py` — per-window noise floor and structure SNR from the
  count-weighted radial profile of the same-epoch planes.
- `analysis/jackknife_q4.py` — 2x2 block jackknife, 29 top-SNR-quartile 400px
  windows, one resumable JSON per window.
- `analysis/jackknife_sub200.py` — the same at 200px (4 quadrants per parent,
  116 subwindows).

Both jackknife scripts keep the per-sample parameter *vectors* rather than
collapsing to marginal standard errors: errors on the axis ratios cannot be
recovered from stderrs on `a1`/`a2`/`a3`, because the axes are strongly
covariant and the two ratio errors are anticorrelated (median rho = -0.43). The
correct visualisation is a tilted 2x2 error ellipse per window, not an upright
cross.

## The arcsinh transform, and the raw-flux rerun

Every published number is measured on `arcsinh(flux / 0.03)` after subtracting an
additive floor of 0.03 from the flux. Two points about that, because the code's
own names for it are misleading:

- The `background=0.03` passed to `compute_s2` is an **additive flux floor**
  unrelated to the echo, *not* a noise level, and it is separate from the
  per-epoch sky already removed when the chunk products were built.
- The comment in `structure_function.py` calling `arcsinh_scale` a noise scale,
  and the docstring saying it is "typically set to the per-pixel noise level",
  do not describe what the parameter does here. Its function is **dynamic-range
  compression**. Do not quote either label in the paper.

Because the floor subtraction happens *inside* the nonlinearity, the two being
the same number (0.03) is one coupled choice, not two independent knobs. That
makes "how much does the transform shape the result?" a real referee question,
so the whole 115-window pipeline was rerun in raw flux units with no transform
and no floor, changing nothing else.

**Raw flux is now the default.** Every driver reads `COMPUTE_KW` from
`analysis/scale_split.py`, which selects raw flux unless `CASA_ARCSINH_UNITS=1`
is set (the legacy `CASA_LINEAR_UNITS=0` does the same; if both are set, arcsinh
wins). It is deliberately one module-level switch rather than a per-script flag:
it has to be identical for the fits, the SNR audit and the figures, and the
drivers reach `COMPUTE_KW` through module state inside `spawn`-ed pool workers,
which would not observe a mutation made in the parent process.

**Filenames keep the `_linear` suffix even though raw flux is the default.** The
suffix was *not* inverted, deliberately: unsuffixed committed files remain the
arcsinh run they have always been, so no tracked path silently changes meaning
between commits. Consumers select the default through
`make_tier_figures.default_variant()` rather than by hard-coding a suffix, so
"which variant is default" lives in exactly one function.

```bash
export PYTHONPATH="$PWD:$PWD/analysis"
# no env var needed -- raw flux is the default

# refit (resumable: one JSON per window, existing files are skipped).
# --windows is REQUIRED; the k=4 run uses all 115 windows.
python analysis/singleband_powerlaw.py --rcut 0.1 --k 4 --procs 6 \
       --windows handoff/all115_windows.json
# re-tier: the SNR table must move WITH the fit table (see below)
python analysis/noise_audit_windows.py
# figures: --variant defaults to '_linear' via default_variant()
python analysis/make_tier_figures.py --k 4
python analysis/make_science_figure.py

# the comparison reads BOTH variants' tables, so it takes no env var
python analysis/compare_linear_vs_arcsinh.py
python analysis/fig_linear_vs_arcsinh.py

# to rebuild the ORIGINAL arcsinh outputs (unsuffixed filenames):
CASA_ARCSINH_UNITS=1 python analysis/make_tier_figures.py --k 4
```

The k=3 top-quartile table used by the two results notebooks needs its own
refit, because the top-SNR quartile is itself re-tiered under raw flux — one
window swaps in — so the window list is not the published `q4_windows.json`:

```bash
python analysis/singleband_powerlaw.py --k 3 --procs 6 \
       --windows handoff/q4_windows_linear.json
```

Thread caps matter for the refit: each worker spawns its own BLAS threads, so
without `OMP_NUM_THREADS=1` (and the OPENBLAS/MKL/VECLIB/NUMEXPR equivalents)
6 processes oversubscribe a 12-core machine badly.

**Every conclusion survives; the significances weaken.** Full table in
`results/linear_vs_arcsinh_headline.csv`, figure in
`results/figures/linear_vs_arcsinh.png`:

| quantity | published | raw flux |
|---|---|---|
| common axis ratios `a1:a2:a3` | 1 : 0.281 : 0.168 | 1 : 0.251 : 0.134 |
| prolate significance | 6.85 sigma | 5.03 sigma |
| windows individually prolate | 25 / 29 | 25 / 29 |
| intrinsic scatter `a2/a1` | 0.127 dex | 0.151 dex |
| intrinsic scatter `a3/a2` | 0.100 dex | 0.140 dex |
| p(one exact shape), `a3/a2` | 8e-08 | 3e-16 |
| common small-lag slope `alpha` | 0.583 | 0.579 |
| rho(inclination vs `b2/b1`) | -0.694 (p=3e-05) | -0.596 (p=6e-04) |
| median inclination to echo plane | 73.6 deg | 68.1 deg |

The direction is coherent and is what removing dynamic-range compression should
do: the shapes come out slightly *more* elongated, the per-window error bars
grow, and every significance that depends on those error bars drops. Result 2 is
the exception that confirms it — the *scatter* grows, so the "one exact shape"
null is rejected harder, not more weakly.

Two things were checked so the comparison is genuinely single-variable:

- **The tiering.** The `snr` column is a ratio of S2 plateau to S2 floor, and the
  nonlinearity compresses those two differently, so the SNR table cannot be
  reused across variants — `load()` therefore refuses to mix them, and the tier
  legend strings are rewritten from whichever table is loaded rather than
  hard-coded. Recomputing the tiering turns out to be nearly a no-op
  (Spearman 0.994 between the two SNR orderings; the top quartile keeps 28 of 29
  windows), and the `linear_frozen_tiers` column in the comparison table shows
  the shifts are attributable to the refit, not to sample membership.
- **Convergence.** Fit success is unchanged (114/115) and the usable count moves
  by one (113 -> 112), but collapsed fits — either ratio below `DEGEN = 0.02`,
  the repository's own cut — rise from 13 to 18, so the figures draw 94 windows
  instead of 100. One collapsed window (row 2000, col 3600) lands in the top
  quartile at `a2/a1 = 0.0021`. It was left in: that is 27 dex of log-space
  standard error, so the ML estimator already gives it essentially no weight.
  Excluding it by hand would be a worse choice than letting the likelihood
  handle it.

One row in the table does change qualitatively and should not be over-read:
`frac_within30_of_W` goes from exactly 0 to 0.034. That is a single window
(row 3200, col 2000) whose fitted inclination moves 76 deg -> 14 deg — but its
own inclination error is 39 deg, so it is unconstrained in both runs rather than
newly pointing along W.

## Known caveat

At 200px the axis *ratios* are robust but the absolute axes are not: 97% of
subwindows fit `a1` larger than the window itself. An accidental duplicate fit of
one subwindow gave two independent converged solutions agreeing on the ratios to
four decimals (0.1911 vs 0.1912) while disagreeing on `a1` (3.48 vs 3.99 ly).
Treat absolute sizes from small windows with suspicion.
