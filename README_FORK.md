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

## Known caveat

At 200px the axis *ratios* are robust but the absolute axes are not: 97% of
subwindows fit `a1` larger than the window itself. An accidental duplicate fit of
one subwindow gave two independent converged solutions agreeing on the ratios to
four decimals (0.1911 vs 0.1912) while disagreeing on `a1` (3.48 vs 3.99 ly).
Treat absolute sizes from small windows with suspicion.
