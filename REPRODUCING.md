# Reproducing the results

Every number in `PROJECT_REPORT.md` is reachable from a clean clone. The results
split into two tiers by what they need:

| tier | needs | time |
|---|---|---|
| **A — from tracked fit outputs** | clone only. No data, no `util_efs`. | seconds |
| **B — from the images** | `data/` symlinks to the bulk arrays | hours |

Tier A covers every headline result in the report, because the fit tables and
the per-window jackknife JSON are tracked. Start there; only re-run tier B if
you are changing the fitting itself.

## Environment

Python 3.13, `numpy scipy pandas matplotlib h5py`. Versions used:
numpy 2.5.2, scipy 1.18.0, pandas 3.0.5, matplotlib 3.11.1.

```bash
git clone <this repo> && cd casa
export PYTHONPATH="$PWD:$PWD/analysis"
```

`util_efs` is an external personal library used by exactly one plotting helper
(`structure_function.plot_s2_scatter`). It is **not** required — the import is
soft and everything in `analysis/` works without it.

## Tier A — reproduce the headline numbers

Confirm the install by reproducing the common shape and its intrinsic scatter
(report §1.3, §1.5):

```python
import numpy as np, pandas as pd
from shape_center import ml_center_and_scatter, profile_ci

d = pd.read_csv('results/singleband_powerlaw_r0.1_s2_k3.csv')
d = d[(d['mode'] == '3d') & (d.profile == 'powerlaw')].dropna(
        subset=['a2a1', 'a3a2', 'se_a2a1', 'se_a3a2'])
d = d[(d.a2a1 > 0.02) & (d.a3a2 > 0.02)]          # drop collapsed fits

lv = np.log10(d.a2a1.values); sl = (d.se_a2a1 / d.a2a1 / np.log(10)).values
print(ml_center_and_scatter(lv, sl))    # -> centre 0.2854, sigma_int 0.120 dex
print(profile_ci(lv, sl))               # -> [0.084, 0.161] dex, dchi2(0) = 7.6
```

Expected, and asserted in the report:

| quantity | a2/a1 | a3/a2 |
|---|---|---|
| common value | 0.2854 | 0.6008 |
| intrinsic scatter | 0.120 dex [0.084, 0.161] | 0.096 dex [0.076, 0.119] |
| p(sigma_int = 0) | 0.006 | 6.4e-6 |

The slicing forward model and the 2D sensitivity limit (§1.6):

```python
from slicing_model import predicted_spread, sensitivity_to_3d_scatter
# inc, b2b1, se_b2b1 from the k=4 table joined to results/noise_audit_table.csv
# tiers (see analysis/make_tier_figures.py load()), top SNR quartile, n = 29
np.median(predicted_spread(inc, 0.281, 0.596, 0, 0))          # 0.175 dex
sensitivity_to_3d_scatter(inc, 0.281, 0.596, b2b1, se_b2b1)   # 2-sigma at 0.28
```

Regenerate both deliverable figures:

```bash
python analysis/make_tier_figures.py     # -> results/figures/
```

## Tier B — re-run the fits from the images

Bulk arrays are symlinked into `data/` from outside the repo (see
`README_FORK.md`):

    resampled_epochs_noclip.npy  U_grid.npy  V_grid.npy
    epoch_mean_w.npy  edge_mask_r50.npy

`data/chunk_windows.csv` is tracked (it is upstream's 115-window map).

```bash
python analysis/noise_audit.py                       # SNR tiers -- run first
python analysis/singleband_powerlaw.py --k 4         # the deliverable fit table
python analysis/scale_profile.py                     # 3D scale profile
python analysis/scale_profile_2d.py                  # in-plane twin
```

Order matters: the SNR tiering in every figure comes from
`results/noise_audit_table.csv`, so `noise_audit.py` runs first.

**`--k` is the jackknife blocking and it is not cosmetic.** `k=2` does not
converge; `k=3` and `k=4` agree to about 1% on the axis ratios. `k=4` is the
deliverable. A run at a different `k` writes a differently suffixed table and
will not overwrite the published one.

**Which `k` goes with which number.** The tier-A check above reads the `k=3`
table because that is the one the report tabulates and asserts (0.2854, 0.6008).
`make_tier_figures.py` defaults to `k=4`, so it prints 0.2811 / 0.5958 — the same
result at the other converged blocking, not a failed reproduction. Pass `--k 3`
to reproduce the report's tabulated values exactly:

```bash
python analysis/make_tier_figures.py --k 3   # -> 0.2854 / 0.6008
python analysis/make_tier_figures.py         # -> 0.2811 / 0.5958  (k=4 default)
```

The 1.5% spread between them is the blocking systematic, and it is smaller than
the 0.12 dex intrinsic scatter that is the actual result. Both are correct; only
quote one, and say which.

**Display clipping never moves a fitted number.** The figures drop collapsed
fits (ratios of ~1e-16) before ranging and drawing, because one of them would
inflate the padded upper axis bound from 1.14 to 8.3. The ML fits deliberately
keep them: their standard errors are enormous, so they carry almost no weight,
and cutting on the fitted value itself would bias the center. If you retune the
display floor, the printed common shape must not change — that is the check.

## Tests

```bash
python -m pytest tests/ -q          # or run each test_*.py's functions directly
```

`tests/test_slicing_model.py` is the one to look at first: it pins the fact that
`params_from_principal_axes` takes **radians**. Passing degrees is silent and
produced a full round of wrong numbers during this project. The test asserts the
exact face-on identity (b2/b1 = a3/a2 for any roll at theta = 0), the edge-on
bracket, that a degrees reading fails, and that the roll band collapses to zero
width when both rolls are frozen.

## Reading the results correctly

Four things will mislead you if you take the tables at face value. All are
argued in `PROJECT_REPORT.md`; they are repeated here because they are easy to
trip over.

1. **Read axis RATIOS, never absolute sizes.** The power-law profile carries an
   exact scale degeneracy (all axes by k, amplitude by k^alpha), so `a1` is not
   a measurement. At 200 px, 97% of subwindows fit `a1` larger than the window.
2. **The two ratio errors are anticorrelated** (median rho = -0.43), from the
   same fit. Per-window error ellipses are tilted, not upright crosses. The
   prolateness significance in the report adds them in quadrature as if
   independent — an open item; the raw jackknife samples needed to fix it are in
   `results/singleband_r0.1_s2_k4/`.
3. **Compare intrinsic to intrinsic.** The forward model predicts a noise-free
   spread. The observed raw sd of log10(b2/b1) is 0.234 dex; with measurement
   error removed it is 0.185 dex. `sensitivity_to_3d_scatter` takes values and
   SEs and deflates internally so this cannot be got wrong.
4. **The orientation result has an unresolved systematic.** The long axis
   prefers the echo plane — but W resolution comes from epoch spacing, not pixel
   scale, so unmodelled smearing along W would push in exactly that direction.
   `analysis/image_coherence.py` validates the orientation independently of the
   fit, but shares the same W sampling and so cannot separate this.
