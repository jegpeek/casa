"""Build notebooks/topline_results.ipynb -- a guided walkthrough of report §1.

The notebook is generated rather than hand-edited so that it stays in step with
the analysis modules: every number in it is recomputed from the tracked result
tables by the same functions the paper's figures call.  Regenerate with

    python analysis/build_topline_notebook.py

Design rules, all deliberate:

* Tier A only.  Every cell runs from files tracked in git.  Nothing here needs
  the bulk input arrays (resampled_epochs_noclip.npy et al.), so a fresh clone
  can execute the whole notebook.  Steps that DO need the arrays are named in
  prose but never executed.
* k = 3 throughout, because that is the blocking the report tabulates.  See
  REPRODUCING.md; k = 4 is equally converged and gives values ~1.5 % different.
* Every claim recomputes.  No cell prints a remembered constant; where a value
  is compared against the report, the expected number is written in the same
  cell so a mismatch is visible rather than silent.
"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'notebooks', 'topline_results.ipynb')

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip('\n')))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip('\n')))


# ----------------------------------------------------------------- intro
md(r"""
# The Cas A light-echo topline results, recomputed

This notebook reproduces every headline claim in §1 of `PROJECT_REPORT.md` and
shows the code that produces it. It is a walkthrough, not a pipeline: each
section states the physical question, calls the same function the paper's
figures call, and prints the number next to the value the report asserts.

**What this notebook needs.** Only files tracked in git — the per-window fit
tables in `results/`. It does *not* need the bulk input arrays
(`resampled_epochs_noclip.npy` and friends, several GB), so it runs in a fresh
clone. The expensive upstream steps that *produce* those tables are described
where they belong but never run here; `REPRODUCING.md` has the commands.

**Why k = 3.** `--k` is the jackknife blocking. `k=3` and `k=4` are both
converged and agree to about 1.5 %; the report tabulates `k=3`, so this notebook
uses `k=3` everywhere. Change `K` in the setup cell to see the other one.

### The measurement in one paragraph

A light echo illuminates a thin sheet of interstellar dust that sweeps through
the cloud as time passes. Each JWST epoch is therefore a *slice* through the 3D
dust distribution, and the slices at different epochs are at different depths.
Transforming (RA, Dec, epoch) into a UVW frame centered on Cas A — with W normal
to the echo plane at the cloud "Mab" — turns the observation into a genuine 3D
scalar field. We measure the second-order structure function
$S_2(\Delta \mathbf{x})$ in small windows of that field and fit each with an
anisotropic ellipsoidal model, which returns three principal axis lengths
$a_1 \ge a_2 \ge a_3$ and an orientation.

The claims below are all about the *ratios* $a_2/a_1$ and $a_3/a_2$, and about
$b_2/b_1$ — the same ratio measured in the 2D plane of the echo, which is what
an observer without the third dimension would have.
""")

code(r"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

# Locate the repo root whether this runs from notebooks/ or the root.
ROOT = os.getcwd()
while not os.path.exists(os.path.join(ROOT, 'analysis', 'shape_center.py')):
    parent = os.path.dirname(ROOT)
    assert parent != ROOT, 'run this from inside the casa repo'
    ROOT = parent
for p in (ROOT, os.path.join(ROOT, 'analysis')):
    if p not in sys.path:
        sys.path.insert(0, p)

# This notebook's prose asserts specific arcsinh numbers, so it pins the
# preprocessing variant rather than following the repo default -- otherwise a
# reader with CASA_LINEAR_UNITS=1 exported gets tables that contradict the text.
# Must precede the mtf import: the variant is read at import time.
os.environ.pop('CASA_LINEAR_UNITS', None)
os.environ['CASA_ARCSINH_UNITS'] = '1'

import make_tier_figures as mtf
import shape_center as sc
import slicing_model as sm

K = 3                      # jackknife blocking; the report tabulates k=3
pd.set_option('display.width', 110)
print('repo :', ROOT)
print('k    :', K)
""")

# ----------------------------------------------------------------- data
md(r"""
## 0. The fit table, and which windows are usable

`mtf.load` reads the per-window fit table and joins it to the SNR audit, which
assigns each window to a tier by signal-to-noise quartile. Two helpers do the
gatekeeping, and it is worth being explicit about them because every number
downstream depends on the sample they define:

* `usable(d)` keeps rows whose fit converged and whose ratios and errors are
  finite and positive — the minimum for a quantity that will be logged.
* the `q4` tier is the top SNR quartile. The headline shape is quoted from `q4`
  alone, because that is where the errors are small enough for the intrinsic
  scatter to be separable from measurement noise.

One caution that matters for anyone extending this: the fit table's `b2b1`
column, on the `mode == '3d'` rows, is the slice ratio *predicted* by the 3D fit
— not an independent measurement. The measured 2D value lives on the
`mode == '2d'` rows. §1 uses both, for different claims, and confusing them
turns a real consistency check into a tautology.
""")

code(r"""
d, (q50, q75) = mtf.load(k=K)
d = d[mtf.usable(d)]
q4 = d[d.tier == 'q4']

print('windows in table      :', len(d))
print('top-SNR quartile (q4) :', len(q4))
print('SNR tier breaks       : median %.2f, upper quartile %.2f' % (q50, q75))
if len(d) == len(q4):
    print()
    print('NOTE: the k=%d table contains ONLY the top-SNR quartile, so d and q4' % K)
    print('      coincide here.  Every headline number is quoted from q4 anyway,')
    print('      but the three-tier FIGURES need the all-115 k=4 table -- see the')
    print('      figure section at the end.')
print()
print(q4[['chunk', 'a2a1', 'se_a2a1', 'a3a2', 'se_a3a2', 'incl', 'alpha', 'snr']]
      .head(6).to_string(index=False))
""")

# ----------------------------------------------------------------- 1.1
md(r"""
## 1.1 The headline: 2D appearance varies because of slicing, not shape

The central claim. Windows that look very different on the sky are consistent
with being the *same* 3D structure seen at different orientations.

The geometry that drives it: $b_2/b_1$ is measured in the echo plane, which cuts
the ellipsoid through its center. When the long axis $a_1$ lies in that plane,
the slice runs along the long axis and looks elongated (small $b_2/b_1$). When
$a_1$ points along the plane normal W, the slice is a rounder cross-section
(larger $b_2/b_1$). So $b_2/b_1$ must *anti*-correlate with the inclination of
$a_1$ to the plane, and the strength of that anti-correlation is a test of
whether slicing is really what is going on.

Two separate tests below. First the anti-correlation, on the **independently
measured** 2D fit — this is the one that could have failed. Then the same test
on the slice **predicted** by the 3D fit, which checks that the effect is
geometry rather than an artifact of the 2D fitting procedure.
""")

code(r"""
# The measured 2D fit is a separate row in the same table (mode == '2d').
tag = 'r%g_s%d_k%d' % (mtf.RCUT, mtf.STRIDE, K)
# Same variant as mtf.load above -- the 2D rows live in the same table, and
# mixing preprocessings between the 3D and 2D fits would be a silent error.
raw = pd.read_csv(os.path.join(ROOT, 'results', 'singleband_powerlaw_%s%s.csv'
                               % (tag, mtf.default_variant())))
d2 = raw[(raw['mode'] == '2d') & (raw.profile == 'powerlaw')]

j = q4[['chunk', 'row', 'col', 'incl', 'se_incl', 'b2b1']].merge(
        d2[['row', 'col', 'b2b1', 'se_b2b1']], on=['row', 'col'],
        suffixes=('_pred3d', '_meas2d'))

rho, p = stats.spearmanr(j.incl, j.b2b1_meas2d)
print('MEASURED 2D vs inclination : rho = %+.3f  p = %.1e  n = %d'
      % (rho, p, len(j)))
print('   report says             : rho = -0.69, p = 3e-05, n = 29')

# Three windows have se_incl > 45 deg: the DIRECTION of a1 is unconstrained
# there, so the inclination on the x-axis is meaningless for them.  This is a
# cut on an error, not on a fitted value.
keep = j.se_incl <= 45
rho2, p2 = stats.spearmanr(j[keep].incl, j[keep].b2b1_meas2d)
print()
print('dropping %d unconstrained-direction windows: rho = %+.3f  p = %.1e  n = %d'
      % ((~keep).sum(), rho2, p2, keep.sum()))
print('   report says                             : rho = -0.75')
""")

code(r"""
# The same test on the slice the 3D fit PREDICTS.  If this also anti-correlates,
# the trend is slicing geometry and not something the 2D fit invented.
rho3, p3 = stats.spearmanr(j.incl, j.b2b1_pred3d)
print('PREDICTED slice vs inclination : rho = %+.3f  p = %.1e' % (rho3, p3))
print('   report says                 : rho = -0.71')

agree = stats.spearmanr(j.b2b1_pred3d, j.b2b1_meas2d)
print()
print('predicted vs measured b2/b1    : rho = %+.3f  (independent fits agree)'
      % agree.statistic)
print('observed 2D range, single band : %.2f to %.2f'
      % (j.b2b1_meas2d.min(), j.b2b1_meas2d.max()))
print('   report quotes 0.06 to 0.88, which is the range across ALL lag bands;')
print('   the single r < 0.1 ly band used here is necessarily narrower.')
""")

md(r"""
The measured 2D range spans a factor of seven — as wide as the 3D axis ratios
themselves. That width is the thing being explained: it is not shape variation,
it is one shape seen from many angles.

The variance decomposition behind the report's "55–88 % of the variance in bands
0–3" is a per-lag-band calculation and needs the band-resolved 2D fits, so it is
not rerun here; this section reproduces the inclination trend, which is the part
that rests on the single-band table.
""")

# ----------------------------------------------------------------- 1.2
md(r"""
## 1.2 The 3D shape: strongly triaxial, prolate-leaning

Now the shape itself. Two estimators appear in this codebase and they are easy
to mix up, so both are called here and compared:

* `shape_center.ml_center_and_scatter(lv, sl)` takes **log** values and their
  log-space errors, and returns `(mu, sigma_int)`.
* `make_tier_figures.ml_center_and_scatter(v, se)` takes **linear** values and
  linear errors, and returns four items including the error on the center.

They implement the same likelihood — a common value plus an intrinsic scatter,
with the center profiled out — and agree to four decimals. The `mtf` one is used
by the figures because it also returns `se_mu`.

The physically meaningful combination is **prolateness**,
$\log_{10}(a_3/a_2) - \log_{10}(a_2/a_1)$: it is zero for a shape exactly on the
oblate/prolate divide, positive for prolate (cigar-like), negative for oblate
(pancake-like). It is log-symmetric by construction, which is why it is the
right statistic rather than the triaxiality parameter $T$.
""")

code(r"""
rows = []
for c, lab in [('a2a1', 'a2/a1'), ('a3a2', 'a3/a2')]:
    lv = np.log10(q4[c].values)
    sl = (q4['se_' + c] / q4[c] / np.log(10)).values      # dex
    mu_sc, sig_sc = sc.ml_center_and_scatter(lv, sl)
    mu_mtf, sig_mtf, se_mu, med_se = mtf.ml_center_and_scatter(
        q4[c].values, q4['se_' + c].values)
    rows.append(dict(ratio=lab, value=10 ** mu_sc, sig_int_dex=sig_sc,
                     se_mu_dex=se_mu, median_meas_se_dex=med_se,
                     mtf_agrees=np.isclose(mu_sc, mu_mtf, atol=1e-4)))
shape = pd.DataFrame(rows)
print(shape.to_string(index=False))
print()
# The report's tabulated values are from the ARCSINH run.  Label them, because
# the table above is built from whichever variant is the current default (raw
# flux), and an unlabelled "report:" line next to it reads as a failed
# reproduction rather than a different preprocessing.
print('report (arcsinh run): a2/a1 = 0.285 +- 0.039 dex,'
      '  a3/a2 = 0.601 +- 0.025 dex')
print('above is the %s variant'
      % ('raw-flux (_linear)' if mtf.default_variant() else 'arcsinh'))
""")

code(r"""
# Prolateness, and how far the shape sits from BOTH degenerate limits.
P = np.log10(q4.a3a2.values) - np.log10(q4.a2a1.values)
s21 = (q4.se_a2a1 / q4.a2a1 / np.log(10)).values
s32 = (q4.se_a3a2 / q4.a3a2 / np.log(10)).values
sP = np.sqrt(s21 ** 2 + s32 ** 2)                  # errors treated as independent

muP, sigP = sc.ml_center_and_scatter(P, sP)
w = 1.0 / (sP ** 2 + sigP ** 2)
seP = 1.0 / np.sqrt(w.sum())

a2a1, a3a2 = 10 ** np.log10(shape.value[0]), shape.value[1]
print('prolateness  = %+.4f +- %.4f dex  ->  %.1f sigma prolate'
      % (muP, seP, muP / seP))
print('report       : +0.350 +- 0.051 dex, 6.9 sigma')
print()
print('common shape : 1 : %.3f : %.3f' % (a2a1, a2a1 * a3a2))
print('on the prolate side : %d of %d windows'
      % ((P > 0).sum(), len(P)))
print('   at >1 sigma      : %d prolate, %d oblate'
      % (((P / sP) > 1).sum(), ((P / sP) < -1).sum()))
""")

md(r"""
Note what this does **not** say. The shape is many sigma from the divide, but it
is also far from either pure limit — the report puts it at 8.9σ from pure prolate
and 13.9σ from pure oblate. The defensible phrasing is *strongly triaxial,
prolate-leaning*. Not "prolate", and not "a cigar": a cigar has
$a_3/a_2 \approx 1$, and here $a_3/a_2 \approx 0.6$.
""")

# ----------------------------------------------------------------- 1.3
md(r"""
## 1.3 Triaxiality is distribution-free

Everything above assumes the errors behave. This section does not.

The block-bootstrap resamples $k\times k$ sub-regions within each window and
refits, giving replicate shapes whose spread needs no normal-theory assumption.
The question then becomes purely geometric: how close does *any* replicate, in
*any* window, ever come to the sphere point $(a_2/a_1, a_3/a_2) = (1, 1)$?

The original statistic here was `f_near_sphere`, the fraction of replicates
within 0.05 of the sphere. That fraction is zero in every window, which is a
weak way to state a strong result — "zero out of 2865" invites the question of
how far away they actually were. The minimum distance answers it directly.
""")

code(r"""
g = pd.read_csv(os.path.join(ROOT, 'results', 'bootstrap_grid_stats.csv'))
i = g.min_dist_sphere.idxmin()
print('windows bootstrapped   : %d' % len(g))
print('converged replicates   : %d' % g.n_ok.sum())
print('f_near_sphere (thr 0.05) all zero : %s' % bool((g.min_dist_sphere > 0.05).all()))
print()
print('nearest ANY replicate comes to (1,1) : %.3f   in window %s'
      % (g.min_dist_sphere.min(), g.loc[i, 'chunk']))
print('report                               : 0.443, r2800_c2400')
""")

# ----------------------------------------------------------------- 1.4
md(r"""
## 1.4 A universal small-lag slope

The fit also returns $\alpha$, the log–log slope of $S_2$ against ellipsoidal
radius. This is a property of the turbulence, not of the shape, and the question
is whether it varies between windows.

The test is an inverse-variance weighted common value plus a $\chi^2$: if one
$\alpha$ fits all windows, $\chi^2$ should be near its dof.

**A blocking caveat specific to this section.** Unlike §1.2 and §1.5, the report's
$\alpha$ numbers are tabulated at the *unsuffixed* `k=2` blocking, not `k=3` —
identifiable because it quotes a median jackknife SE of 0.058, which is the `k=2`
value (`k=3` gives 0.054). The median $\alpha$ is identical either way; only the
error-weighted quantities move, and the $\chi^2$ moves a lot because it is a pure
function of the errors. Both are printed. This is exactly the trap
`REPRODUCING.md` warns about, and it does not affect the conclusion: no blocking
gives any evidence that $\alpha$ varies.
""")

code(r"""
def alpha_stats(path):
    t = pd.read_csv(path)
    t = t[(t['mode'] == '3d') & (t.profile == 'powerlaw')]
    t = t[t.fit_success.astype(bool) & np.isfinite(t.alpha)
          & np.isfinite(t.se_alpha)]
    a, sa = t.alpha.values, t.se_alpha.values
    w = 1.0 / sa ** 2
    ivw = (w * a).sum() / w.sum()
    return dict(n=len(t), median=np.median(a), ivw=ivw,
                se_ivw=1 / np.sqrt(w.sum()), median_SE=np.median(sa),
                lo=a.min(), hi=a.max(),
                chi2=(w * (a - ivw) ** 2).sum(), dof=len(t) - 1)

for lab, f in [('k=2 (report §1.4)', 'singleband_powerlaw_r0.1_s2.csv'),
               ('k=3 (this notebook)', 'singleband_powerlaw_r0.1_s2_k3.csv')]:
    s = alpha_stats(os.path.join(ROOT, 'results', f))
    print('%-21s n=%d  median=%.3f  ivw=%.3f +- %.3f  median SE=%.3f'
          % (lab, s['n'], s['median'], s['ivw'], s['se_ivw'], s['median_SE']))
    print('%-21s range %.3f-%.3f   common-alpha chi2 = %.1f / %d'
          % ('', s['lo'], s['hi'], s['chi2'], s['dof']))

print()
print('report : median 0.591, ivw 0.566 +- 0.004, range 0.413-0.734,')
print('         median SE 0.058, chi2 90.7/28 (p=0.995 after inflation)')
""")

md(r"""
$\alpha \approx 0.57$ sits below the Kolmogorov 2/3. That comparison is not
direct — this traces *density*, not velocity — but a referee will ask, so the
paper should raise it first.
""")

# ----------------------------------------------------------------- 1.5
md(r"""
## 1.5 The shapes are not *identical* — and by how much

This is where the honest headline lives. A single common shape is not the right
model: there is real, measurable window-to-window variation on top of the
measurement noise.

`profile_ci` separates the two. It profiles the likelihood over the intrinsic
scatter $\sigma_{\rm int}$, returns a 1σ interval on it, and gives
$\Delta\chi^2$ against $\sigma_{\rm int} = 0$ — the test that the variation is
real rather than noise.
""")

code(r"""
rows = []
for c, lab in [('a2a1', 'a2/a1'), ('a3a2', 'a3/a2')]:
    lv = np.log10(q4[c].values)
    sl = (q4['se_' + c] / q4[c] / np.log(10)).values
    sig, lo, hi, dchi2 = sc.profile_ci(lv, sl)
    tot = np.std(lv, ddof=1)
    rows.append(dict(ratio=lab, sig_int_dex=sig, lo68=lo, hi68=hi,
                     pct=100 * (10 ** sig - 1),
                     var_frac=sig ** 2 / tot ** 2,
                     dchi2_vs_zero=dchi2, p=stats.chi2.sf(dchi2, 1)))
print(pd.DataFrame(rows).to_string(index=False))
print()
print('report (arcsinh run): a2/a1 0.120 dex [0.084, 0.161] p=0.006 ;'
      ' a3/a2 0.096 dex [0.076, 0.119] p=6e-06')
print('above is the %s variant'
      % ('raw-flux (_linear)' if mtf.default_variant() else 'arcsinh'))
""")

md(r"""
So the claim is *not* "the shapes are identical". It is: **the shapes differ by a
measured 25–32 %, and most of the apparent window-to-window spread is
measurement noise.** Both halves matter — the first is a detection, the second is
what makes the 2D result in §1.1 make sense.
""")

# ----------------------------------------------------------------- 1.6
md(r"""
## 1.6 What 2D could never have seen — the honest limitation

§1.1 showed the 2D spread is consistent with pure slicing. §1.5 showed there
*is* real 3D shape variation. Both are true, and the reason is that the 2D data
have almost no power to detect variation of the size actually present.

`sensitivity_to_3d_scatter` makes that quantitative: it pushes an assumed 3D
$\sigma_{\rm int}$ through the slicing forward model, predicts the 2D spread it
would produce, and compares against the observed 2D spread. The output is the
$\sigma_{\rm int}$ at which 2D would have noticed.
""")

code(r"""
sens = pd.read_csv(os.path.join(ROOT, 'results',
                                'twod_sensitivity_to_3d_scatter.csv'))
near2 = sens.iloc[(sens.tension_sigma - 2.0).abs().argsort()[:1]]
print(sens[['sig_int_a2a1_dex', 'pred_2D_sd_dex', 'tension_sigma']]
      .to_string(index=False))
print()
print('2-sigma tension reached at sig_int = %.2f dex'
      % near2.sig_int_a2a1_dex.iloc[0])
print('measured 3D value          = 0.120 dex')
print('   -> a factor %.1f below the 2D detection threshold'
      % (near2.sig_int_a2a1_dex.iloc[0] / 0.120))
""")

# ----------------------------------------------------------------- 1.8
md(r"""
## 1.8 Structures flatten with scale — carried by the short axis

Refitting each window in five overlapping 0.6-dex lag bands turns each shape
into a shape-versus-scale curve. The per-window slopes are aggregated by
**median** with a Wilcoxon signed-rank test, not by inverse-variance weighting —
the slopes have heavy tails and one badly-determined window would dominate a
weighted mean. This matches `results/scale_profile_slopes_summary.csv`.

The phrasing here needs care. "Prolate → triaxial with scale" wrongly implies the
structures become *less elongated*. The flattening is the robust part and it is
carried by $a_3$: $a_3/a_2$ falls −0.28/dex at $p = 4\times10^{-9}$.

$a_2/a_1$ is the part to state carefully. Its slope over the full lag range is
null ($-0.10$/dex, $p = 0.26$), but that is not the same as scale-invariance:
it falls across the inner three bands and then the windows diverge, with 11 of
22 rising and 11 falling beyond 0.08 ly. Drop the widest band alone and the
slope is $-0.17$/dex at $p = 9\times10^{-4}$. The honest summary is that the
data do not settle whether the elongation changes with scale.

These numbers are the canonical power-law band fits. The Weibull was used
historically and gives $a_3/a_2 = -0.24$/dex; see §1.8b for why its shape
parameter is not identified inside a single 0.6-dex band.
""")

code(r"""
sl_ = pd.read_csv(os.path.join(ROOT, 'results',
                               'scale_profile_d0.6_s2_slopes.csv'))
rows = []
for c, lab in [('a3a2', 'a3/a2'), ('a3a1', 'a3/a1'), ('a2a1', 'a2/a1')]:
    v = sl_['slope_' + c].values
    v = v[np.isfinite(v)]
    rows.append(dict(ratio=lab, median_slope_per_dex=np.median(v),
                     n=len(v), n_negative=(v < 0).sum(),
                     wilcoxon_p=stats.wilcoxon(v).pvalue))
print(pd.DataFrame(rows).to_string(index=False))
print()
print()
print('Canonical (power-law) band fits. The historical Weibull run gave'
      ' a3/a2 -0.24/dex p=7e-08.')
""")

code(r"""
# Net displacement across the full lag range, innermost to outermost band.
#
# Subset matters here and the report is specific about it: this statistic uses
# only windows where ALL FIVE bands converged (n_degenerate == 0, 23 of 29).
# A window with a degenerate band has no trustworthy endpoint, and since this
# quantity is defined by its two endpoints, including it would be worse than
# for the slope, which at least fits over whatever bands survived.  Both
# subsets are printed so the effect of the choice is visible rather than
# hidden in a filter.
for c, lab in [('a3a2', 'a3/a2'), ('a2a1', 'a2/a1')]:
    f, l = sl_[c + '_first'].values, sl_[c + '_last'].values
    finite = np.isfinite(f) & np.isfinite(l) & (f > 0) & (l > 0)
    for sub, tag in [(finite & (sl_.n_degenerate.values == 0), 'all 5 bands ok'),
                     (finite, 'any converged  ')]:
        dd = np.log10(l[sub]) - np.log10(f[sub])
        print('dlog10(%-6s) [%s] median = %+.3f  %2d/%2d neg  p = %.1e'
              % (lab, tag, np.median(dd), (dd < 0).sum(), sub.sum(),
                 stats.wilcoxon(dd).pvalue))
print()
print('Canonical (power-law) fits: dlog10(a3/a2) = -0.331, 22/22 neg, p=5e-07')
print('                      vs    dlog10(a2/a1) = -0.107, 12/22 neg, p=0.12')
print('Historical weibull run gave -0.313 (22/23, p=5e-07) and +0.013 (p=0.71).')
""")

# ----------------------------------------------------------------- 1.8b
md(r"""
## 1.8b Why the band fits use a power law, not the Weibull

The band fits above use a simple power law inside each 0.6-dex band. That is a
deliberate change from the historical run, and the reason is identifiability
rather than fit quality.

The Weibull profile, $S_2 = \mathrm{var}_\infty\,(1-e^{-r^\beta})^{\alpha/\beta}$,
exists to describe the turnover from the power-law regime to saturation. Its
turnover sits at lag $\approx a_1$. But the fitted $a_1$ *exceeds the outer
radius of the band it was measured in* for 99.3% of band fits — so within one
band the turnover is never sampled, and $\beta$ has nothing to constrain it. It
pins to a bound in 63% of bands. Where it pins upward the Weibull is
numerically the power law it reduces to; where it floats, the extra freedom is
fitting noise. The power law is the honest parameter count for a narrow band.

Two cautions. First, a bare power law $A r^\alpha$ is invariant under
$r \to kr$ — an exact flat direction against the ellipsoid size — so it must be
fit with the amplitude frozen; `scale_split.BAND_PROFILES` pairs each profile
with the freeze its form requires so a caller cannot forget. Second, do not use
an information criterion to choose here: the two forms differ by 0.27% in rms
over ~73k highly correlated residuals per band, so AIC "prefers" the Weibull by
an enormous margin on a difference carrying almost no independent information.

The choice does not threaten the headline — $a_3/a_2$ falls under both, and
slightly more steeply under the power law — but it does move $a_2/a_1$ band by
band, which is why §1.8 states that result with the range dependence attached.
`results/profile_comparison.png` shows both.
""")

code(r"""
_pl = pd.read_csv(os.path.join(ROOT, 'results',
                               'scale_profile_slopes_summary.csv'))
_wb = pd.read_csv(os.path.join(ROOT, 'results',
                               'scale_profile_weibull_slopes_summary.csv'))
_m = _wb.merge(_pl, on=['measure', 'subset'], suffixes=('_weib', '_plaw'))
_m = _m[_m.measure.isin(['a3/a2', 'a3/a1', 'a2/a1'])]
print('%-8s %-18s %10s %10s %11s %11s'
      % ('measure', 'subset', 'weibull', 'powerlaw', 'p weib', 'p plaw'))
for _, r in _m.iterrows():
    print('%-8s %-18s %+10.3f %+10.3f %11.1e %11.1e'
          % (r['measure'], r['subset'], r['slope_per_dex_weib'],
             r['slope_per_dex_plaw'], r['wilcoxon_p_weib'],
             r['wilcoxon_p_plaw']))
""")

# ----------------------------------------------------------------- 1.9
md(r"""
## 1.9 Orientation is scale-invariant, and prefers the echo plane

Two independent statements, both from `results/scale_split_s2.csv`, which
splits each window's lags into inner and outer halves and refits the two
independently.

*Scale invariance*: the long axis found from the inner lags agrees with the one
from the outer lags to a few degrees, against 60° for random unsigned axes in 3D.
The structure has one orientation, not a scale-dependent one.

*In-plane preference*: the long axis is not isotropically distributed — it prefers
to lie in the echo plane. This used to carry a mandatory caveat, on the reasoning
that W is sampled ~14× more coarsely than the in-plane axes (epoch spacing rather
than pixel scale), so unmodelled smearing along W could inflate apparent in-plane
extent and manufacture the preference. **That reasoning was backwards.** Smearing
along W is a convolution along W: it adds correlation length there and *lengthens*
the recovered ellipsoid along W, rotating the long axis toward W and away from the
plane. `analysis/w_smear_injection.py` measures it — a true 45° tilt is recovered
at 34.5° under the most severe smearing tested — so the systematic works against
the in-plane preference and the measured angle is a conservative floor
(report §2.7).

The same injection also answers the geometric half of the selection worry: with
the real coverage pattern and real lag grid but no smearing, every injected tilt
comes back to better than 0.3°, so the window's own lag geometry does not
manufacture an in-plane preference. What that test does *not* cover is
window-level selection — which windows are bright enough to fit at all — and it
uses one window's coverage, so the magnitudes above should be checked on a few
more before they are quoted.
""")

code(r"""
summ = pd.read_csv(os.path.join(ROOT, 'results', 'scale_split_summary.csv'))
ang = summ[summ.quantity.isin(['ang1', 'ang2'])][
    ['split', 'quantity', 'n', 'median_inner_minus_outer', 'wilcoxon_p',
     'n_positive']]
print(ang.to_string(index=False))
print()
print('report: long axis agrees to 7.3 deg (equal-dex) / 14 deg (equal-count),'
      ' vs 60 deg random')
""")

# ----------------------------------------------------------------- 1.11-1.12
md(r"""
## 1.11–1.12 What may be quoted, and what the ratios actually are

Two governance results that constrain the paper's wording.

**Ratios are robust; absolute sizes are not.** Under the power-law profile the
absolute scale is *formally unidentifiable* — there is an exact degeneracy — so
$a_1$ must never be quoted from these fits. The ratios are ~12× more stable than
$a_1$ against the $\beta$ nuisance parameter. Since the 3D-similarity argument
rests entirely on ratios, this does not weaken it.

**And the ratios are lag ratios, not amplitudes.** The fit's ellipsoidal radius
$r = |M^{-1}\Delta x|$ equals $s/a_i$ exactly along principal axis $i$. With
$S_2 = A r^\alpha$, the $a_i$ are iso-$S_2$ contour lengths — so $a_2/a_1$ and
$a_3/a_2$ are **ratios of lag lengths at fixed $S_2$**, and are $\alpha$-free.
They are *not* amplitude contrasts. The alternative (an $S_2$ ratio at fixed lag)
is a genuinely different statistic that reorders 13 % of window pairs.
""")

code(r"""
print(pd.read_csv(os.path.join(ROOT, 'results',
                               'shape_parameter_definitions.csv'))
      .to_string(index=False))
""")

# ----------------------------------------------------------------- open item
md(r"""
## An open item: the two ratio errors are not independent

Every $\sigma$ above treats $\delta(a_2/a_1)$ and $\delta(a_3/a_2)$ as
independent when combining them into prolateness. They are not: both come from
one fit to one window, and $a_2$ appears in the numerator of one ratio and the
denominator of the other, so an error in $a_2$ moves them in opposite directions.

The block-bootstrap already measured the correlation per window
(`corr` in `bootstrap_k3_B100_s2.csv`), so the size of the effect can be checked
directly rather than argued about. It is worth doing before submission because
prolateness carries the central shape claim.
""")

code(r"""
bs = pd.read_csv(os.path.join(ROOT, 'results', 'bootstrap_k3_B100_s2.csv'))
m = q4.merge(bs[['chunk', 'corr']], on='chunk', how='inner')

P = np.log10(m.a3a2.values) - np.log10(m.a2a1.values)
s21 = (m.se_a2a1 / m.a2a1 / np.log(10)).values
s32 = (m.se_a3a2 / m.a3a2 / np.log(10)).values
rho_w = m['corr'].values

for lab, var in [
        ('independent (as published)', s21 ** 2 + s32 ** 2),
        ('with measured correlation', s21 ** 2 + s32 ** 2 - 2 * rho_w * s21 * s32)]:
    sP = np.sqrt(var)
    mu, sig = sc.ml_center_and_scatter(P, sP)
    se = 1.0 / np.sqrt((1.0 / (sP ** 2 + sig ** 2)).sum())
    print('%-27s  prolateness = %+.4f +- %.4f dex  ->  %.1f sigma'
          % (lab, mu, se, mu / se))

print()
print('per-window correlation: median %+.3f, %d of %d negative'
      % (np.median(rho_w), (rho_w < 0).sum(), len(rho_w)))
""")

md(r"""
The correction moves the significance by well under a sigma, so the conclusion
holds — but the paper should quote the corrected number, since the correlation
was measured rather than assumed. Note also that the per-window correlations
scatter in *both* directions, which is why a hand-waved "the errors are
correlated so the significance is overstated" would have been wrong: the sign
varies window to window and only the data settle it.
""")

# ----------------------------------------------------------------- figures
md(r"""
## The two deliverable figures

`make_tier_figures.main` regenerates both published figures. One thing to get
right before running it: **the figures are k = 4, while the numbers above are
k = 3.** That is not an inconsistency, it is a property of what each table
contains.

The `k=3` table holds only the 29 top-SNR windows — enough for every headline
number, since all of those are quoted from `q4` anyway. The `k=4` table covers
all 115 windows and so is the only one that can populate a three-tier figure;
drawing the shape plane at `k=3` would render a single tier and silently drop the
lower two, while still carrying the `all115` filename. So the figures below are
generated at `k=4`, and the small shift in the printed common shape (0.2811 vs
0.2854) is the blocking systematic documented in `REPRODUCING.md`.

Recall also the standing rule `main` implements: the display clip (dropping
collapsed fits with ratios $\sim 10^{-16}$) affects only what is **drawn**. The
maximum-likelihood fits always use the full usable sample, so no printed number
moves when a display threshold is retuned.
""")

code(r"""
import matplotlib
matplotlib.use('Agg')
out = os.path.join(ROOT, 'notebooks', 'figures')

# k=4 deliberately -- see above.  n_drawn should be 100 of 115; at K=3 it would
# be 29, which is the diagnostic that the tiers had collapsed to one.
mtf.main(k=4, outdir=out)
print()
print('written to', out)
""")

code(r"""
from IPython.display import Image, display
for f in ('shape_plane_all115.png', 'b2b1_vs_inclination_all115.png'):
    display(Image(filename=os.path.join(out, f), width=620))
""")

md(r"""
## Where the numbers came from, and what is not here

Everything above ran from tracked per-window fit tables. Producing *those* needs
the bulk input arrays and hours of compute, and is out of scope for this
notebook — see `REPRODUCING.md` for the commands. In brief:

| stage | script | needs bulk arrays |
|---|---|---|
| SNR audit and tiering | `analysis/noise_audit.py` | yes |
| per-window single-band fits | `analysis/singleband_powerlaw.py` | yes |
| lag-band shape profiles | `analysis/scale_profile.py` | yes |
| inner/outer split refits | `analysis/scale_split.py` | yes |
| block bootstrap | `analysis/bootstrap_windows.py` | yes |
| W-smearing injection test | `analysis/w_smear_injection.py` | yes |
| **everything in this notebook** | — | **no** |

Report sections not recomputed here, and why:

* **§1.7** is approved prose, not a computation.
* **§1.10** (orientation validated by image coherence, ρ = +0.65) needs the
  first-epoch images — `analysis/image_coherence.py`. It is the strongest
  independent check on orientation and belongs in any tier-B rerun.
* **§1.3**'s replicate distributions are summarized from the tracked bootstrap
  table; the replicates themselves are not tracked.
* **§2.7** (the W-smearing systematic, which §1.9 above depends on) needs a real
  window's lag grid — `analysis/w_smear_injection.py`, ~3 min. Its conclusion is
  quoted in §1.9 rather than recomputed here.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                   'name': 'python3'},
    'language_info': {'name': 'python'},
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
nbf.write(nb, OUT)
print('wrote %s (%d cells)' % (OUT, len(cells)))
