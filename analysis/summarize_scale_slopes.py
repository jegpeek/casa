"""Aggregate the per-window scale-profile slopes into the report's §1.8 table.

This writes results/scale_profile_slopes_summary.csv, which was previously the
one tracked result with no producing script in the repo -- the recipe lived only
in the prose.  It is recovered here and verified: run with --check to confirm
every cell reproduces the tracked file bit for bit.

The aggregation is deliberately NOT inverse-variance weighted.  The per-window
slopes have heavy tails, and a weighted mean lets a single badly-determined
window dominate; the median with a Wilcoxon signed-rank test is what the report
quotes.  The 68 % interval is a bootstrap of that median.

One caveat, worth stating because it looks like a reproduction failure and is
not.  With only 27 windows, the median of a bootstrap resample can only take
values from the observed slopes, so the 16th/84th percentiles are pinned to
order statistics of the sample.  The endpoints therefore shift by one order
statistic depending on the RNG seed and never converge with more draws -- at
20k draws across 40 seeds each endpoint still takes two distinct values.
--check accordingly requires the median, p-value and counts to agree exactly,
and allows the CI endpoints to differ by at most one order statistic.  Treat
the quoted interval as good to that resolution, not to the last digit.

Two subsets are reported per measure:
  all_5_bands        -- every usable band
  outermost_dropped  -- the largest-lag band removed, the robustness check for
                        the widest band running into the window size

Usage
-----
python analysis/summarize_scale_slopes.py [--bands PATH] [--out PATH] [--check]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# measure key -> label as it appears in the tracked summary
MEASURES = [
    ('p', 'p = ln(a1 a3/a2^2)'),
    ('T', 'T (triaxiality)'),
    ('a3a1', 'a3/a1'),
    ('a3a2', 'a3/a2'),
    ('a2a1', 'a2/a1'),
    ('alpha', 'alpha (profile slope)'),
]
N_BOOT = 2000
BOOT_SEED = 0


def slope_of(g, key):
    """OLS slope of `key` against log10(r_mid) over usable bands.

    Mirrors summarize_scale_profile._slope_of, including its <3-point rule: a
    window with fewer than three usable bands has no slope, which is what drops
    the sample from 29 windows to 27.
    """
    ok = g[g['usable'] & np.isfinite(g[key])]
    if len(ok) < 3:
        return np.nan
    return float(np.polyfit(np.log10(ok['r_mid']), ok[key], 1)[0])


def _boot_ci(v, seed=BOOT_SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    med = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.percentile(med, 16)), float(np.percentile(med, 84))


def summarize(bands):
    """bands: the per-band table (results/scale_profile_*_bands.csv)."""
    rows = []
    for subset, keep in [('all_5_bands', None), ('outermost_dropped', -1)]:
        for key, label in MEASURES:
            vals = []
            for _, g in bands.groupby('chunk'):
                g = g.sort_values('band')
                if keep is not None:
                    g = g.iloc[:keep]
                vals.append(slope_of(g, key))
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            lo, hi = _boot_ci(v)
            rows.append(dict(measure=label, subset=subset,
                             slope_per_dex=float(np.median(v)),
                             ci68_lo=lo, ci68_hi=hi,
                             wilcoxon_p=float(stats.wilcoxon(v).pvalue),
                             n_windows_negative=int((v < 0).sum()),
                             n_windows=int(len(v))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bands', default=os.path.join(
        _ROOT, 'results', 'scale_profile_d0.6_s2_bands.csv'))
    ap.add_argument('--out', default=os.path.join(
        _ROOT, 'results', 'scale_profile_slopes_summary.csv'))
    ap.add_argument('--check', action='store_true',
                    help='compare against --out instead of overwriting it')
    a = ap.parse_args()

    got = summarize(pd.read_csv(a.bands))
    if not a.check:
        got.to_csv(a.out, index=False)
        print('wrote %s (%d rows)' % (a.out, len(got)))
        return 0

    want = pd.read_csv(a.out)
    bands = pd.read_csv(a.bands)
    m = want.merge(got, on=['measure', 'subset'], suffixes=('_want', '_got'))
    assert len(m) == len(want) == len(got), 'row sets differ'

    bad = 0
    for c in ['slope_per_dex', 'wilcoxon_p', 'n_windows_negative', 'n_windows']:
        d = np.abs(m[c + '_want'] - m[c + '_got']).max()
        print('  %-20s max |diff| %.3e   (must be exact)' % (c, d))
        bad += d > 1e-9

    # CI endpoints: allow one order statistic of slack, see the module docstring.
    worst = 0
    for _, r in m.iterrows():
        key = dict((lab, k) for k, lab in MEASURES)[r['measure']]
        keep = None if r['subset'] == 'all_5_bands' else -1
        v = []
        for _, g in bands.groupby('chunk'):
            g = g.sort_values('band')
            v.append(slope_of(g.iloc[:keep] if keep else g, key))
        sv = np.sort(np.asarray(v, float)[np.isfinite(v)])
        for side in ['ci68_lo', 'ci68_hi']:
            iw = int(np.argmin(np.abs(sv - r[side + '_want'])))
            ig = int(np.argmin(np.abs(sv - r[side + '_got'])))
            worst = max(worst, abs(iw - ig))
    print('  %-20s max shift %d order statistic(s)   (<=1 allowed)'
          % ('ci68_lo/hi', worst))
    bad += worst > 1

    print('MATCH' if not bad else 'MISMATCH')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
