#!/usr/bin/env python
"""Build results/scale_split_summary<variant>.csv from the two split modes.

One row per (split, quantity).  `split` is the lag-splitting mode:

    equal_count  <- results/scale_split_s<stride><variant>.csv        (median split)
    equal_dex    <- results/scale_split_logmid_s<stride><variant>.csv (log-mid split)

WHY THIS EXISTS.  This table is read by analysis/build_topline_notebook.py
(section 1.9) but had no producer anywhere in the tree: it was made once by an
ad-hoc step that was never committed, so it could not be regenerated -- and in
particular could not be built for a second preprocessing variant.  The
conventions below were recovered from the committed table and validated to
float precision against BOTH split modes; `--check` re-asserts that.

TWO DIFFERENT TESTS, deliberately.

*Shape quantities* (a2a1, a3a2, alpha) are tested as PAIRED differences: each
window contributes d = inner - outer, and the reported p is a two-sided Wilcoxon
signed-rank test of d against zero.  `n_positive` counts d > 0.  The null is
"the inner and outer lags see the same shape".

*Angles* (ang1, ang2) cannot be tested that way -- they are unsigned angles
between two axis directions, so they are non-negative by construction and a
signed-rank test against zero is meaningless.  Instead the null is ISOTROPY: if
the inner and outer long axes were independently random in 3D, the unsigned
angle between them would satisfy P(angle < T) = 1 - cos T.  We report a
one-sided binomial test on the count of windows agreeing to better than
AGREE_DEG, against that isotropic rate.  For these rows `n_positive` is that
count (not a count of positive differences) and the 68% interval is left blank,
because a bootstrap interval on the median angle is not what the test is about.
Reusing the `wilcoxon_p` / `n_positive` column names for the binomial test is
inherited from the committed table's schema, not a claim that the same test ran.

The 68% interval on the shape rows is a percentile bootstrap of the MEDIAN of d
(B=BOOT resamples, seed BOOT_SEED, fixed so the table is reproducible).

Usage:
    python analysis/summarize_scale_split_table.py [stride]
    python analysis/summarize_scale_split_table.py --check
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

import preprocessing_mode as pm  # noqa: E402

# Angle agreement threshold, degrees.  P(angle < T) = 1 - cos T under isotropy.
AGREE_DEG = 20.0
BOOT = 10000
BOOT_SEED = 0

SHAPE_QUANTITIES = ('a2a1', 'a3a2', 'alpha')
ANGLE_QUANTITIES = ('ang1', 'ang2')

# split label -> window-table basename stem (before stride/variant)
SPLITS = (('equal_count', 'scale_split'),
          ('equal_dex', 'scale_split_logmid'))


def window_table(stem, stride):
    v = pm.variant_suffix()
    if stem.endswith('_logmid'):
        path = f'{_ROOT}/results/{stem}_s{stride}{v}.csv'
    else:
        path = f'{_ROOT}/results/{stem}_s{stride}{v}.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(
            '%s missing -- build it first with:\n'
            '    python analysis/summarize_scale_split.py %d %s'
            % (os.path.relpath(path, _ROOT), stride,
               'logmid' if stem.endswith('_logmid') else 'median'))
    return pd.read_csv(path)


def shape_row(split, q, d):
    boot = np.median(
        np.random.default_rng(BOOT_SEED).choice(d, (BOOT, d.size)), axis=1)
    lo, hi = np.percentile(boot, [16, 84])
    return {
        'split': split,
        'quantity': q,
        'n': int(d.size),
        'median_inner_minus_outer': float(np.median(d)),
        'lo68': float(lo),
        'hi68': float(hi),
        'wilcoxon_p': float(stats.wilcoxon(d).pvalue),
        'n_positive': int((d > 0).sum()),
    }


def angle_row(split, q, a):
    """Isotropy test on an unsigned inter-axis angle.  See module docstring."""
    p_iso = 1.0 - np.cos(np.radians(AGREE_DEG))
    k = int((a < AGREE_DEG).sum())
    return {
        'split': split,
        'quantity': q,
        'n': int(a.size),
        'median_inner_minus_outer': float(np.median(a)),
        'lo68': np.nan,
        'hi68': np.nan,
        'wilcoxon_p': float(stats.binomtest(
            k, a.size, p_iso, alternative='greater').pvalue),
        'n_positive': k,
    }


def build(stride=2):
    rows = []
    for split, stem in SPLITS:
        w = window_table(stem, stride)
        for q in SHAPE_QUANTITIES:
            d = w['d_%s' % q].dropna().values
            r = shape_row(split, q, d)
            r['median_inner'] = float(np.median(w['%s_in' % q].dropna()))
            r['median_outer'] = float(np.median(w['%s_out' % q].dropna()))
            rows.append(r)
        for q in ANGLE_QUANTITIES:
            r = angle_row(split, q, w[q].dropna().values)
            r['median_inner'] = np.nan
            r['median_outer'] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    check = '--check' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    stride = int(args[0]) if args else 2

    got = build(stride)
    out = f'{_ROOT}/results/scale_split_summary{pm.variant_suffix()}.csv'

    if check:
        if not os.path.exists(out):
            print('  %s absent, nothing to check' % os.path.relpath(out, _ROOT))
            return 1
        want = pd.read_csv(out)
        m = want.merge(got, on=['split', 'quantity'], suffixes=('_w', '_g'))
        if len(m) != len(want):
            print('  ROW MISMATCH: %d common of %d committed' % (len(m), len(want)))
            return 1
        worst = ('', 0.0)
        for c in ('n', 'median_inner_minus_outer', 'lo68', 'hi68',
                  'wilcoxon_p', 'n_positive', 'median_inner', 'median_outer'):
            a, b = m[c + '_w'].values.astype(float), m[c + '_g'].values.astype(float)
            d = np.abs(a - b)
            d = np.where(np.isnan(a) & np.isnan(b), 0.0, d)
            mx = float(np.nanmax(d)) if d.size else 0.0
            if mx > worst[1]:
                worst = (c, mx)
            print('  %-26s max |diff| %.3e' % (c, mx))
        print('MATCH' if worst[1] < 1e-9 else 'DIFFERS (worst: %s)' % worst[0])
        return 0 if worst[1] < 1e-9 else 1

    got.to_csv(out, index=False)
    print('%d rows -> %s' % (len(got), os.path.relpath(out, _ROOT)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
