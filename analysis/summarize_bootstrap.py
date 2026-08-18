"""Summarise a block-bootstrap run into a per-window table.

Reads data/bs_k<k>_B<B>_s<stride>/*.json and writes
results/bootstrap_k<k>_B<B>_s<stride>.csv with, per window:

  central a2/a1, a3/a2, a1                 (the all-data fit)
  bootstrap median and 16/84 percentiles   (distribution-free interval)
  bootstrap sd and the 2x2 ratio covariance
  bias   = median(replicates) - central    (bootstrap bias estimate)
  n_ok   = replicates that fitted successfully

Percentile intervals are the point of doing this: the jackknife could only
give a normal-theory sigma, which is a poor description of a ratio bounded
in [0, 1] whose distribution is visibly skewed near the boundary.

Usage:  python analysis/summarize_bootstrap.py [run_dir]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def _ratios(rec):
    """Sorted-axis ratios from a fit record, or NaN if it failed."""
    if rec is None or 'error' in rec or not rec.get('fit_success', True):
        return np.nan, np.nan, np.nan
    a = np.sort([rec['a1'], rec['a2'], rec['a3']])[::-1]
    if not np.all(np.isfinite(a)) or a[0] <= 0 or a[1] <= 0:
        return np.nan, np.nan, np.nan
    return a[1] / a[0], a[2] / a[1], a[0]


def summarize(run_dir):
    rows = []
    for fn in sorted(glob.glob(f'{run_dir}/bs_*.json')):
        r = json.load(open(fn))
        c21, c32, c1 = _ratios(r.get('central'))

        s21, s32, s1 = [], [], []
        for s in r['samples']:
            x, y, z = _ratios(s)
            if np.isfinite(x) and np.isfinite(y):
                s21.append(x); s32.append(y); s1.append(z)
        s21 = np.array(s21); s32 = np.array(s32); s1 = np.array(s1)

        row = dict(chunk='r%d_c%d' % (r['row'], r['col']),
                   row=r['row'], col=r['col'], k=r['k'],
                   n_boot=r['n_boot'], n_ok=len(s21),
                   a2a1=c21, a3a2=c32, a1=c1)
        if len(s21) >= 10:
            row.update(
                a2a1_med=np.median(s21), a3a2_med=np.median(s32),
                a2a1_lo=np.percentile(s21, 16), a2a1_hi=np.percentile(s21, 84),
                a3a2_lo=np.percentile(s32, 16), a3a2_hi=np.percentile(s32, 84),
                a2a1_sd=s21.std(ddof=1), a3a2_sd=s32.std(ddof=1),
                a1_sd=s1.std(ddof=1),
                # bootstrap bias: does resampling systematically shift the ratio?
                a2a1_bias=np.median(s21) - c21,
                a3a2_bias=np.median(s32) - c32,
                cov_2121=np.cov(s21, s32)[0, 0],
                cov_3232=np.cov(s21, s32)[1, 1],
                cov_2132=np.cov(s21, s32)[0, 1],
                corr=np.corrcoef(s21, s32)[0, 1],
                # fraction of replicates within 0.05 of the sphere point (1,1):
                # a direct, assumption-free triaxiality statement
                f_near_sphere=float(np.mean((s21 > 0.95) & (s32 > 0.95))),
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    tag = os.path.basename(run_dir.rstrip('/')).replace('bs_', '')
    out = f'results/bootstrap_{tag}.csv'
    os.makedirs('results', exist_ok=True)
    df.to_csv(out, index=False)

    ok = df[df.n_ok >= 10] if 'n_ok' in df else df
    print(f'{len(df)} windows -> {out}')
    if len(ok):
        print('  replicates ok: median %d/%d' % (ok.n_ok.median(), ok.n_boot.iloc[0]))
        for v in ('a2a1', 'a3a2'):
            frac = (ok[f'{v}_hi'] - ok[f'{v}_lo']) / 2 / ok[v]
            print('  %s: central median %.3f | 68%% half-width / value: median %.0f%%'
                  % (v, ok[v].median(), 100 * frac.median()))
        print('  bias (median replicate - central): a2a1 %+.4f  a3a2 %+.4f'
              % (ok.a2a1_bias.median(), ok.a3a2_bias.median()))
        print('  ratio-ratio correlation: median %+.2f' % ok['corr'].median())
        print('  windows with any replicate near sphere (1,1): %d/%d'
              % ((ok.f_near_sphere > 0).sum(), len(ok)))
    return df


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else 'data/bs_k3_B100_s2'
    summarize(d)
