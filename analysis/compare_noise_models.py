"""Compare weibull_log_s2 against weibull_noise_log_s2 on the same windows.

Reads the per-window JSONs written by jackknife_noise.py (data/jk_noise/) and
jackknife_q4.py (data/jk_q4/, run 1's 3-parameter fits) and tests the four
predictions of the noise-pedestal diagnosis:

  1. beta comes off its bounds
  2. the alpha-SNR correlation (+0.97 in run 1) collapses
  3. fits whose a1 exceeds the window extent become rarer
  4. the jackknife error ellipses shrink

Writes results/noise_model_comparison.csv and prints the summary table.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

WINDOW_LY = 0.320 * (400 / 200)   # 400px window extent in light years


def _ok(rec):
    return rec is not None and 'error' not in rec


def jk_stats(samples, fn):
    """Delete-one-block jackknife: SE^2 = (N-1)/N * sum (x_i - xbar)^2."""
    vals = np.array([fn(s) for s in samples], float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n < 2:
        return np.nan, np.nan
    mean = vals.mean()
    se = np.sqrt((n - 1) / n * np.sum((vals - mean) ** 2))
    return mean, se


def load_dir(d, tag):
    rows = []
    for fn in sorted(glob.glob(f'{d}/*.json')):
        r = json.load(open(fn))
        good = [s for s in r['samples'] if _ok(s)]
        if not good:
            continue
        c = r.get('central') if _ok(r.get('central')) else None
        row = dict(model=tag, row=r['row'], col=r['col'], n_ok=len(good),
                   wall_s=r.get('wall_s', np.nan))
        for k in ('alpha', 'beta', 'var_inf', 's2_noise', 'a1', 'a2', 'a3'):
            row[f'{k}_c'] = c.get(k, np.nan) if c else np.nan
        r21, r21e = jk_stats(good, lambda s: s['a2'] / s['a1'])
        r32, r32e = jk_stats(good, lambda s: s['a3'] / s['a2'])
        a1m, a1e = jk_stats(good, lambda s: s['a1'])
        bm, _ = jk_stats(good, lambda s: s['beta'])
        am, _ = jk_stats(good, lambda s: s['alpha'])
        row.update(r21_jk=r21, r21_se=r21e, r32_jk=r32, r32_se=r32e,
                   a1_jk=a1m, a1_se=a1e, beta_jk=bm, alpha_jk=am)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    frames = []
    for d, tag in [('data/jk_q4', 'weibull'), ('data/jk_noise', 'noise')]:
        if os.path.isdir(d):
            frames.append(load_dir(d, tag))
    df = pd.concat(frames, ignore_index=True)

    snr = None
    for cand in ('results/noise_audit_table.csv', 'noise_audit_merged.csv'):
        if os.path.exists(cand):
            snr = pd.read_csv(cand)
            break
    if snr is not None:
        keys = [c for c in ('row', 'col') if c in snr.columns]
        scol = next((c for c in snr.columns
                     if 'snr' in c.lower()), None)
        if keys and scol:
            df = df.merge(snr[keys + [scol]].rename(columns={scol: 'snr'}),
                          on=keys, how='left')

    print(f'\nloaded: ' + ', '.join(
        f'{t} n={len(g)}' for t, g in df.groupby("model")))
    print('\n%-8s %7s %7s %7s %8s %8s %8s' % (
        'model', 'beta_lo', 'beta_hi', 'a1>win', 'med_r21', 'med_se21',
        'r_aSNR'))
    for tag, g in df.groupby('model'):
        b = g.beta_c
        lo = int((b <= 1.001).sum())
        hi = int((b >= 9.99).sum())
        over = int((g.a1_jk > WINDOW_LY).sum())
        rr = np.nan
        if 'snr' in g and g.snr.notna().sum() > 3:
            m = g.alpha_c.notna() & g.snr.notna()
            rr = np.corrcoef(g.alpha_c[m], g.snr[m])[0, 1]
        print('%-8s %4d/%-3d %4d/%-3d %4d/%-3d %8.3f %8.3f %8.3f' % (
            tag, lo, len(g), hi, len(g), over, len(g),
            g.r21_jk.median(), g.r21_se.median(), rr))

    if 's2_noise_c' in df:
        gn = df[df.model == 'noise']
        at_floor = int((gn.s2_noise_c <= 1.01e-8).sum())
        print(f'\ns2_noise at lower bound (term switched off): '
              f'{at_floor}/{len(gn)}')
        print('s2_noise median = %.4g  (range %.2g - %.2g)' % (
            gn.s2_noise_c.median(), gn.s2_noise_c.min(), gn.s2_noise_c.max()))

    os.makedirs('results', exist_ok=True)
    df.to_csv('results/noise_model_comparison.csv', index=False)
    print(f'\nwrote results/noise_model_comparison.csv  ({len(df)} rows)')
    return df


if __name__ == '__main__':
    main()
