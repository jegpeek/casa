"""Summarize the full 115-window jackknife run into a per-window table.

Reads data/jk_<profile>[_s<stride>]/*.json and writes a tidy CSV with the
central-fit parameters, the jackknife standard errors, and quality flags.

The quality cut is RE-DERIVED here rather than inherited: the var_inf < 0.25
screen was found on the 4-parameter fits, where the failure mode was the noise
term absorbing the signal.  The 3-parameter model has no noise term, so that
mechanism cannot operate and the threshold must be justified again (or
discarded) on this sample.

Usage:  python analysis/summarize_full_sample.py <dir> <out.csv>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def jackknife_se(values):
    """Delete-one-block jackknife standard error."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return np.nan
    return float(np.sqrt((v.size - 1) / v.size * np.sum((v - v.mean()) ** 2)))


def load_dir(d):
    rows = []
    for fn in sorted(glob.glob(os.path.join(d, '*.json'))):
        r = json.load(open(fn))
        c = r.get('central') or {}
        if 'error' in c or not c:
            rows.append(dict(chunk=f"r{r['row']}_c{r['col']}", row=r['row'],
                             col=r['col'], central_ok=False))
            continue
        good = [s for s in r['samples'] if 'error' not in s]
        a1 = c['a1']
        rows.append(dict(
            chunk=f"r{r['row']}_c{r['col']}", row=r['row'], col=r['col'],
            central_ok=True, converged=bool(c.get('fit_success', True)),
            fit_stride=r.get('fit_stride', 1),
            a1=a1, a2a1=c['a2'] / a1, a3a2=c['a3'] / c['a2'],
            alpha=c['alpha'], beta=c['beta'], var_inf=c['var_inf'],
            rms=c.get('rms_resid', np.nan),
            se_a2a1=jackknife_se([s['a2'] / s['a1'] for s in good]),
            se_a3a2=jackknife_se([s['a3'] / s['a2'] for s in good]),
            se_a1=jackknife_se([s['a1'] for s in good]),
            n_ok=len(good), wall_s=r.get('wall_s', np.nan)))
    return pd.DataFrame(rows)


def add_quality_flags(df):
    """Flag windows whose fit should not be trusted.

    Criteria that apply to ANY profile:
      - the central fit errored or did not converge
      - beta pinned is NOT a disqualifier (it is a model-form statement, see
        results/beta_ratio_invariance.png), so it is recorded, not cut
      - alpha driven to an unphysical value
      - a ratio that is exactly 0 or 1 (degenerate axis)
    """
    df = df.copy()
    df['alpha_bad'] = df.alpha >= 10
    df['ratio_bad'] = (df.a2a1 <= 1e-3) | (df.a3a2 <= 1e-3) | (df.a3a2 >= 0.999)
    df['degen'] = (~df.central_ok.astype(bool) | ~df.converged.astype(bool)
                   | df.alpha_bad | df.ratio_bad)
    return df


if __name__ == '__main__':
    d = sys.argv[1]
    out = sys.argv[2]
    df = add_quality_flags(load_dir(d))
    df.to_csv(out, index=False)
    print(f'{len(df)} windows -> {out}')
    print(f'  degenerate: {int(df.degen.sum())}   clean: {int((~df.degen).sum())}')
