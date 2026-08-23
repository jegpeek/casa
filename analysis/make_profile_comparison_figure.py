#!/usr/bin/env python
"""Does the per-band profile choice (Weibull vs power law) matter?

Within one 0.6-dex band of lags the Weibull's saturation scale is never
sampled -- the fitted a1 exceeds the band's own outer radius in >99% of fits --
so its shape parameter beta has nothing to constrain it and pins to a bound in
~63% of bands.  Where beta pins to the UPPER bound the Weibull is numerically
indistinguishable from the power law it is reducing to; where beta floats, the
two forms give measurably different axis ratios.

The figure has three panels:
  (a) where beta lands relative to its (1, 10) bounds, per band;
  (b) the a3/a2 scale trend under both profiles -- the headline result;
  (c) the a2/a1 scale trend under both profiles -- the one that moves.

Usage:
    python analysis/make_profile_comparison_figure.py --out fig.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scale_split as ss                # noqa: E402
import summarize_scale_profile as ssp   # noqa: E402
import summarize_scale_slopes as sss    # noqa: E402

# Threading (figure-style §4.1): one colour per profile, reused in every panel.
C_WEIB = '#1f77b4'
C_PLAW = '#d95f02'
BASE, MID, SMALL = 8, 7, 6


def load(profile, data_dir=None):
    bands, _ = ssp.summarize(data_dir=data_dir, profile=profile)
    return pd.DataFrame(bands)


def beta_table(data_dir=None):
    """Per-band fraction of Weibull fits with beta pinned to each bound.

    Reads the *weibull* tree by name.  It must not read the unsuffixed
    directory: that is whichever profile is canonical, which since the power
    law was adopted has no `beta` at all.  Every file is checked against the
    profile tag it recorded, so a stale or mislabelled tree fails loudly
    rather than being silently counted.
    """
    import glob
    import json
    root = data_dir or os.path.join(_ROOT, 'data')
    sub = 'scale_profile_d0.6_s2' + ss.profile_suffix('weibull')
    rows = []
    files = sorted(glob.glob(os.path.join(root, sub, 'sp_r*.json')))
    if not files:
        raise SystemExit(
            'no Weibull band fits under %s -- run\n'
            "    python analysis/scale_profile.py --profile weibull\n"
            'to produce them (see REPRODUCING.md).' % os.path.join(root, sub))
    for f in files:
        d = json.load(open(f))
        tag = d.get('profile')
        if tag != 'weibull':
            raise SystemExit('%s records profile=%r, expected weibull' % (f, tag))
        for bi, b in enumerate(d['bands']):
            if 'error' in b or not b.get('fit_success'):
                continue
            rows.append(dict(band=bi, beta=b['beta'], r_mid=b['r_mid'],
                             a1=b['a1'], r_hi=b['r_hi']))
    W = pd.DataFrame(rows)
    # The bounds are (1.0, 10.0); "pinned" means the optimiser stopped there.
    W['where'] = np.where(W.beta > 9.999, 'upper',
                          np.where(W.beta < 1.001, 'lower', 'interior'))
    return W


def per_window_slopes(B, key):
    return B.groupby('chunk').apply(lambda d: sss.slope_of(d, key),
                                    include_groups=False).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(_ROOT, 'results',
                                                  'profile_comparison.png'))
    ap.add_argument('--data-dir', default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    W = beta_table(a.data_dir)
    BW = load('weibull', a.data_dir)
    BP = load('powerlaw', a.data_dir)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))

    # ---- (a) where beta lands -------------------------------------------
    ax = axes[0]
    frac = (W.groupby(['band', 'where']).size()
            / W.groupby('band').size()).unstack(fill_value=0.0)
    for col in ('lower', 'interior', 'upper'):
        if col not in frac:
            frac[col] = 0.0
    r_mid = W.groupby('band').r_mid.median()
    x = np.arange(len(frac))
    # Stacked: upper-bound fits are the ones where Weibull == power law.
    ax.bar(x, frac['upper'], color=C_PLAW, label=r'$\beta$ at upper bound (10)')
    ax.bar(x, frac['interior'], bottom=frac['upper'], color='0.75',
           label=r'$\beta$ interior')
    ax.bar(x, frac['lower'], bottom=frac['upper'] + frac['interior'],
           color=C_WEIB, label=r'$\beta$ at lower bound (1)')
    ax.set_xticks(x)
    ax.set_xticklabels(['%.3f' % v for v in r_mid], fontsize=SMALL)
    ax.set_xlabel('band centre  [ly]', fontsize=BASE)
    ax.set_ylabel('fraction of window fits', fontsize=BASE)
    ax.set_ylim(0, 1)
    ax.set_title(r'The Weibull $\beta$ is unconstrained in a band',
                 loc='left', fontsize=BASE)
    ax.legend(fontsize=SMALL, frameon=False, loc='lower left')

    # ---- (b), (c) the two ratio trends ----------------------------------
    for ax, key, lab, claim in (
            # Panel (c)'s title must not claim a slope change: BOTH full-range
            # a2/a1 slopes are null (p = 0.66, 0.26).  What actually differs is
            # the band-by-band level -- up to 0.10 in the middle band -- and the
            # robustness subset, annotated below.
            (axes[1], 'a3a2', r'$a_3/a_2$', 'The short-axis trend is unchanged'),
            (axes[2], 'a2a1', r'$a_2/a_1$',
             'The long-axis ratio shifts band by band')):
        for B, c, name in ((BW, C_WEIB, 'Weibull'), (BP, C_PLAW, 'power law')):
            u = B[B.usable]
            g = u.groupby('band').agg(r=('r_mid', 'median'),
                                      m=(key, 'median'),
                                      lo=(key, lambda s: s.quantile(0.25)),
                                      hi=(key, lambda s: s.quantile(0.75)))
            ax.fill_between(g.r, g.lo, g.hi, color=c, alpha=0.15, lw=0)
            ax.plot(g.r, g.m, 'o-', color=c, ms=3.5, lw=1.4, label=name)
        # Per-window slope + signed-rank p, the repo's own aggregation.
        for B, c, dy in ((BW, C_WEIB, 0.15), (BP, C_PLAW, 0.05)):
            s = per_window_slopes(B, key)
            from scipy import stats
            p = stats.wilcoxon(s).pvalue
            ax.annotate('%s: %+.2f/dex, p = %s'
                        % ('Weibull' if c == C_WEIB else 'power law',
                           s.median(), _pfmt(p)),
                        xy=(0.03, dy), xycoords='axes fraction',
                        fontsize=SMALL, color=c, ha='left', va='bottom')
        ax.set_xscale('log')
        ax.set_xticks([0.02, 0.05, 0.1, 0.2])
        ax.set_xticklabels(['0.02', '0.05', '0.1', '0.2'], fontsize=SMALL)
        ax.minorticks_off()
        ax.set_xlim(0.015, 0.36)
        ax.set_xlabel('band centre  [ly]', fontsize=BASE)
        ax.set_ylabel(lab + '   (median over windows)', fontsize=BASE)
        ax.set_title(claim, loc='left', fontsize=BASE)
        ax.legend(fontsize=SMALL, frameon=False, loc='upper right')
        if key == 'a2a1':
            # Both full-range slopes are null, but the repo's robustness subset
            # (widest band dropped) separates the two profiles: Weibull is
            # marginal, the power law is not.  Say so on the panel.
            sub = []
            for B, nm in ((BW, 'Weibull'), (BP, 'power law')):
                u = B[B.usable & (B.band < B.band.max())]
                s = per_window_slopes(u, key)
                from scipy import stats
                sub.append('%s %+.2f (p = %s)'
                           % (nm, s.median(), _pfmt(stats.wilcoxon(s).pvalue)))
            ax.annotate('widest band dropped:\n' + '\n'.join(sub),
                        xy=(0.03, 0.72), xycoords='axes fraction',
                        fontsize=SMALL, color='0.30', ha='left', va='top')

    for ax in axes:
        ax.tick_params(labelsize=SMALL)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print('wrote %s' % a.out)


def _pfmt(p):
    # 7e-8 must not print as 1e-7: show mantissa and exponent explicitly.
    if p > 0.01:
        return '%.2f' % p
    e = int(np.floor(np.log10(p)))
    return r'$%.0f\times10^{%d}$' % (p / 10.0 ** e, e)


if __name__ == '__main__':
    raise SystemExit(main())
