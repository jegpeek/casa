"""Image montage of the 29 scale-profile windows, ordered by Weibull turnover.

WHY THIS EXISTS.  Some windows' isotropic S2 flattens (saturates) at small lag;
others keep rising like a power law out to the longest lag the data sample.
This asks whether that difference is visible in the images, and orders the
windows by *where* the turnover sits so the eye can walk the spectrum.

The turnover scale is derived from the full-range Weibull fit
(results/full_sample_weibull_s2.csv): S2(r) = var_inf * (1 - exp(-r^beta))^(alpha/beta)
with r = |L^-1 . lag|.  We average the saturation factor (1-exp(-r^beta))^(alpha/beta)
over solid angle -- so it uses alpha, beta and all three axes but is orientation
free -- and define the turnover as the lag at which that spherical-mean factor
reaches half its plateau.  This is robust to the band-4 a1 runaway (the fitted
a1 is unconstrained where the profile is scale free; the turnover is not, because
it is fixed by the smallest coherence length a3, Spearman rho = 0.97).

Rendering (rgb_window) is shared with make_band4_montage and is preprocessing-
variant independent -- an epoch-to-epoch RGB composite of the raw flux, one
joint asinh stretch per window.  Only the RANKING carries a variant: it is read
from the arcsinh full-sample Weibull table, the one the paper's fits use.

Usage:  python analysis/make_turnover_montage.py [--outdir results/figures]
Needs the bulk input arrays (tier B); writes turnover_montage.png and
turnover_montage_table.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, 'analysis')):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_band4_montage as mb          # rgb_window, PCT/SOFT stretch constants

NCOL = 8
# Largest and smallest lag the data actually sample (results band table).
R_LO_SAMPLED = 0.0096
R_HI_SAMPLED = 0.5007


def _sphere_dirs(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


_DIRS = _sphere_dirs()


def turnover_ly(a1, a2, a3, alpha, beta, frac=0.5):
    """Lag at which the solid-angle-averaged Weibull saturation factor hits `frac`."""
    Dn = np.sqrt((_DIRS[:, 0] / a1) ** 2 + (_DIRS[:, 1] / a2) ** 2
                 + (_DIRS[:, 2] / a3) ** 2)
    lags = np.geomspace(1e-3, 1e3, 400)
    g = np.array([np.mean(np.maximum(-np.expm1(-(L * Dn) ** beta), 1e-300)
                          ** (alpha / beta)) for L in lags])
    g /= g[-1]
    i = np.searchsorted(g, frac)
    if i <= 0 or i >= len(lags):
        return np.nan
    return float(np.exp(np.interp(frac, [g[i - 1], g[i]],
                                  [np.log(lags[i - 1]), np.log(lags[i])])))


def knee_ly(a1, a2, a3, alpha, beta):
    """Lag of the max-curvature 'knee' of the solid-angle-averaged Weibull.

    Marks the corner of the log-log S2(lag) curve (its most concave-down
    point), i.e. where the profile bends away from its small-scale power law
    -- distinct from turnover_ly, which marks the *onset* of saturation (half
    of the plateau) and sits well below the visible knee.  Orientation-free:
    the saturation factor is averaged over solid angle before curvature.
    """
    Dn = np.sqrt((_DIRS[:, 0] / a1) ** 2 + (_DIRS[:, 1] / a2) ** 2
                 + (_DIRS[:, 2] / a3) ** 2)
    lags = np.geomspace(1e-3, 1e2, 800)
    F = np.array([np.mean(np.maximum(-np.expm1(-(L * Dn) ** beta), 1e-300)
                          ** (alpha / beta)) for L in lags])
    x = np.log10(lags)
    y = np.log10(F)
    dy = np.gradient(y, x)
    d2 = np.gradient(dy, x)
    kappa = -d2 / (1.0 + dy ** 2) ** 1.5      # signed: >0 = concave down
    return float(lags[int(np.argmax(kappa))])


#: SNR tiers, defined exactly as make_tier_figures.load(): q4 = top quartile
#: (SNR >= 75th pct), q3 = second quartile (50th-75th), bottom_half = below.
#: The SNR column is arcsinh (noise_audit_table.csv, no suffix), matching the
#: arcsinh full-range Weibull fits this figure reads.  For q4 this set is
#: byte-identical to the old scale_profile glob (verified).
TIER_LABELS = {'q4': 'top quartile', 'q3': '2nd quartile',
               'bottom_half': 'bottom half'}


def tier_windows(tier='q4'):
    """(key, row, col, snr) for the windows in one SNR tier, arcsinh SNR."""
    snr = pd.read_csv(os.path.join(_ROOT, 'results', 'noise_audit_table.csv'))
    q75, q50 = np.percentile(snr.snr, [75, 50])
    snr['tier'] = np.where(snr.snr >= q75, 'q4',
                           np.where(snr.snr >= q50, 'q3', 'bottom_half'))
    sel = snr[snr.tier == tier].copy()
    sel['key'] = sel.apply(lambda x: 'r%d_c%d' % (x.row, x.col), axis=1)
    return sel[['key', 'row', 'col', 'snr']]


def build_table(tier='q4'):
    wb = pd.read_csv(os.path.join(_ROOT, 'results', 'full_sample_weibull_s2.csv'))
    wb['key'] = wb.apply(lambda x: 'r%d_c%d' % (x.row, x.col), axis=1)
    win = tier_windows(tier)
    d = wb[wb.key.isin(set(win.key))].copy()
    d = d.merge(win[['key', 'snr']], on='key', how='left')
    d['a2'] = d.a1 * d.a2a1
    d['a3'] = d.a2 * d.a3a2
    d['turnover'] = [turnover_ly(r.a1, r.a2, r.a3, r.alpha, r.beta)
                     for r in d.itertuples()]
    d['knee'] = [knee_ly(r.a1, r.a2, r.a3, r.alpha, r.beta)
                 for r in d.itertuples()]
    return d.sort_values('turnover').reset_index(drop=True)


def main(outdir='results/figures', tier='q4'):
    try:
        from figure_style import apply_figure_style          # noqa
        apply_figure_style()
    except Exception:
        pass

    d = build_table(tier)
    snr_lo, snr_hi = float(d.snr.min()), float(d.snr.max())
    tier_lab = TIER_LABELS.get(tier, tier)
    suffix = '' if tier == 'q4' else '_' + tier
    nrow = int(np.ceil(len(d) / NCOL))

    # turnover -> colour, sequential (small = flattens early, large = power-law-like)
    norm = mpl.colors.LogNorm(vmin=d.turnover.min(), vmax=d.turnover.max())
    cmap = mpl.cm.viridis

    fig = plt.figure(figsize=(NCOL * 1.42, nrow * 1.66 + 1.35))
    gs = GridSpec(nrow, NCOL, figure=fig, hspace=0.42, wspace=0.06,
                  top=0.885, bottom=0.075, left=0.010, right=0.988)

    for k, rec in enumerate(d.itertuples()):
        ax = fig.add_subplot(gs[k // NCOL, k % NCOL])
        img, _ = mb.rgb_window(int(rec.row), int(rec.col))
        ax.imshow(img, origin='lower', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        col = cmap(norm(rec.turnover))
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(2.4)
        ax.set_title('%d,%d' % (rec.row, rec.col), fontsize=6, pad=1.6)
        # resolved deep in the data vs only near the longest lag
        frac_of_range = rec.turnover / R_HI_SAMPLED
        ax.text(0.5, -0.055,
                r'turnover=%.3f ly' % rec.turnover,
                transform=ax.transAxes, ha='center', va='top', fontsize=6.0,
                color=col if frac_of_range < 0.9 else '0.15')

    # colourbar for turnover, placed in the empty bottom-right cells so it
    # never collides with the footer caption.
    n_empty = nrow * NCOL - len(d)
    if n_empty >= 3:
        cax = fig.add_axes([0.655, 0.150, 0.30, 0.022])
    else:
        cax = fig.add_axes([0.32, 0.033, 0.36, 0.016])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation='horizontal')
    cb.set_label('Weibull turnover scale (ly)\nspherical half-saturation lag',
                 fontsize=7)
    cb.ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    cb.ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    cb.set_ticks([0.03, 0.05, 0.1, 0.2, 0.4])
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(
        ['0.03', '0.05', '0.1', '0.2', '0.4']))
    cb.ax.tick_params(labelsize=6, which='both')
    # mark the sampled lag ceiling on the bar
    if norm.vmin <= R_HI_SAMPLED <= norm.vmax:
        cb.ax.axvline(R_HI_SAMPLED, color='0.1', lw=1.0)

    fig.suptitle(
        'Ordered by where the structure function turns over: top-left windows '
        'saturate at small lag,\nbottom-right keep rising like a power law out '
        'to the longest lag sampled ($r$ = %.2f ly)  —  %s windows (SNR %.1f–%.1f)'
        % (R_HI_SAMPLED, tier_lab, snr_lo, snr_hi),
        fontsize=9, y=0.980)
    fig.text(0.988, 0.008,
             'RGB = epochs 3/4/5, one joint asinh stretch per window (colour = '
             'epoch-to-epoch change).  Frame colour = turnover scale; labels: '
             'window row,col.  Turnover from the full-range Weibull fit '
             '(arcsinh).  Sampled lags $r$ = %.3f–%.2f ly.'
             % (R_LO_SAMPLED, R_HI_SAMPLED),
             fontsize=6.0, ha='right', va='bottom', color='0.25')

    out = os.path.join(_ROOT, outdir)
    os.makedirs(out, exist_ok=True)
    p1 = os.path.join(out, 'turnover_montage%s.png' % suffix)
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    tbl = os.path.join(out, 'turnover_montage_table%s.csv' % suffix)
    d[['key', 'row', 'col', 'turnover', 'a1', 'a2', 'a3', 'alpha', 'beta',
       'var_inf']].to_csv(tbl, index=False)
    print('wrote %s' % p1)
    print('wrote %s' % tbl)
    return p1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results/figures')
    ap.add_argument('--tier', default='q4', choices=['q4', 'q3', 'bottom_half'],
                    help='SNR tier: q4=top quartile (default), q3=2nd quartile')
    a = ap.parse_args()
    main(outdir=a.outdir, tier=a.tier)
