"""Paired image + S2(lag) profile montage, ordered by Weibull turnover.

Companion to make_turnover_montage.py.  Each window gets two cells side by side:
  (left)  the RGB epoch composite (make_band4_montage.rgb_window), and
  (right) its isotropic structure function S2 vs physical lag, with the
          full-range Weibull fit overlaid and the turnover lag marked.

WHY PHYSICAL LAG (not the ellipsoidal radius r=|L^-1.lag|).  The question this
figure answers -- "why do some windows flatten at small scales and others look
power-law throughout" -- is a statement about physical lag in light-years, the
same axis the montage frame-colour encodes.  So we bin the measured S2 by |lag|
and overlay the model evaluated per pixel and binned the SAME way, so the curve
inherits the window's anisotropy and lag sampling rather than a spherically
symmetric idealisation.  The turnover (spherical half-saturation lag from
make_turnover_montage.turnover_ly) is drawn as a vertical marker; where it sits
inside vs beyond the sampled range is exactly the flatten-early / power-law
distinction.

Fit provenance: arcsinh (CASA_ARCSINH_UNITS=1), matching the paper's fit table
data/jk_weibull_s2/*.json and results/full_sample_weibull_s2.csv.  The IMAGES
are preprocessing-variant independent; the fit and the binned S2 are arcsinh.

Usage:  python analysis/make_turnover_profiles.py [--outdir results/figures]
Writes turnover_profiles.png.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, 'analysis')):
    if p not in sys.path:
        sys.path.insert(0, p)

# The fit table is arcsinh; set BEFORE importing scale_split so COMPUTE_KW binds
# to the arcsinh settings (module state captured at import).
os.environ['CASA_ARCSINH_UNITS'] = '1'

import structure_function as sf          # noqa: E402
import scale_split as ss                 # noqa: E402
import make_band4_montage as mb          # noqa: E402  rgb_window
import make_turnover_montage as tm       # noqa: E402  turnover_ly, build_table

_GEOM = ['s11', 's22', 's33', 'l12', 'l13', 'l23']
NCOL_PAIRS = 4          # 4 (image,curve) pairs per row -> 8 axis columns
R_LO_SAMPLED = 0.0096
R_HI_SAMPLED = 0.5007
# in-plane physical scale (ly per pixel), from the U grid spacing
_Ug = np.load(os.path.join(_ROOT, 'data', 'U_grid.npy'), mmap_mode='r')
LY_PER_PIX = float(_Ug[0, 1] - _Ug[0, 0])


def phys_profile(row, col, central, nbin=14, rmax=R_HI_SAMPLED, rmin=8e-3):
    """Measured S2 and model, both binned by physical lag |lag| [ly]."""
    d = sf.read_window(row, col, 400, 400, data_dir=os.path.join(_ROOT, 'data'),
                       **ss.READ_KW)
    s2 = sf.compute_s2(d, **ss.COMPUTE_KW)
    val, nc = s2['s2'], s2['n_counts']
    DV, DU = np.meshgrid(s2['lag_dv'], s2['lag_du'], indexing='ij')
    geom = [central[k] for k in _GEOM]
    prof = [central['alpha'], central['beta'], central['var_inf']]
    LAG, S, W, MOD = [], [], [], []
    for k, dw in enumerate(s2['lag_dw']):
        lags = np.column_stack([DU.ravel(), DV.ravel(),
                                np.full(DU.size, dw)])
        lagr = np.sqrt((lags ** 2).sum(1))
        r = sf._compute_r(geom, lags)
        mod = 10 ** sf.weibull_log_s2(r, prof)
        v = val[k].ravel()
        w = nc[k].ravel().astype(float)
        m = (np.isfinite(v) & (w > 0) & (v > 0)
             & (lagr > 0) & (lagr <= rmax))
        LAG.append(lagr[m]); S.append(v[m]); W.append(w[m]); MOD.append(mod[m])
    lag = np.concatenate(LAG); v = np.concatenate(S)
    w = np.concatenate(W); mod = np.concatenate(MOD)
    edges = np.geomspace(max(lag.min(), rmin), rmax, nbin + 1)
    idx = np.digitize(lag, edges) - 1
    lb, sb, mbin = [], [], []
    for b in range(nbin):
        sel = idx == b
        if sel.sum() < 5:
            continue
        ww = w[sel]
        lb.append(np.exp(np.average(np.log(lag[sel]), weights=ww)))
        sb.append(np.average(v[sel], weights=ww))
        mbin.append(np.average(mod[sel], weights=ww))
    # smooth model line on a fine physical grid, from the same directional mix:
    # reuse the per-pixel model already binned (mb) but draw as a line through it
    return np.array(lb), np.array(sb), np.array(mbin), prof


def main(outdir='results/figures', tier='q4'):
    try:
        from figure_style import apply_figure_style          # noqa
        apply_figure_style()
    except Exception:
        pass

    d = tm.build_table(tier)      # arcsinh table (carries turnover + knee)
    d = d.sort_values('knee').reset_index(drop=True)   # order by max-curv knee
    snr_lo, snr_hi = float(d.snr.min()), float(d.snr.max())
    tier_lab = tm.TIER_LABELS.get(tier, tier)
    suffix = '' if tier == 'q4' else '_' + tier
    n = len(d)
    nrow = int(np.ceil(n / NCOL_PAIRS))

    norm = mpl.colors.LogNorm(vmin=d.knee.min(), vmax=d.knee.max())
    cmap = mpl.cm.viridis

    fig = plt.figure(figsize=(NCOL_PAIRS * 3.55, nrow * 1.95 + 1.0))
    # each pair = [image col (wider), curve col]; small gap between pairs
    gs = GridSpec(nrow, NCOL_PAIRS * 2, figure=fig,
                  width_ratios=[1.0, 1.15] * NCOL_PAIRS,
                  hspace=0.55, wspace=0.42,
                  top=0.925, bottom=0.055, left=0.035, right=0.990)

    for k, rec in enumerate(d.itertuples()):
        gr = k // NCOL_PAIRS
        gc = (k % NCOL_PAIRS) * 2
        col = cmap(norm(rec.knee))

        # ---- image ----
        axi = fig.add_subplot(gs[gr, gc])
        img, _ = mb.rgb_window(int(rec.row), int(rec.col))
        axi.imshow(img, origin='lower', interpolation='nearest')
        axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(2.2)
        axi.set_title('%d,%d' % (rec.row, rec.col), fontsize=6.5, pad=1.5)
        # ---- 0.1 ly scale bar (lower-left) ----
        npx = img.shape[1]                       # window width in px (400)
        bar_px = 0.1 / LY_PER_PIX                # 0.1 ly in pixels
        x0, y0 = 0.06 * npx, 0.075 * npx         # bar left end, in px
        axi.plot([x0, x0 + bar_px], [y0, y0], '-', color='white', lw=2.0,
                 solid_capstyle='butt', zorder=5)
        axi.text(x0 + bar_px / 2.0, y0 + 0.035 * npx, '0.1 ly',
                 color='white', fontsize=5.5, ha='center', va='bottom', zorder=5)

        # ---- profile ----
        axp = fig.add_subplot(gs[gr, gc + 1])
        j = json.load(open(os.path.join(
            _ROOT, 'data', 'jk_weibull_s2',
            'jk_%s_s400.json' % rec.key)))['central']
        lb, sb, mbin, prof = phys_profile(int(rec.row), int(rec.col), j)
        axp.plot(lb, sb, 'o', ms=2.6, color='0.15', zorder=3,
                 label='S2')
        axp.plot(lb, mbin, '-', lw=1.4, color=col, zorder=2, label='Weibull fit')
        axp.axvline(rec.knee, color=col, ls='--', lw=1.0, zorder=1)
        axp.set_xscale('log'); axp.set_yscale('log')
        axp.set_xlim(8e-3, R_HI_SAMPLED * 1.05)
        # y ticks on the RIGHT so they never collide with the image to the left
        axp.yaxis.tick_right()
        axp.yaxis.set_label_position('right')
        axp.tick_params(labelsize=5.5, pad=1.2)
        # keep the minor-tick label clutter down on the log y axis
        axp.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # knee label; flag when the knee falls beyond the sampled lag range
        klab = (r'$k$=%.2f$\rightarrow$' % rec.knee
                if rec.knee > R_HI_SAMPLED else r'$k$=%.3f' % rec.knee)
        axp.text(0.05, 0.93, klab, transform=axp.transAxes,
                 fontsize=6.0, va='top', ha='left', color=col)
        # x label only on bottom row of each column
        if gr == nrow - 1 or k + NCOL_PAIRS >= n:
            axp.set_xlabel('lag (ly)', fontsize=6)
        axp.set_ylabel(r'$S_2$', fontsize=6)

    fig.suptitle(
        'Structure function and Weibull fit beside each image, ordered by the '
        'knee.  Dashed line: max-curvature knee lag $k$ (ly);\nsmall $k$ '
        '(top-left) bends inside the data, large $k$ (bottom-right) is still '
        'power-law at the longest lag sampled ($r$=%.2f ly; $k\\rightarrow$ = '
        'knee beyond range)  —  %s windows (SNR %.1f–%.1f)'
        % (R_HI_SAMPLED, tier_lab, snr_lo, snr_hi),
        fontsize=9, y=0.985)
    fig.text(0.990, 0.008,
             'Left: RGB epochs 3/4/5, one joint asinh stretch (colour = '
             'epoch-to-epoch change).  Right: measured $S_2$ (points) vs '
             'physical lag with the full-range Weibull fit (line), both '
             'anisotropy-weighted; arcsinh.  Frame/line colour = knee.',
             fontsize=6.0, ha='right', va='bottom', color='0.25')

    out = os.path.join(_ROOT, outdir)
    os.makedirs(out, exist_ok=True)
    p1 = os.path.join(out, 'turnover_profiles%s.png' % suffix)
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print('wrote %s' % p1)
    return p1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results/figures')
    ap.add_argument('--tier', default='q4', choices=['q4', 'q3', 'bottom_half'],
                    help='SNR tier: q4=top quartile (default), q3=2nd quartile')
    a = ap.parse_args()
    main(outdir=a.outdir, tier=a.tier)
