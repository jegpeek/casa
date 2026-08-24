"""Image montage of the 29 scale-profile windows, ordered by band-4 a2/a1.

WHY THIS EXISTS.  In the longest lag band (band 4, r = 0.152-0.501 ly) the
fitted long-to-middle axis ratio a2/a1 appears to BIFURCATE: eight windows sit
at a2/a1 < 0.07 and the next one up is at 0.175, with nothing in between.  The
obvious question is whether those eight windows look different on the sky --
i.e. whether there are two populations of cloud morphology.

The montage answers it, and the answer is no.  The split is not in the images,
it is in the fitted a1, which in this band is unconstrained.  Over the 28
windows that pass the summarizer's degeneracy cut:

  * the two groups' MIDDLE axis a2 is statistically identical
    (median 0.491 vs 0.455 ly, Mann-Whitney p = 0.86);
  * they differ only in a1 (median 13.0 ly vs 0.97 ly), which for the low group
    is 14-84x the largest lag the band samples (r_hi = 0.501 ly) and is
    therefore an extrapolation, not a measurement.  27 of the 28 windows have
    a1 > r_hi at all;
  * a2/a1 tracks a1 almost perfectly (Spearman rho = -0.90, p = 5e-11) and
    tracks a2 not at all (rho = -0.10, p = 0.60);
  * a1 anticorrelates with the fitted alpha (rho = -0.39, p = 0.04), and the
    low group's alpha is flat where the high group's is not (median 0.162 vs
    0.457, p = 0.002).  That is the expected signature of the power law's exact
    scale degeneracy: for S2 = A r^alpha with A frozen, alpha -> 0 makes the
    profile scale-free and a1 runs away.  Small alpha and huge a1 are one
    pathology, not two.

So a2/a1 in this band is not a shape measurement and the bifurcation is not a
result.  The montage is the falsifiable version of that statement: if the eight
were a distinct population, they would look distinct, and they do not.

Ordering is by a2/a1 ascending, so the eight runaway fits occupy the first row
and are separated by a rule.  Each panel is an RGB composite of epochs 3/4/5
(indices 2, 3, 4) -- the three epochs the caller asked for -- stretched jointly
across the three channels so that colour encodes real epoch-to-epoch change
(the echo sweeping through the cloud) rather than a per-channel normalisation.

Usage:  python analysis/make_band4_montage.py [--band 4] [--outdir results/figures]
Needs the bulk input arrays (tier B); writes band4_montage.png and
band4_degeneracy.png plus band4_montage_table.csv.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

import scale_split as ss          # noqa: E402
import structure_function as sf   # noqa: E402

RGB_EPOCHS = (2, 3, 4)      # third/fourth/fifth epoch -> R/G/B
# Stretch percentiles.  The upper cut is 99.0 rather than the max because a few
# windows contain a compact source ~50x the diffuse echo (r3200_c2400: median
# 0.016, p99.5 = 0.90); scaling to it renders the echo itself black.
PCT = (1.0, 99.0)
# asinh softening, in units of the (p1, p99) range.  A linear stretch is wrong
# here for the same reason: these windows are strongly skewed, so linear scaling
# buys highlight fidelity we do not need at the cost of the faint filamentary
# structure that IS the morphology being compared.  asinh is the standard
# astronomical compromise and keeps both.  0.3 is the value that leaves the
# faintest window (r3200_c2400) legible at a stretched median of 0.11 while
# holding the brightest below 4 % saturated pixels; 0.1 amplifies the noise.
SOFT = 0.30
SPLIT = 0.10                # a2/a1 value that separates the two groups
NCOL = 8


def _psci(p):
    """p-value as LaTeX scientific notation ('5\\times10^{-11}')."""
    if p >= 1e-3:
        return '%.3f' % p
    e = int(np.floor(np.log10(p)))
    return r'%.0f\times10^{%d}' % (p / 10.0 ** e, e)


def usable_chunks(band, csv_name='results/scale_profile_d0.6_s2_powerlaw_bands.csv'):
    """The (row, col) of band fits the summarizer marks usable.

    The degeneracy cut is NOT ours to re-invent: summarize_scale_profile.py
    rejects fits that park on a boundary (a3 -> 0, or an axis ratio below the
    0.02 floor).  In band 4 exactly one window fails it -- r2400_c4400, whose
    a1 reached 293 ly for a 0.5 ly lag range -- and an excluded row must not
    re-enter a figure alongside the included ones.  It is named in the caption
    instead, since it is the same pathology in the extreme.
    """
    import csv as _csv
    keep = set()
    with open(os.path.join(_ROOT, csv_name)) as fh:
        for r in _csv.DictReader(fh):
            if int(r['band']) == band and r['usable'].strip().lower() in ('true', '1'):
                keep.add(r['chunk'])
    return keep


def band_records(band, profile_dir='data/scale_profile_d0.6_s2'):
    """Per-window central fit for one band, read from the tracked JSONs."""
    rows = []
    for path in sorted(glob.glob(os.path.join(_ROOT, profile_dir, '*.json'))):
        j = json.load(open(path))
        bands = j.get('bands', [])
        if band >= len(bands):
            continue
        b = bands[band]
        c = b.get('central', b)
        chunk = j.get('chunk', os.path.basename(path))
        # sp_r2800_c1200_s400 -> row 2800, col 1200
        parts = chunk.split('_')
        row = int([p for p in parts if p.startswith('r')][0][1:])
        col = int([p for p in parts if p.startswith('c')][0][1:])
        a1, a2, a3 = c.get('a1'), c.get('a2'), c.get('a3')
        rows.append(dict(chunk=chunk, key='r%d_c%d' % (row, col), row=row, col=col,
                         r_lo=b.get('r_lo'), r_hi=b.get('r_hi'),
                         a1=a1, a2=a2, a3=a3,
                         a2a1=(a2 / a1) if a1 else np.nan,
                         alpha=c.get('alpha'),
                         fit_success=c.get('fit_success', b.get('fit_success'))))
    return rows


def rgb_window(row, col, size=400):
    """RGB composite of the three requested epochs, jointly stretched."""
    d = sf.read_window(row, col, size, size,
                       data_dir=os.path.join(_ROOT, 'data'), **ss.READ_KW)
    f = d['flux_epochs'][list(RGB_EPOCHS)].astype(float)
    finite = np.isfinite(f)
    if not finite.any():
        return np.zeros((size, size, 3)), 0.0
    lo, hi = np.nanpercentile(f[finite], PCT)     # one stretch for all three
    if not (hi > lo):
        return np.zeros((size, size, 3)), float(finite.mean())
    x = np.clip((f - lo) / (hi - lo), 0, 1)
    img = np.arcsinh(x / SOFT) / np.arcsinh(1.0 / SOFT)
    img[~finite] = 0.0
    return np.moveaxis(img, 0, -1), float(finite.mean())


def main(band=4, outdir='results/figures'):
    keep = usable_chunks(band)
    allrecs = [r for r in band_records(band) if np.isfinite(r['a2a1'])]
    recs = [r for r in allrecs if r['key'] in keep]
    dropped = [r for r in allrecs if r['key'] not in keep]
    recs.sort(key=lambda r: r['a2a1'])
    lo_group = [r for r in recs if r['a2a1'] < SPLIT]
    hi_group = [r for r in recs if r['a2a1'] >= SPLIT]

    out = os.path.join(_ROOT, outdir)
    os.makedirs(out, exist_ok=True)

    # ---- montage -------------------------------------------------------
    nrow = 1 + int(np.ceil(len(hi_group) / NCOL))
    fig = plt.figure(figsize=(NCOL * 1.42, nrow * 1.62 + 1.25))
    # The two group headers are full-width text, so the rows they head need
    # vertical room that hspace alone cannot give unevenly.  Height ratios add
    # a thin spacer row between group 1 and group 2 for the second header.
    heights = [1.0, 0.30] + [1.0] * (nrow - 1)
    gs = GridSpec(nrow + 1, NCOL, figure=fig, hspace=0.34, wspace=0.06,
                  height_ratios=heights,
                  top=0.895, bottom=0.050, left=0.012, right=0.988)

    written = []
    for k, rec in enumerate(recs):
        if rec['a2a1'] < SPLIT:
            gr, gc = 0, k
        else:
            j = k - len(lo_group)
            gr, gc = 2 + j // NCOL, j % NCOL   # +2 skips the spacer row
        ax = fig.add_subplot(gs[gr, gc])
        img, frac = rgb_window(rec['row'], rec['col'])
        ax.imshow(img, origin='lower', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title('%d,%d' % (rec['row'], rec['col']), fontsize=6, pad=1.6)
        # the two numbers that matter: the ratio, and the axis driving it
        ax.text(0.5, -0.055,
                r'$a_2/a_1$=%.3f   $a_1$=%.3g ly' % (rec['a2a1'], rec['a1']),
                transform=ax.transAxes, ha='center', va='top', fontsize=5.6)
        rec['coverage'] = frac
        written.append(rec)

    # rule + group headers.  The rule sits in the spacer row, midway between
    # the last label of group 1 and the first title of group 2.
    y_top = gs[0, 0].get_position(fig).y0        # bottom of the group-1 row
    y_bot = gs[2, 0].get_position(fig).y1        # top of the group-2 row
    y_rule = 0.5 * (y_top + y_bot)
    fig.add_artist(plt.Line2D([0.012, 0.988], [y_rule, y_rule],
                              color='0.35', lw=0.8, zorder=5))
    a1lo = np.array([r['a1'] for r in lo_group])
    a1hi = np.array([r['a1'] for r in hi_group])
    fig.text(0.012, gs[0, 0].get_position(fig).y1 + 0.010,
             r'$a_2/a_1 < %.2f$  (%d windows) — fitted $a_1$ = %.1f–%.0f ly, '
             r'i.e. %.0f–%.0f$\times$ the largest lag sampled: an '
             r'extrapolation, not a measurement'
             % (SPLIT, len(lo_group), a1lo.min(), a1lo.max(),
                a1lo.min() / recs[0]['r_hi'], a1lo.max() / recs[0]['r_hi']),
             fontsize=6.6, ha='left', va='bottom')
    fig.text(0.012, y_rule - 0.006,
             r'$a_2/a_1 > %.2f$  (%d windows) — fitted $a_1$ = %.2f–%.1f ly'
             % (SPLIT, len(hi_group), a1hi.min(), a1hi.max()),
             fontsize=6.6, ha='left', va='top')

    a2lo = np.array([r['a2'] for r in lo_group])
    a2hi = np.array([r['a2'] for r in hi_group])
    p_a2 = stats.mannwhitneyu(a2lo, a2hi).pvalue
    fig.suptitle('The band-%d $a_2/a_1$ split is in the fitted $a_1$, not in the '
                 'images:\nthe middle axis $a_2$ is the same in both groups '
                 '(%.3f vs %.3f ly, $p=%.2f$)'
                 % (band, np.median(a2lo), np.median(a2hi), p_a2),
                 fontsize=9, y=0.985)
    drop_txt = ('  Excluded by the degeneracy cut and not shown: %s.'
                % ', '.join('%s ($a_1$=%.0f ly)' % (r['key'], r['a1'])
                            for r in dropped)) if dropped else ''
    fig.text(0.988, 0.013,
             'RGB = epochs 3/4/5, one joint stretch per window (colour = '
             'epoch-to-epoch change).  Ordered by $a_2/a_1$ in band 4 '
             '($r$ = 0.15–0.50 ly).  Labels: window row,col.' + drop_txt,
             fontsize=6.0, ha='right', va='bottom', color='0.25')

    p1 = os.path.join(out, 'band4_montage.png')
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    # ---- the diagnostic behind the caption -----------------------------
    a2a1 = np.array([r['a2a1'] for r in recs])
    a1 = np.array([r['a1'] for r in recs])
    a2 = np.array([r['a2'] for r in recs])
    alpha = np.array([r['alpha'] for r in recs])
    lo = a2a1 < SPLIT
    r_hi = recs[0]['r_hi']

    fig2, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), layout='constrained')
    blue, grey = '#1f6fb4', '#8a8a8a'
    # Labels quote the observed gap edges, so they agree with the montage
    # headers by construction rather than by a transcribed constant.
    lab_lo = r'$a_2/a_1 \leq %.3f$  (n=%d)' % (a2a1[lo].max(), lo.sum())
    lab_hi = r'$a_2/a_1 \geq %.3f$  (n=%d)' % (a2a1[~lo].min(), (~lo).sum())

    ax = axes[0]
    ax.scatter(a1[lo], a2a1[lo], s=26, c=blue, zorder=3, label=lab_lo)
    ax.scatter(a1[~lo], a2a1[~lo], s=26, c=grey, zorder=3, label=lab_hi)
    ax.axvline(r_hi, color='0.3', ls=':', lw=1.0)
    # top-anchored: the legend occupies the lower-left corner
    ax.annotate('largest lag sampled ($r$ = %.2f ly)' % r_hi,
                xy=(r_hi, a2a1.max()), xytext=(4, -2), textcoords='offset points',
                fontsize=6.2, rotation=90, va='top', ha='left', color='0.3')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('fitted $a_1$ [ly]')
    ax.set_ylabel('$a_2/a_1$')
    rho, pv = stats.spearmanr(a2a1, a1)
    ax.set_title('The ratio just tracks $a_1$\n'
                 r'$\rho=%+.2f$, $p=%s$' % (rho, _psci(pv)), fontsize=8)
    ax.legend(fontsize=6.4, frameon=False, loc='lower left')

    ax = axes[1]
    ax.scatter(a2[lo], a2a1[lo], s=26, c=blue, zorder=3)
    ax.scatter(a2[~lo], a2a1[~lo], s=26, c=grey, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # a2 spans well under a decade, so the default log minor labels collide.
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: ('%g' % v)))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xticks([0.3, 0.5, 1.0, 2.0])
    ax.set_xlabel('fitted $a_2$ [ly]')
    ax.set_ylabel('$a_2/a_1$')
    ax.set_title('and not $a_2$, which is the same\n'
                 'in both groups ($p=%.2f$)'
                 % stats.mannwhitneyu(a2[lo], a2[~lo]).pvalue, fontsize=8)

    ax = axes[2]
    ax.scatter(alpha[lo], a1[lo], s=26, c=blue, zorder=3)
    ax.scatter(alpha[~lo], a1[~lo], s=26, c=grey, zorder=3)
    ax.axhline(r_hi, color='0.3', ls=':', lw=1.0)
    ax.annotate('largest lag sampled', xy=(alpha.max(), r_hi),
                xytext=(0, 3), textcoords='offset points',
                fontsize=6.2, ha='right', va='bottom', color='0.3')
    ax.set_yscale('log')
    ax.set_xlabel(r'fitted $\alpha$')
    ax.set_ylabel('fitted $a_1$ [ly]')
    rho2, pv2 = stats.spearmanr(alpha, a1)
    ax.set_title(r'$a_1$ runs away as $\alpha\to0$: the' '\n'
                 r'scale degeneracy ($\rho=%+.2f$, $p=%.2f$)' % (rho2, pv2),
                 fontsize=8)

    for ax in axes:
        ax.margins(0.08)
    fig2.suptitle('Why the band-%d $a_2/a_1$ split is a fit degeneracy, not a '
                  'morphology difference  (%d usable windows)'
                  % (band, len(recs)), fontsize=8.5)
    p2 = os.path.join(out, 'band4_degeneracy.png')
    fig2.savefig(p2, dpi=200)
    plt.close(fig2)

    # ---- table ---------------------------------------------------------
    import csv
    p3 = os.path.join(out, 'band4_montage_table.csv')
    cols = ['chunk', 'row', 'col', 'a2a1', 'a1', 'a2', 'a3', 'alpha',
            'r_lo', 'r_hi', 'coverage', 'fit_success']
    with open(p3, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in written:
            w.writerow(r)

    return [p1, p2, p3]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', type=int, default=4)
    ap.add_argument('--outdir', default='results/figures')
    a = ap.parse_args()
    for p in main(band=a.band, outdir=a.outdir):
        print(os.path.relpath(p, _ROOT))
