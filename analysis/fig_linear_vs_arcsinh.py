"""Referee figure: does the arcsinh preprocessing create the result?

Three panels, one message -- every conclusion survives in raw flux units, with
larger error bars.

  a  per-window axis ratios, published against linear, on the 1:1 line.  This
     is the fairest comparison: same window, same estimator, only preprocessing
     differs.  Points on the line mean the transform did not move the fit.
  b  the two common shapes with their measurement (+) intrinsic ellipses, in
     the shape plane, against the prolate/oblate divide.
  c  the five headline statistics as a dumbbell, normalized so each sits on a
     common axis; the significance drop is visible without hiding the survival.

Writes results/figures/linear_vs_arcsinh.{png,pdf}.
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

import make_tier_figures as mtf  # noqa: E402
import compare_linear_vs_arcsinh as cmp  # noqa: E402

BLUE = '#1f6fb4'      # published (arcsinh)
ORANGE = '#d1650b'    # linear -- distinct hue, CVD-safe against blue
GREY = '#9a9a9a'


def _q4(variant):
    d = mtf.load(k=4, variant=variant)[0]
    return d[mtf.usable(d) & (d.tier == 'q4')]


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mtf.apply_figure_style(sizes=(9, 8, 7))

    pub, lin = _q4(''), _q4('_linear')
    key = ['row', 'col']
    m = pub[key + ['a2a1', 'a3a2', 'se_a2a1', 'se_a3a2']].merge(
        lin[key + ['a2a1', 'a3a2', 'se_a2a1', 'se_a3a2']],
        on=key, suffixes=('_p', '_l'))

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          hspace=0.34, wspace=0.30)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, :])

    # ---------------------------------------------------------------- panel a
    # Blue/orange are bound to published/linear for the whole figure, so this
    # panel -- whose two AXES already carry that contrast -- separates the two
    # ratios by marker shape in a neutral hue instead of stealing the hues.
    for c, mk, fc, lab in (('a2a1', 'o', '0.25', r'$a_2/a_1$'),
                           ('a3a2', 's', 'none', r'$a_3/a_2$')):
        axa.errorbar(m[c + '_p'], m[c + '_l'],
                     xerr=m['se_' + c + '_p'], yerr=m['se_' + c + '_l'],
                     fmt=mk, ms=4.5, mew=0.9, lw=0.6, alpha=0.6,
                     color='0.25', mfc=fc, mec='0.25', label=lab)
    lims = (0.03, 1.6)
    axa.plot(lims, lims, color=GREY, lw=1.0, zorder=0)
    axa.set(xscale='log', yscale='log', xlim=lims, ylim=lims)
    # Default log locators emit a 10^1 label just outside the 1.6 limit, whose
    # text box then falls off the axes; pin the decades that are in range.
    for axis in (axa.xaxis, axa.yaxis):
        axis.set_ticks([0.1, 1.0])
        axis.set_ticklabels(['0.1', '1'])
        axis.set_ticks([], minor=True)
    axa.set_xlabel('published (arcsinh)')
    axa.set_ylabel('linear flux units')
    axa.set_title('Same window, same estimator:\nthe fits track the 1:1 line',
                  loc='left')
    axa.legend(frameon=False, loc='upper left', handletextpad=0.4,
               borderaxespad=0.15)

    # ---------------------------------------------------------------- panel b
    from matplotlib.patches import Ellipse
    for d, col, lab in ((pub, BLUE, 'published (arcsinh)'),
                        (lin, ORANGE, 'linear flux units')):
        cen, sig = {}, {}
        for c in ('a2a1', 'a3a2'):
            mu, s, _, _ = mtf.ml_center_and_scatter(d[c].values,
                                                    d['se_' + c].values)
            cen[c], sig[c] = mu, s
        med = {c: np.median((d['se_' + c] / d[c] / np.log(10)).values)
               for c in ('a2a1', 'a3a2')}
        w = 2 * np.hypot(sig['a2a1'], med['a2a1'])
        h = 2 * np.hypot(sig['a3a2'], med['a3a2'])
        axb.add_patch(Ellipse((cen['a2a1'], cen['a3a2']), w, h, fill=False,
                              ec=col, lw=1.6))
        axb.plot(cen['a2a1'], cen['a3a2'], 'D', color=col, ms=7, mew=0,
                 label=lab)
    xl = (-1.05, -0.25)
    axb.plot(xl, xl, color=GREY, lw=1.0, zorder=0)
    axb.set(xlim=xl, ylim=(-0.72, 0.12))
    axb.set_xticks([-1.0, -0.8, -0.6, -0.4])
    axb.set_yticks([-0.6, -0.4, -0.2, 0.0])
    axb.set_xlabel(r'$\log_{10}\,a_2/a_1$')
    axb.set_ylabel(r'$\log_{10}\,a_3/a_2$')
    axb.set_title('Both common shapes stay\nwell inside the prolate half',
                  loc='left')
    axb.text(0.04, 0.34, 'prolate', transform=axb.transAxes, color=GREY,
             va='top')
    axb.text(0.96, 0.06, 'oblate', transform=axb.transAxes, color=GREY,
             ha='right')
    axb.legend(frameon=False, loc='upper left', handletextpad=0.4,
               borderaxespad=0.15, labelspacing=0.3)

    # ---------------------------------------------------------------- panel c
    t = pd.read_csv(os.path.join(
        _ROOT, 'results', 'linear_vs_arcsinh_headline.csv')).set_index('key')
    rows = [
        ('prolateness_sigma', 'prolate significance [$\\sigma$]', '%.2f'),
        ('sigint_a2a1_dex', 'intrinsic scatter $a_2/a_1$ [dex]', '%.3f'),
        ('sigint_a3a2_dex', 'intrinsic scatter $a_3/a_2$ [dex]', '%.3f'),
        ('alpha_common', r'common slope $\alpha$', '%.3f'),
        ('spearman_rho', r'$\rho$(incl, $b_2/b_1$)', '%.3f'),
        ('incl_median_deg', 'median inclination [deg]', '%.1f'),
    ]
    # The five statistics span 0.06 to 74 in their native units, so a shared
    # linear axis would crush all but one against the spine (and a log axis
    # would hide the sign).  Plot each as the RATIO to its published value --
    # published sits at 1.0 by construction -- and print both native values.
    ypos = np.arange(len(rows))[::-1]
    for y, (k, lab, fmt) in zip(ypos, rows):
        a, b = float(t.loc[k, 'published']), float(t.loc[k, 'linear'])
        ratio = b / a
        axc.plot([1.0, ratio], [y, y], color=GREY, lw=1.4, zorder=1)
        axc.plot(1.0, y, 'o', color=BLUE, ms=7, mew=0, zorder=2)
        axc.plot(ratio, y, 'o', color=ORANGE, ms=7, mew=0, zorder=2)
        left, right = (1.0, ratio) if ratio > 1.0 else (ratio, 1.0)
        cl, cr = (BLUE, ORANGE) if ratio > 1.0 else (ORANGE, BLUE)
        axc.annotate(fmt % (a if cl is BLUE else b), (left, y),
                     xytext=(-7, -3), textcoords='offset points',
                     ha='right', color=cl)
        axc.annotate(fmt % (b if cr is ORANGE else a), (right, y),
                     xytext=(7, -3), textcoords='offset points',
                     ha='left', color=cr)
    axc.axvline(1.0, color=BLUE, lw=0.8, ls='--', alpha=0.5, zorder=0)
    axc.set_yticks(ypos)
    axc.set_yticklabels([r[1] for r in rows])
    axc.set_xlim(0.52, 1.72)
    axc.set_xlabel('linear value / published value    '
                   '(numbers are the native values)')
    axc.set_title('Every headline statistic survives; the significance '
                  'weakens and the scatter grows', loc='left')
    axc.grid(axis='x', color='0.92', lw=0.6)
    axc.set_axisbelow(True)

    for ax, L in zip((axa, axb, axc), 'abc'):
        ax.text(-0.14, 1.06, L, transform=ax.transAxes, fontweight='bold',
                fontsize=11, va='bottom')

    outdir = os.path.join(_ROOT, 'results', 'figures')
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, 'linear_vs_arcsinh.png')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    print('n matched windows: %d' % len(m))
    print('wrote %s' % os.path.relpath(png, _ROOT))
    return fig, (axa, axb, axc)


if __name__ == '__main__':
    build()
