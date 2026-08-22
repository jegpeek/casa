"""Figure: shape versus lag scale, and which axis ratio carries the trend.

Two panels sharing a log lag axis.  Left: a3/a2 per window against band centre.
Right: a2/a1, identically scaled, so the eye compares slopes directly.  The
claim the figure has to make true is that the left panel trends down and the
right one does not -- structures get flatter with scale without getting less
elongated.

Usage
-----
python analysis/make_scale_profile_figure.py [--bands PATH] [--out PATH]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE, SMALL, TICK = 8, 7, 6
FOCAL = '#1b4965'     # the trending ratio
FLAT = '#9a6fb0'      # the scale-invariant one
GREY = '#b0b0b0'


def _style():
    plt.rcParams.update({
        'font.size': BASE, 'axes.titlesize': BASE, 'axes.labelsize': BASE,
        'legend.fontsize': SMALL, 'xtick.labelsize': TICK,
        'ytick.labelsize': TICK, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.dpi': 300,
        'savefig.bbox': 'tight', 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    })


def make(bands, summary, out):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True, sharey=True)

    panels = [('a3a2', r'$a_3/a_2$  (short / middle)', FOCAL, 'a3/a2'),
              ('a2a1', r'$a_2/a_1$  (middle / long)', FLAT, 'a2/a1')]

    for ax, (key, ylab, colour, meas) in zip(axes, panels):
        # one thin line per window: the raw evidence, drawn behind everything
        n_win = 0
        for _, g in bands.groupby('chunk'):
            g = g[g.usable & np.isfinite(g[key])].sort_values('r_mid')
            if len(g) < 3:
                continue                       # no slope; excluded from stats too
            ax.plot(g.r_mid, g[key], '-', color=GREY, lw=0.5, alpha=0.55,
                    zorder=1)
            n_win += 1

        # median across windows in each band, the summary the stats act on
        use = bands[bands.usable & np.isfinite(bands[key])]
        med = use.groupby('band').agg(r=('r_mid', 'median'),
                                      v=(key, 'median')).sort_values('r')
        ax.plot(med.r, med.v, 'o-', color=colour, lw=1.8, ms=5,
                mec='white', mew=0.8, zorder=3)

        row = summary[(summary.measure == meas) &
                      (summary.subset == 'all_5_bands')].iloc[0]
        ax.set_ylabel(ylab)
        ax.set_xlabel('lag band centre  (light-years)')
        # Explicit decade-free ticks: the data span 0.019-0.28 ly, so the
        # default log locator crowds four minor labels into half a decade.
        ax.set_xscale('log')
        ax.set_xticks([0.02, 0.05, 0.1, 0.2])
        ax.set_xticklabels(['0.02', '0.05', '0.1', '0.2'])
        ax.set_xticks([], minor=True)
        ax.set_xlim(0.016, 0.34)
        ax.set_ylim(0, 1.05)

        # the headline number, on the panel it belongs to
        p = row.wilcoxon_p
        if p > 0.01:
            ptxt = 'p = %.2f' % p
        else:                       # 7e-08 must not print as 10^-7
            e = int(np.floor(np.log10(p)))
            ptxt = r'p = %.0f$\times10^{%d}$' % (p / 10.0 ** e, e)
        ax.annotate('%+.2f per dex\n%s' % (row.slope_per_dex, ptxt),
                    xy=(0.03, 0.04), xycoords='axes fraction',
                    fontsize=SMALL, color=colour, va='bottom', ha='left')

    axes[0].set_title('The short axis shrinks with scale', loc='left')
    axes[1].set_title('The long-to-middle ratio does not', loc='left')

    # Honesty about the right panel: its median turns up in the widest band, and
    # that upturn is what cancels the trend.  Drop that band and a2/a1 goes to
    # -0.086 per dex, p = 0.046 -- so "scale-invariant" is a statement about the
    # full range, not a robust null.  A reader can see the upturn; say what it does.
    rob = summary[(summary.measure == 'a2/a1') &
                  (summary.subset == 'outermost_dropped')].iloc[0]
    axes[1].annotate('widest band excluded:\n%+.2f per dex, p = %.2f'
                     % (rob.slope_per_dex, rob.wilcoxon_p),
                     xy=(0.97, 0.96), xycoords='axes fraction',
                     fontsize=TICK, color='0.35', va='top', ha='right')
    # Both y-labels stay: the panels share a SCALE, not a quantity, so dropping
    # the right one would leave its marks unidentified.
    axes[1].tick_params(labelleft=True)

    # identity of the two mark types, stated once, in whitespace
    axes[0].plot([], [], '-', color=GREY, lw=0.5, label='one window (n=%d)' % n_win)
    axes[0].plot([], [], 'o-', color=FOCAL, lw=1.8, ms=5, label='median')
    axes[0].legend(frameon=False, loc='upper right', handlelength=1.6)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(out)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bands', default=os.path.join(
        _ROOT, 'results', 'scale_profile_d0.6_s2_bands.csv'))
    ap.add_argument('--summary', default=os.path.join(
        _ROOT, 'results', 'scale_profile_slopes_summary.csv'))
    ap.add_argument('--out', default=os.path.join(
        _ROOT, 'results', 'scale_profile_ratios.png'))
    a = ap.parse_args()
    make(pd.read_csv(a.bands), pd.read_csv(a.summary), a.out)
    print('wrote %s' % a.out)


if __name__ == '__main__':
    raise SystemExit(main())
