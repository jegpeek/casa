"""Figure: shape versus lag scale, and which axis ratio carries the trend.

Two panels sharing a log lag axis.  Left: a3/a2 per window against band centre.
Right: a2/a1, identically scaled, so the eye compares slopes directly.  The
claim the figure has to make true is that the left panel's decline is robust
while the right panel's depends on the lag range included -- structures get
flatter with scale, and whether they also get less elongated is not settled by
these data.

Bands come from the canonical (power-law) per-band fits; see
scale_split.BAND_PROFILES for why the Weibull is not used inside one band.

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
from scipy.stats import wilcoxon

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


def _ptxt(p):
    """Format a p-value without ever rounding a significant one to '0.00'.

    %.2f would print p = 9e-04 as 0.00, which reads as an error rather than as
    strong evidence; and a bare 10^-7 would understate 7e-08 by an order of
    magnitude.  Two decimals above 0.01, mantissa-and-exponent below it.
    """
    if p > 0.01:
        return 'p = %.2f' % p
    e = int(np.floor(np.log10(p)))
    return r'p = %.0f$\times10^{%d}$' % (p / 10.0 ** e, e)


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

        # Median over a MATCHED window set -- only windows usable in all five
        # bands.  The marginal median (a different subset per band) puts a
        # spurious 0.09 dip in a2/a1 at the middle band purely because the
        # windows that drop out there are not a random subset; the matched
        # median is the same quantity the paired statistics actually test.
        use = bands[bands.usable & np.isfinite(bands[key])]
        wide = use.pivot_table(index='chunk', columns='band', values=key)
        matched = wide.dropna()
        n_match = len(matched)
        rmid = use.groupby('band')['r_mid'].median()
        med = pd.DataFrame({'r': rmid.reindex(matched.columns).values,
                            'v': matched.median().values}).sort_values('r')
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
        ax.annotate('%+.2f per dex\n%s' % (row.slope_per_dex,
                                           _ptxt(row.wilcoxon_p)),
                    xy=(0.03, 0.04), xycoords='axes fraction',
                    fontsize=SMALL, color=colour, va='bottom', ha='left')

    axes[0].set_title('The short axis shrinks with scale', loc='left')
    # The right panel's title must not claim scale-invariance.  Under the
    # canonical power-law band fits the full-range a2/a1 slope is null, but
    # dropping the widest band alone takes it to -0.17 per dex at p ~ 9e-04 --
    # the upturn a reader can see in the widest band is doing all the work.
    # State the dependence, not a null the very next annotation contradicts.
    # The right panel must claim neither scale-invariance nor a clean decline.
    # Under the canonical power-law band fits a2/a1 falls significantly across
    # the inner three bands (paired median -0.13, p = 3e-04) and then the
    # windows diverge: from the middle to the widest band exactly 11 of 22 rise
    # and 11 fall, and the paired median change is +0.001 (p = 0.32).  The
    # upturn visible in the plotted line is a difference-of-medians artifact --
    # the band-4 median is 0.14 above the band-2 median while the median of the
    # per-window DIFFERENCES is zero -- so the trace must be captioned or a
    # reader will read a real recovery off it.
    axes[1].set_title('The long-to-middle ratio falls, then diverges',
                      loc='left')

    rob = summary[(summary.measure == 'a2/a1') &
                  (summary.subset == 'outermost_dropped')].iloc[0]
    axes[1].annotate('widest band excluded:\n%+.2f per dex, %s'
                     % (rob.slope_per_dex, _ptxt(rob.wilcoxon_p)),
                     xy=(0.97, 0.96), xycoords='axes fraction',
                     fontsize=TICK, color='0.35', va='top', ha='right')
    # Caption the upturn where it is drawn, so it cannot be read as a recovery.
    # Computed, not hardcoded: this figure regenerates from the tables, and a
    # frozen count would go stale silently the next time the fits change.
    a21 = bands[bands.usable & np.isfinite(bands.a2a1)]
    w21 = a21.pivot_table(index='chunk', columns='band',
                          values='a2a1').dropna()
    b_lo = int(w21.median().idxmin())            # band where the median bottoms
    b_hi = int(max(w21.columns))
    delta = w21[b_hi] - w21[b_lo]
    p_up = wilcoxon(w21[b_lo], w21[b_hi]).pvalue
    r_lo = a21[a21.band == b_lo].r_mid.median()
    axes[1].annotate('beyond %.2f ly the windows split:\n'
                     '%d of %d rise, %d fall (%s)'
                     % (r_lo, int((delta > 0).sum()), len(delta),
                        int((delta < 0).sum()), _ptxt(p_up)),
                     xy=(0.97, 0.06), xycoords='axes fraction',
                     fontsize=TICK, color='0.35', va='bottom', ha='right')
    # Both y-labels stay: the panels share a SCALE, not a quantity, so dropping
    # the right one would leave its marks unidentified.
    axes[1].tick_params(labelleft=True)

    # identity of the two mark types, stated once, in whitespace
    # n_match, not n_win: the heavy line is a matched-subset median, and a bare
    # 'median' label would let a reader assume it runs over every thin line.
    axes[0].plot([], [], '-', color=GREY, lw=0.5, label='one window (n=%d)' % n_win)
    axes[0].plot([], [], 'o-', color=FOCAL, lw=1.8, ms=5,
                 label='median, all-band\nwindows (n=%d)' % n_match)
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
