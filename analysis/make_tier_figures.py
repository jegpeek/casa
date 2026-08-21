"""Both deliverable figures over all 115 windows, layered by SNR tier.

b2/b1 vs inclination, and the a2/a1 - a3/a2 shape plane.  Both log; the two
shape-ratio axes share one range across BOTH figures so the panels can be read
against each other.  Tiers: top quartile in colour (as before), second quartile
mid grey, bottom half light grey, drawn back-to-front so the top quartile is
never occluded.
"""
import os
import numpy as np
import pandas as pd

RCUT, STRIDE, K = 0.1, 2, 4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, RED = '#2c6fbb', '#c0392b'

# back-to-front: faintest first so q4 lands on top
TIERS = [
    ('bottom_half', '#d0d0d0', 'bottom half (SNR 1.4-2.3)', 12, 0.75),
    ('q3',          '#8a8a8a', '2nd quartile (SNR 2.3-3.7)', 15, 0.85),
    ('q4',          BLUE,      'top quartile (SNR 4.0-7.0)', 26, 1.00),
]


def load(k=K):
    """Fit table joined to SNR tier, one row per window (3D power law)."""
    tag = 'r%g_s%d' % (RCUT, STRIDE)
    if k != 2:
        tag += '_k%d' % k
    df = pd.read_csv(os.path.join(ROOT, 'results',
                                  'singleband_powerlaw_%s.csv' % tag))
    d3 = df[(df['mode'] == '3d') & (df.profile == 'powerlaw')].copy()
    snr = pd.read_csv(os.path.join(ROOT, 'results', 'noise_audit_table.csv'))
    q75, q50 = np.percentile(snr.snr, [75, 50])
    snr['tier'] = np.where(snr.snr >= q75, 'q4',
                           np.where(snr.snr >= q50, 'q3', 'bottom_half'))
    out = d3.merge(snr[['row', 'col', 'snr', 'tier']], on=['row', 'col'],
                   how='left')
    return out, (q50, q75)


def usable(d, cols=('a2a1', 'a3a2', 'b2b1', 'incl')):
    """Rows with finite, positive values in every column a log plot needs."""
    m = d.fit_success.astype(bool)
    for c in cols:
        m &= np.isfinite(d[c]) & (d[c] > 0)
        se = 'se_' + c
        if se in d:
            m &= np.isfinite(d[se])
    return m


def shared_ratio_range(d, pad=0.06, floor=None):
    """One [lo, hi] in dex covering a2/a1 AND a3/a2 across all tiers.

    `floor` overrides the data-driven lower bound.  Raising it drops windows
    whose ratio falls below it, so callers must report the visible count
    rather than the fitted count -- see `n_visible`.
    """
    # NOTE the span (and hence the padded upper bound) is computed from the
    # data min, so callers MUST drop collapsed fits first -- see DEGEN in
    # main().  A collapsed fit returns a ratio of ~1e-16, which makes the span
    # ~16 dex and inflates the upper bound by nearly a decade.
    v = np.concatenate([d.a2a1.values, d.a3a2.values])
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = np.log10(v.min()), np.log10(v.max())
    span = hi - lo
    hi = 10 ** (hi + pad * span)
    return (float(floor) if floor is not None else 10 ** (lo - pad * span)), hi


def n_visible(s, rng, cols):
    """Count of rows whose plotted quantities all fall inside `rng`.

    The legend must report what is DRAWN, not what was fitted: raising the
    axis floor silently removes markers, and a legend count taken before the
    clip would overstate the sample.
    """
    m = np.ones(len(s), bool)
    for c in cols:
        m &= (s[c].values >= rng[0]) & (s[c].values <= rng[1])
    return int(m.sum())


def ml_center_and_scatter(v, se, grid=None):
    """Max-likelihood common log10 value + intrinsic scatter, in dex.

    Returns (mu_dex, sig_int_dex, se_mu_dex, median_meas_se_dex).  The center is
    the likelihood-maximising one, which is what a hypothesis test on the shape
    plane requires -- testing against a non-minimizing center (e.g. the plain
    median) spuriously rejects a common shape.

    Thin wrapper over shape_center.ml_center_and_scatter, which is the canonical
    implementation; this module previously carried its own grid-search copy of
    the same estimator, which is exactly how two answers to one question start
    to diverge.  `grid` is accepted and ignored for backward compatibility.
    """
    from shape_center import ml_center_and_scatter as _ml, weighted_center

    l, sl = np.log10(v), se / (v * np.log(10))
    mu, sig = _ml(l, sl)
    _, se_mu = weighted_center(l, sl, sig)
    return mu, sig, se_mu, float(np.median(sl))


def roll_band(a2a1, a3a2, inc_grid, n=4000, seed=31):
    """16/50/84 percentiles of b2/b1 vs inclination at ONE fixed 3D shape.

    Both roll angles are marginalised uniformly; the band is therefore purely
    the roll-angle spread and carries NO measurement error and NO
    window-to-window shape variation.  theta/phi/psi are RADIANS.

    Returns an (len(inc_grid), 3) array of [16, 50, 84] percentiles.  Delegates
    to slicing_model.roll_band, which is the canonical implementation and is
    pinned by tests/test_slicing_model.py.
    """
    from slicing_model import roll_band as _rb

    lo, mid, hi = _rb(a2a1, a3a2, np.atleast_1d(inc_grid), n=n, seed=seed)
    return np.column_stack([lo, mid, hi])


def log_ellipse(cx, cy, rx, ry, n=241):
    """Axis-aligned ellipse in log10 space, returned in linear coordinates.

    Axis-aligned because the within-window covariance of se_a2a1 and se_a3a2 is
    not recoverable from the stored fits, so a tilt would not be justified.
    """
    t = np.linspace(0, 2 * np.pi, n)
    return 10 ** (cx + rx * np.cos(t)), 10 ** (cy + ry * np.sin(t))


def _clip_lower(ax, x, y, ylo, **kw):
    """Symmetric log-space error bars go negative for large SE; clip at floor.

    Drawing a symmetric bar y +/- se on a log axis is invalid once se >= y.
    Rather than dropping those windows or faking one-sided limits, the lower
    end is clipped at the axis floor so every marker stays visible.
    """
    return kw


def fig_inclination(d, rng, shape=None, out='b2b1_vs_inclination_all115.png'):
    """b2/b1 vs inclination, all tiers, log b2/b1 with the slicing curve.

    `shape` is (a2a1, a3a2) for the reference 3D shape; when given, the
    roll-angle 16-84% band is drawn behind the points.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    if shape is not None:
        g = np.linspace(0.0, 90.0, 91)
        B = roll_band(shape[0], shape[1], g)
        obs_lo = d.incl.min()
        ax.fill_between(g, B[:, 0], B[:, 2], color=BLUE, alpha=0.15, lw=0,
                        zorder=1)
        ax.plot(g, B[:, 1], '-', color=BLUE, lw=1.3, zorder=2)
        un = g <= obs_lo
        if un.sum() > 1:
            ax.axvspan(0, obs_lo, color='0.90', zorder=0)
    for tier, col, lab, sz, al in TIERS:
        s = d[d.tier == tier]
        m = usable(s)
        s = s[m]
        if not len(s):
            continue
        s = s[s.b2b1 >= rng[0]]
        if not len(s):
            continue
        lo = np.maximum(s.b2b1 - s.se_b2b1, rng[0] * 1.001)
        # inclination is physically confined to [0, 90]; 10 windows have
        # se_incl exceeding that whole range (the axis direction is simply
        # unconstrained there), so clip the bars at the physical bounds
        # rather than letting them set the x limits.
        xl = np.clip(s.incl - s.se_incl, 0.0, 90.0)
        xh = np.clip(s.incl + s.se_incl, 0.0, 90.0)
        ax.errorbar(s.incl, s.b2b1,
                    yerr=[s.b2b1 - lo, s.se_b2b1],
                    xerr=[s.incl - xl, xh - s.incl],
                    fmt='o', ms=np.sqrt(sz), color=col, alpha=al,
                    ecolor=col, elinewidth=0.7, capsize=0, mew=0,
                    label='%s  (n=%d)' % (lab, n_visible(s, rng, ['b2b1'])),
                    zorder=3 + al * 3)
    ax.set_xlabel('inclination of the long axis to the echo plane  [deg]')
    ax.set_ylabel(r'in-plane lag ratio at fixed $S_2$   $b_2/b_1$')
    ax.set_yscale('log')
    ax.set_ylim(*rng)
    ax.set_xlim(0.0, 90.0)

    # Legend goes ABOVE the axes: inside, it lands on the low-b2/b1 points at
    # high inclination, which are exactly the ones the slicing curve is being
    # judged against.  The band gets its own entry because its width is roll
    # spread ONLY -- no measurement error -- and a reader will otherwise take it
    # for a confidence band.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple

    h, l = ax.get_legend_handles_labels()
    if shape is not None:
        h = list(h) + [(Line2D([], [], color=BLUE, lw=1.3),
                        Patch(facecolor=BLUE, alpha=0.15, lw=0))]
        l = list(l) + ['one fixed 3D shape ($a_2/a_1$=%.2f, $a_3/a_2$=%.2f):\n'
                       'median and 16-84%% roll spread (no measurement error)'
                       % (shape[0], shape[1])]
    fig.set_size_inches(5.0, 4.6)
    fig.legend(h, l, frameon=False, fontsize=6.0, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.4,
               handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=300)
    return fig, ax


def fig_shape_plane(d, rng, ell=None, out='shape_plane_all115.png'):
    """a3/a2 against a2/a1, log-log, equal spans so the diagonal is 45 deg.

    `ell` is a list of (cx, cy, rx, ry, style) in dex.  Only ONE of these is
    directly comparable with the plotted points: measurement (+) intrinsic,
    whose semi-axes add the median measurement SE and the intrinsic scatter in
    quadrature.  The other two are its nested components and must be labeled as
    such -- measurement-only is the null (one exact shape blurred by
    measurement), and intrinsic-only describes the spread of the TRUE shapes,
    which are not the quantities plotted, so it will under-cover by
    construction.
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    # Tall figure: the legend needs six entries with two-line captions, so it
    # goes BELOW the axes rather than covering the low-ratio corner where the
    # oblate outliers live.
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot(rng, rng, '-', color='0.55', lw=0.9, zorder=2)
    ell_handles = []
    for cx, cy, rx, ry, st in (ell or []):
        ex, ey = log_ellipse(cx, cy, rx, ry)
        if st.get('fill'):
            ax.fill(ex, ey, color=st['color'], alpha=st['fill'], lw=0, zorder=1)
        ax.plot(ex, ey, st.get('ls', '-'), color=st['color'],
                lw=st.get('lw', 1.2), zorder=st.get('z', 2))
        if st.get('label'):
            ell_handles.append(mlines.Line2D(
                [], [], color=st['color'], ls=st.get('ls', '-'),
                lw=st.get('lw', 1.2),
                marker='D' if st.get('center_marker') else None,
                ms=5.5, mew=0, label=st['label']))
    if ell:
        cx, cy = ell[0][0], ell[0][1]
        ax.plot([10 ** cx], [10 ** cy], 'D', color=BLUE, ms=5.5, mew=0,
                zorder=9)
    for tier, col, lab, sz, al in TIERS:
        s = d[d.tier == tier]
        s = s[usable(s)]
        if not len(s):
            continue
        # a raised axis floor can put a marker off-scale; drop those rows
        # rather than emit a negative bar length (matplotlib raises), and
        # keep the legend count equal to what is drawn.
        s = s[(s.a2a1 >= rng[0]) & (s.a3a2 >= rng[0])]
        if not len(s):
            continue
        xlo = np.maximum(s.a2a1 - s.se_a2a1, rng[0] * 1.001)
        ylo = np.maximum(s.a3a2 - s.se_a3a2, rng[0] * 1.001)
        ax.errorbar(s.a2a1, s.a3a2,
                    xerr=[s.a2a1 - xlo, s.se_a2a1],
                    yerr=[s.a3a2 - ylo, s.se_a3a2],
                    fmt='o', ms=np.sqrt(sz), color=col, alpha=al,
                    ecolor=col, elinewidth=0.7, capsize=0, mew=0,
                    label='%s  (n=%d)' % (lab, n_visible(s, rng, ['a2a1', 'a3a2'])),
                    zorder=3 + al * 3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*rng)
    ax.set_ylim(*rng)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$a_2/a_1$   (lag ratio at fixed $S_2$)')
    ax.set_ylabel(r'$a_3/a_2$   (lag ratio at fixed $S_2$)')

    # Which side of the diagonal means what.  Placed inside the axes on the
    # side each regime occupies, so the diagonal needs no separate legend key.
    ax.text(0.04, 0.42, 'prolate\n' r'$a_3/a_2 > a_2/a_1$', transform=ax.transAxes,
            ha='left', va='center', fontsize=8.5, color='0.35')
    ax.text(0.96, 0.10, 'oblate\n' r'$a_3/a_2 < a_2/a_1$', transform=ax.transAxes,
            ha='right', va='center', fontsize=8.5, color='0.35')

    h, l = ax.get_legend_handles_labels()
    fig.legend(h + ell_handles, l + [x.get_label() for x in ell_handles],
               frameon=False, fontsize=8.0, loc='lower center',
               bbox_to_anchor=(0.5, -0.005), handlelength=1.6,
               labelspacing=0.9, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.26, 1, 1))
    fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax

# --- entry point --------------------------------------------------------------
# Defaults reproduce the two committed deliverable figures exactly: k=4 table,
# shared log ratio range with the hand-set 0.08 floor (below which a2/a1 and
# a3/a2 are not trusted), and the ML common shape as the reference for the
# slicing band.  NOTE the fits always use the FULL top-quartile sample; the
# floor clips only what is DRAWN, and the legend counts follow the drawing.
FLOOR = 0.08
DEGEN = 0.02          # ratios below this are collapsed fits, not measurements


def apply_figure_style(sizes=(8, 7, 6)):
    """Publication text sizes: (axes title, axis label, tick/legend)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': sizes[2],
        'axes.titlesize': sizes[0],
        'axes.labelsize': sizes[1],
        'xtick.labelsize': sizes[2],
        'ytick.labelsize': sizes[2],
        'legend.fontsize': sizes[2],
    })

def main(k=K, floor=FLOOR, outdir=None):
    import matplotlib
    matplotlib.use('Agg')
    outdir = outdir or os.path.join(ROOT, 'results', 'figures')
    os.makedirs(outdir, exist_ok=True)

    apply_figure_style()

    d, _ = load(k=k)
    d = d[usable(d)]

    # Drop collapsed fits BEFORE ranging.  A degenerate fit returns a ratio of
    # ~1e-16; it is off-scale on any sane axis, but if it reaches
    # shared_ratio_range it sets the span and inflates the upper bound by
    # nearly a decade.  This is the cut used for the published figures.
    d = d[(d.a2a1 >= DEGEN) & (d.a3a2 >= DEGEN)].copy()

    rng = shared_ratio_range(d, floor=floor)
    q4 = d[d.tier == 'q4']
    mu21, s21, _, me21 = ml_center_and_scatter(q4.a2a1.values, q4.se_a2a1.values)
    mu32, s32, _, me32 = ml_center_and_scatter(q4.a3a2.values, q4.se_a3a2.values)
    c21, c32 = 10 ** mu21, 10 ** mu32

    # Three nested 1-sigma ellipses, all on the ML center.  Only the OUTER one
    # (measurement + intrinsic in quadrature) is comparable with the plotted
    # points; the other two are its components.  See fig_shape_plane's docstring
    # and results/shape_ellipses_three_k4.csv, which tabulates the coverage of
    # each against the 39.35% a 2D 1-sigma contour should hold.
    # Coverage is measured on the SAME sample the ellipses are drawn over (the
    # top quartile), so the legend reports the actual fraction inside rather
    # than a remembered number.  A 2D 1-sigma contour should hold 39.35%.
    EXPECTED = 39.35

    def _inside(rx, ry):
        dx = (np.log10(q4.a2a1.values) - mu21) / rx
        dy = (np.log10(q4.a3a2.values) - mu32) / ry
        return 100.0 * np.mean(dx ** 2 + dy ** 2 <= 1.0)

    r21, r32 = np.hypot(me21, s21), np.hypot(me32, s32)
    ells = [
        (mu21, mu32, r21, r32,
         dict(color=BLUE, lw=1.5, fill=0.10, z=3, center_marker=True,
              label=('center, and measurement $\\oplus$ intrinsic\n'
                     '(the only one comparable with the points: %.0f%% inside)'
                     % _inside(r21, r32)))),
        (mu21, mu32, me21, me32,
         dict(color=BLUE, lw=1.5, ls='--', z=3,
              label=('measurement only \u2014 the null of one exact shape\n'
                     '(%.0f%% inside, %.0f%% expected: the null fails)'
                     % (_inside(me21, me32), EXPECTED)))),
        (mu21, mu32, s21, s32,
         dict(color=BLUE, lw=1.8, ls=':', z=3,
              label=('intrinsic only \u2014 spread of the TRUE shapes,\n'
                     'not of the plotted points'))),
    ]

    f1 = os.path.join(outdir, 'b2b1_vs_inclination_all115.png')
    f2 = os.path.join(outdir, 'shape_plane_all115.png')
    fig_inclination(d, rng, shape=(c21, c32), out=f1)
    fig_shape_plane(d, rng, ell=ells, out=f2)

    print('k=%d  n_drawn=%d  range=[%.3f, %.3f]' % (k, len(d), rng[0], rng[1]))
    print('common shape  a2/a1=%.4f  a3/a2=%.4f' % (c21, c32))
    print('intrinsic sig %.3f, %.3f dex | median meas SE %.3f, %.3f dex'
          % (s21, s32, me21, me32))
    print('wrote %s\n      %s' % (f1, f2))
    return f1, f2


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--k', type=int, default=K, help='jackknife blocking (4 = deliverable)')
    ap.add_argument('--floor', type=float, default=FLOOR, help='lower ratio-axis floor')
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()
    main(k=a.k, floor=a.floor, outdir=a.outdir)
