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


def ml_center_and_scatter(v, se, grid=np.linspace(0.0, 0.6, 601)):
    """Max-likelihood common log10 value + intrinsic scatter, in dex.

    Returns (mu_dex, sig_int_dex, se_mu_dex, median_meas_se_dex).  The center is
    the likelihood-maximising one, which is what a hypothesis test on the shape
    plane requires -- testing against a non-minimizing center (e.g. the plain
    median) spuriously rejects a common shape.
    """
    l, sl = np.log10(v), se / (v * np.log(10))
    best = None
    for s in grid:
        w = 1.0 / (sl ** 2 + s ** 2)
        mu = np.sum(w * l) / np.sum(w)
        ll = -0.5 * np.sum(w * (l - mu) ** 2) + 0.5 * np.sum(np.log(w))
        if best is None or ll > best[0]:
            best = (ll, mu, s, 1.0 / np.sqrt(np.sum(w)))
    return best[1], best[2], best[3], float(np.median(sl))


def roll_band(a2a1, a3a2, inc_grid, n=4000, seed=31):
    """16/50/84 percentiles of b2/b1 vs inclination at ONE fixed 3D shape.

    Both roll angles are marginalised uniformly; the band is therefore purely
    the roll-angle spread and carries NO measurement error and NO
    window-to-window shape variation.  theta/phi/psi are RADIANS.
    """
    import structure_function as sf
    rng = np.random.default_rng(seed)
    phi, psi = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)

    def one(inc_deg):
        v = []
        for f, s in zip(phi, psi):
            ax2 = sf.principal_axes_2d(sf.params_from_principal_axes(
                1.0, a2a1, a2a1 * a3a2, np.radians(inc_deg), f, s))
            v.append(ax2['b2'] / ax2['b1'] if ax2['b1'] > 0 else np.nan)
        return np.percentile(v, [16, 50, 84])

    return np.array([one(t) for t in inc_grid])


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
    ax.legend(frameon=False, fontsize=6.0, loc='lower left', handlelength=1.2)
    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot(rng, rng, '-', color='0.55', lw=0.9, zorder=2)
    for cx, cy, rx, ry, st in (ell or []):
        ex, ey = log_ellipse(cx, cy, rx, ry)
        if st.get('fill'):
            ax.fill(ex, ey, color=st['color'], alpha=st['fill'], lw=0, zorder=1)
        ax.plot(ex, ey, st.get('ls', '-'), color=st['color'],
                lw=st.get('lw', 1.2), zorder=st.get('z', 2))
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
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    return fig, ax
