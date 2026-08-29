"""Science-format version of the shape plane (shape_plane_all115.png).

Same data, same estimator, same 0.08 axis floor as
`make_tier_figures.fig_shape_plane` -- this module changes only the PRESENTATION
so the panel meets the Science family's figure specification and reads at final
printed size.  Everything quantitative is imported from make_tier_figures, so
the two figures cannot drift apart:

    load, usable, shared_ratio_range, n_visible, ml_center_and_scatter,
    log_ellipse, TIERS, FLOOR, DEGEN

What changes, and why:

1. TWO ellipses, not three.  The intrinsic-only contour is dropped on the
   project's standing decision: it describes the spread of the TRUE shapes while
   every plotted point additionally carries measurement error, so it
   under-covers by construction and a reader who compares it to the points the
   way they compare the other two concludes the intrinsic scatter is too small
   -- the opposite of what the fit found.  sigma_int is quoted in the caption
   instead, where it cannot be mistaken for a contour the points should fill.

2. Science page geometry.  Width is fixed to one of the three allowed column
   widths (5.7 / 12.1 / 18.4 cm); this figure uses the 2-column 12.1 cm, since a
   5.7 cm square panel cannot carry 94 markers with error bars.  Lettering is
   ~7 pt at final size and never below 5 pt, symbols >= 6 pt, all strokes
   >= 0.5 pt, and there are no minor ticks and no grid lines.

3. Vector primary output.  A PDF is the deliverable; the PNG is a 600 dpi
   raster for on-screen checking only.

4. The long explanatory parentheticals move out of the legend and into
   `shape_plane_science_caption.txt`, which this script writes with the numbers
   COMPUTED, not transcribed.  The caption file is the companion deliverable --
   the coverage fractions, sigma_int, the drawn-vs-fitted counts and the
   off-scale window name all come out of the same arrays that are plotted.

5. Ratio ticks are plain decimals (0.1, 0.2, 0.5, 1.0) rather than powers of
   ten.  These are axis ratios of order unity; 10^-1 makes a reader do
   arithmetic to see that a point sits at one tenth.

STANDING RULE inherited from make_tier_figures: the FITS always use the full
usable top-quartile sample; the 0.08 floor clips only what is DRAWN, and every
count in the legend and caption follows the drawing.  Both counts are stated in
the caption so a referee tallying markers finds no discrepancy.
"""
import os
import numpy as np
import matplotlib.patheffects as pe

import make_tier_figures as mtf

ROOT = mtf.ROOT
BLUE = mtf.BLUE

# Science family column widths, cm -> inches.  Only these three are accepted by
# the journal; anything else gets rescaled at production and the point sizes
# stop meaning what they say here.
SCIENCE_WIDTH_CM = {1: 5.7, 2: 12.1, 3: 18.4}
CM = 1 / 2.54

# Font ladder at FINAL size (§5.2 role-mapped, three sizes max).  Science asks
# for ~7 pt lettering and sets a hard 5 pt floor, so the smallest role here is
# 6 pt with a full point of headroom.
SIZES = dict(label=7.0, annot=6.0, tick=6.0)

# A 2D one-sigma contour contains this fraction of a bivariate normal.  Quoted
# so the measurement-only coverage can be read as a null test rather than as an
# unanchored percentage.
EXPECTED = 39.35


def apply_science_style():
    """rcParams for a Science-family figure at final printed size.

    Helvetica is the journal's preferred sans; the fallback chain ends at
    DejaVu Sans so the script still renders on a machine without it rather than
    silently substituting a serif.  Minor ticks and grids are switched OFF
    globally here because the specification forbids them, not as a taste call.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'Helvetica Neue',
                            'DejaVu Sans'],
        # Mathtext must resolve to the SAME family as the surrounding text.
        # Left at the default it renders in DejaVu Sans, so "$a_2/a_1$" and the
        # words beside it in one axis label come out in two different
        # typefaces -- visible at print size and a spec violation.
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Helvetica:normal',
        'mathtext.it': 'Helvetica:italic',
        'mathtext.bf': 'Helvetica:bold',
        'mathtext.default': 'it',
        'font.size': SIZES['tick'],
        'axes.labelsize': SIZES['label'],
        'axes.titlesize': SIZES['label'],
        'xtick.labelsize': SIZES['tick'],
        'ytick.labelsize': SIZES['tick'],
        'legend.fontsize': SIZES['annot'],
        # every stroke at or above the 0.5 pt reproduction floor
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'axes.grid': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 2.4,
        'ytick.major.size': 2.4,
        # keep text as text in the PDF so production can restyle it
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.bbox': None,
    })


def ratio_ticks(rng):
    """Plain-decimal ticks inside `rng` for an axis-ratio log axis.

    Powers of ten are the wrong labelling for quantities of order unity: a
    reader should see 0.1 and 1.0, not 10^-1 and 10^0.
    """
    cand = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    return cand[(cand >= rng[0]) & (cand <= rng[1])]


def shape_plane_data(k=mtf.K, floor=mtf.FLOOR, variant=None):
    """Everything the figure and its caption need, computed once.

    Returns a dict.  Three top-quartile counts are carried explicitly and are
    NOT interchangeable: `n_q4_fit` is the full usable quartile the ML center
    and scatter are estimated from; `n_q4_cov` is that quartile minus collapsed
    (degenerate) fits, and is the coverage denominator; the drawn count is
    `n_q4_cov` minus whatever the display floor pushes off-scale.

    Coverage is deliberately measured on the non-degenerate sample rather than
    on the drawn sample, so that retuning the display floor cannot move a
    quoted statistic (the project's standing rule).  It is therefore NOT a
    tally of markers a reader can see inside the contour: under the raw-flux
    variant one non-degenerate window sits below the floor and is undrawn but
    still counted.  The caption states the count and denominator explicitly,
    and names the off-scale windows, so the difference is accountable.

    `variant` selects the preprocessing exactly as in `mtf.load`: '_linear' is
    the raw-flux run and the default, '' the original arcsinh run; None resolves
    via `mtf.default_variant()`.  Every number in the caption is derived from
    this table, so the caption follows the variant automatically and cannot
    drift from the figure it describes.
    """
    if variant is None:
        variant = mtf.default_variant()
    raw = mtf.load(k=k, variant=variant)[0]
    d_all = raw[mtf.usable(raw)]
    n_fitted, n_usable = int(len(raw)), int(len(d_all))

    # Collapsed fits (ratio ~1e-16) are kept in the likelihood -- their huge SEs
    # give them almost no weight, and dropping them would be a cut on the fitted
    # value itself -- but excluded from the drawn set and from the axis-range
    # calculation, which takes its span from the data minimum.
    d = d_all[(d_all.a2a1 >= mtf.DEGEN) & (d_all.a3a2 >= mtf.DEGEN)].copy()
    rng = mtf.shared_ratio_range(d, floor=floor)
    # PRINT VERSION ONLY: stop the panel at exactly 1.  Both ratios are <= 1 by
    # construction (a1 >= a2 >= a3), so the screen version's headroom above
    # unity is unphysical space, and clipping it makes the frame itself carry
    # the physics -- the top spine IS the a3 = a2 (prolate) locus, the right
    # spine IS the a2 = a1 (oblate) locus, and their corner is the sphere.  This
    # hides no drawn point (max ratios 0.92, 0.92); it does truncate the upper
    # arm of 9 x- and 15 y-error bars at the physical limit, which is stated in
    # the caption.  The data minimum still sets the lower end.
    rng = (rng[0], 1.0)

    q4_fit = d_all[d_all.tier == 'q4']
    mu21, s21, _, me21 = mtf.ml_center_and_scatter(q4_fit.a2a1.values,
                                                   q4_fit.se_a2a1.values)
    mu32, s32, _, me32 = mtf.ml_center_and_scatter(q4_fit.a3a2.values,
                                                   q4_fit.se_a3a2.values)
    # Coverage denominator: the top quartile MINUS collapsed fits (`d` is the
    # degen-filtered table, not the floor-clipped one), so a retune of the
    # display floor cannot move a quoted statistic -- the project's standing
    # rule.  Note this is one window short of `q4_fit` under raw flux, where a
    # collapsed fit is dropped here but kept in the likelihood above.  The
    # caption reports count and denominator explicitly rather than a bare
    # percentage, and names the off-scale windows, so a referee tallying
    # markers can account for the difference.
    q4_cov = d[d.tier == 'q4']

    def inside(rx, ry):
        dx = (np.log10(q4_cov.a2a1.values) - mu21) / rx
        dy = (np.log10(q4_cov.a3a2.values) - mu32) / ry
        n = int(np.sum(dx ** 2 + dy ** 2 <= 1.0))
        return n, len(q4_cov), 100.0 * n / len(q4_cov)

    # measurement (+) intrinsic, added in quadrature: the only contour directly
    # comparable with the plotted points, since the points carry both.
    r21, r32 = float(np.hypot(me21, s21)), float(np.hypot(me32, s32))

    # Windows the floor pushes off-scale, named so a referee counting markers
    # against the quoted n finds the difference accounted for.
    off = d[(d.a2a1 < rng[0]) | (d.a3a2 < rng[0])]
    off_names = ['r%d_c%d' % (r.row, r.col) for r in off.itertuples()]
    off_q4 = ['r%d_c%d' % (r.row, r.col)
              for r in off[off.tier == 'q4'].itertuples()]

    return dict(
        d=d, rng=rng, mu=(mu21, mu32), sig_int=(s21, s32),
        med_se=(me21, me32), r_tot=(r21, r32),
        n_fitted=n_fitted, n_usable=n_usable,
        n_q4_fit=int(len(q4_fit)), n_q4_cov=int(len(q4_cov)),
        cov_tot=inside(r21, r32), cov_meas=inside(me21, me32),
        off_names=off_names, off_q4=off_q4, floor=float(rng[0]), k=k)


def fig_shape_plane_science(S, ncol=2):
    """The panel.  `S` is the dict from shape_plane_data.

    Layout: one square axes (equal aspect is REQUIRED -- the diagonal separating
    prolate from oblate is only at 45 deg if the two log spans are equal and the
    aspect is 1), with the legend in a band below rather than inside the frame.
    The lower-left corner is the only interior whitespace large enough for a
    five-entry key, and the prolate/oblate diagonal runs straight through it, so
    a frameless legend there would sit on top of a line.
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.ticker import FixedLocator, FuncFormatter

    rng = S['rng']
    mu21, mu32 = S['mu']

    # Geometry is solved in INCHES, not in axes fractions.  The panel must be
    # square (see docstring) and the figure must be exactly one Science column
    # width, so the only free quantity is the figure height -- fixing fractions
    # by hand instead leaves either a non-square panel or a band of dead space
    # under the axis label.
    w = SCIENCE_WIDTH_CM[ncol] * CM
    # Right and top margins must now clear the two edge labels (rotated
    # "oblate (a2 = a1)" on the right, "prolate (a3 = a2)" above); the x-label
    # band shrank because the label is now the bare ratio on one line.
    L, R, T = 0.44, 0.44, 0.18          # left / right / top margins, inches
    # XLAB must clear the descender of the mathtext subscript in "$a_2/a_1$",
    # which hangs ~0.01 in below the text baseline box.
    XLAB, LEGH = 0.32, 0.0              # x-label band; legend now sits INSIDE
    side = w - L - R                    # square panel side
    h = T + side + XLAB + LEGH
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([L / w, (XLAB + LEGH) / h, side / w, side / h])

    # 1:1 line: the prolate/oblate divide.  Identified by the two regime labels
    # flanking it, so it needs no legend entry.
    ax.plot(rng, rng, '-', color='0.62', lw=0.6, zorder=2)

    # --- ellipses (two, see module docstring) --------------------------------
    ell = [
        # Labels name what each contour IS, in as few words as fit the column;
        # the caption carries which one is comparable with the points and what
        # each coverage fraction tests.
        (S['r_tot'], dict(ls='-', lw=1.0, fill=0.10,
                          label='measurement $\\oplus$ intrinsic')),
        (S['med_se'], dict(ls=(0, (3.2, 1.6)), lw=0.9, fill=None,
                           label='measurement only (null)')),
    ]
    handles = []
    for (rx, ry), st in ell:
        ex, ey = mtf.log_ellipse(mu21, mu32, rx, ry)
        if st['fill']:
            ax.fill(ex, ey, color=BLUE, alpha=st['fill'], lw=0, zorder=1)
        ax.plot(ex, ey, ls=st['ls'], color=BLUE, lw=st['lw'], zorder=4)
        handles.append(mlines.Line2D([], [], color=BLUE, ls=st['ls'],
                                     lw=st['lw'], label=st['label']))
    ax.plot([10 ** mu21], [10 ** mu32], 'D', color=BLUE, ms=3.4, mew=0,
            zorder=9)
    handles.append(mlines.Line2D([], [], color=BLUE, ls='none', marker='D',
                                 ms=3.4, mew=0, label='common shape (ML)'))

    # --- points, faintest tier first so the top quartile is never occluded ---
    pt_handles = []
    for tier, col, _lab, _sz, al in mtf.TIERS:
        s = S['d'][S['d'].tier == tier]
        s = s[mtf.usable(s)]
        s = s[(s.a2a1 >= rng[0]) & (s.a3a2 >= rng[0])]
        if not len(s):
            continue
        # A symmetric bar in log space goes negative once SE >= value; clip the
        # lower end at the floor so the marker stays visible instead of raising.
        xlo = np.maximum(s.a2a1 - s.se_a2a1, rng[0] * 1.001)
        ylo = np.maximum(s.a3a2 - s.se_a3a2, rng[0] * 1.001)
        # Focal dominance (figure-style §4.2): the lower two tiers carry ~3x the
        # top quartile's error-bar ink simply because there are 66 of them with
        # comparable fractional errors, which buries the tier every quantitative
        # claim rests on.  The bars are NOT dropped -- their length is why those
        # windows scatter, and hiding it would make them read as precise
        # discrepant measurements -- but they are drawn at reduced opacity, with
        # the markers at full tier opacity so identity survives.  Every stroke
        # stays at or above the 0.5 pt reproduction floor; opacity, not width,
        # does the recession.
        focal = tier == 'q4'
        ms = 2.6 if focal else 2.0
        ax.errorbar(s.a2a1, s.a3a2,
                    xerr=[s.a2a1 - xlo, s.se_a2a1],
                    yerr=[s.a3a2 - ylo, s.se_a3a2],
                    fmt='none', ecolor=col, alpha=al if focal else al * 0.45,
                    elinewidth=0.5, capsize=0,
                    zorder=3 + al * 3)
        ax.plot(s.a2a1, s.a3a2, 'o', ms=ms, color=col, alpha=al, mew=0,
                zorder=3.5 + al * 3)
        pt_handles.append(mlines.Line2D(
            [], [], color=col, ls='none', marker='o', ms=ms, mew=0, alpha=al,
            label='%s (%s = %d)' % (SCIENCE_TIER_LABEL[tier], '$n$',
                                    mtf.n_visible(s, rng, ['a2a1', 'a3a2']))))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*rng)
    ax.set_ylim(*rng)
    ax.set_aspect('equal')
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(FixedLocator(ratio_ticks(rng)))
        a.set_minor_locator(FixedLocator([]))
        a.set_major_formatter(FuncFormatter(lambda v, _p: ('%g' % v)))
    # Bare ratios: the parenthetical gloss ("lag ratio at fixed S_2") moves to
    # the caption at the user's request.  The caption must therefore keep the
    # sentence establishing that these are lag lengths at fixed S_2 and hence
    # independent of the structure-function slope -- it is now the only place
    # that is said.
    ax.set_xlabel(r'$a_2/a_1$')
    ax.set_ylabel(r'$a_3/a_2$')

    # Which side of the diagonal is which.  Inside the axes, on the side each
    # regime occupies, so the divide needs no key.
    # Regime labels annotate the EDGES, not corners: prolate is the degenerate
    # limit a3 = a2, which is the whole top spine, and oblate is a2 = a1, the
    # whole right spine -- a corner label would imply the regime lives only
    # there.  Both sit just outside the frame, centred on their spine, so they
    # cannot land on data; the isotropic label goes inside its corner because
    # outside it would collide with the other two, and the corner interior is
    # empty (no window has both ratios above 0.80).
    ax.text(0.5, 1.012, 'prolate  ($a_3 = a_2$)', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=SIZES['annot'], color='0.30')
    ax.text(1.012, 0.5, 'oblate  ($a_2 = a_1$)', transform=ax.transAxes,
            ha='center', va='bottom', rotation=270, rotation_mode='anchor',
            fontsize=SIZES['annot'], color='0.30')
    ax.text(0.978, 0.972, 'isotropic', transform=ax.transAxes, ha='right',
            va='top', fontsize=SIZES['annot'], color='0.30')

    # --- legend inside the panel, bottom centre ------------------------------
    # Moved in from below the frame to save ~1.3 cm of column height.  The
    # bottom band holds no markers (nothing has a3/a2 below the band except
    # five lower-tier windows at the far left and right) and the 1:1 diagonal
    # passes well above it, but 14 error bars do cross it -- three of them top
    # quartile, whose lower arms run to the axis floor.  An OPAQUE patch would
    # make those bars appear to terminate at the legend edge, inventing a bar
    # end where the real signal is "uncertainty exceeds the value", so the
    # legend stays frameless and the bars read through behind it.  Verified
    # legible at 12.1 cm print size.
    # y offset clears the single lowest marker (a lower-tier window at axes
    # fraction y = 0.019) rather than covering it; checked programmatically,
    # not eyeballed.
    leg = ax.legend(pt_handles + handles,
                    [h.get_label() for h in pt_handles + handles],
                    loc='lower center', bbox_to_anchor=(0.5, 0.035),
                    ncol=2, frameon=False, fontsize=SIZES['annot'],
                    handlelength=1.7, handletextpad=0.6,
                    columnspacing=1.1, labelspacing=0.34,
                    borderaxespad=0.0)
    # 13 error bars pass behind the legend.  Two fixes, both needed: the tier
    # zorders run to ~6.5, above a legend's default of 5, so a top-quartile bar
    # was drawing straight OVER the words -- the legend is lifted clear of every
    # data artist.  And a white halo on the glyphs (not a filled frame) keeps
    # the text readable where a bar passes close, while leaving the bars
    # continuous: an opaque patch would break three top-quartile bars at the
    # legend edge and read as a bar end, when the real signal is that their
    # uncertainty runs to the axis floor.
    leg.set_zorder(20)
    for t in leg.get_texts():
        t.set_path_effects([pe.withStroke(linewidth=1.6, foreground='white')])
    return fig, ax, leg


# Tier identity is floor, not budget (§2.1), but the S/N ranges that the screen
# version spelled out here move to the caption -- at 6 pt in a two-column
# legend they are what pushes the band past the figure edge.
SCIENCE_TIER_LABEL = {
    'q4': 'top S/N quartile',
    'q3': 'second quartile',
    'bottom_half': 'lower half',
}


def caption(S):
    """Caption text with every number computed from the plotted arrays.

    Written to disk beside the figure so the manuscript can paste it rather than
    re-deriving values by hand -- the failure mode this avoids is a caption that
    quotes a coverage or a sigma_int from an earlier run of the fit.
    """
    s21, s32 = S['sig_int']
    me21, me32 = S['med_se']
    r21, r32 = S['r_tot']
    c21, c32 = 10 ** S['mu'][0], 10 ** S['mu'][1]
    n_draw = sum(mtf.n_visible(S['d'][S['d'].tier == t], S['rng'],
                               ['a2a1', 'a3a2'])
                 for t, _c, _l, _s, _a in mtf.TIERS)

    # S/N ranges per tier, computed from the drawn sample so the caption cannot
    # quote a boundary the figure does not show.
    snr_txt = '; '.join(
        '%s %.1f\u2013%.1f' % (SCIENCE_TIER_LABEL[t],
                               S['d'][S['d'].tier == t].snr.min(),
                               S['d'][S['d'].tier == t].snr.max())
        for t, _c, _l, _s, _a in reversed(mtf.TIERS)
        if len(S['d'][S['d'].tier == t]))

    off = ''
    if S['off_q4']:
        off = (' One highest-quartile window (%s) falls below the axis floor '
               'and is not drawn; it is retained in the fit.'
               % ', '.join(S['off_q4']))
    n_off_other = len(S['off_names']) - len(S['off_q4'])
    if n_off_other:
        off += (' %d further window%s from the lower tiers fall%s below the '
                'floor.' % (n_off_other, 's' if n_off_other != 1 else '',
                            '' if n_off_other != 1 else 's'))

    return (
        'Fig. X. Three-dimensional shapes of the Mab light-echo structure are '
        'consistent with a single common shape.\n\n'
        'Each point is one independent sky window, fitted separately: the '
        'ellipsoidal second-order structure function of the reconstructed '
        'three-dimensional emissivity gives three principal axes '
        '$a_1 \\ge a_2 \\ge a_3$, plotted as the two independent axis ratios. '
        'Both ratios are lengths of the lag vector at fixed $S_2$, so they are '
        'independent of the structure-function slope. Error bars are '
        '$\\pm1\\sigma$ from block jackknife ($k = %d$); on a logarithmic axis '
        'a symmetric bar reaches the axis floor where the uncertainty exceeds '
        'the value. Both axes stop at unity, the physical limit of each ratio, '
        'so the top and right edges are the prolate ($a_3 = a_2$) and oblate '
        '($a_2 = a_1$) degenerate loci and their corner is a sphere; error bars '
        'reaching the limit are truncated there. Windows are split at the '
        'quartiles of their echo '
        'signal-to-noise ratio (%s) and drawn back to front, so the '
        'best-measured '
        'quartile is never hidden. The grey diagonal is $a_2/a_1 = a_3/a_2$, '
        'on which the two ratios are equal: windows above it are the more '
        'prolate (cigar-like), those below the more oblate (pancake-like), '
        'while the marked edges are the limits at which an ellipsoid becomes '
        'exactly prolate or exactly oblate.\n\n'
        'The blue diamond is the maximum-likelihood common shape of the '
        'highest signal-to-noise quartile, $a_2/a_1 = %.2f$, '
        '$a_3/a_2 = %.2f$. The dashed ellipse is the null hypothesis that '
        'every window has exactly this one shape and the observed spread is '
        'measurement error alone; its semi-axes are the median measurement '
        'uncertainties (%.3f and %.3f dex) and it contains %d of %d windows '
        'in that quartile (%.0f%%) against the %.1f%% a two-dimensional '
        '$1\\sigma$ contour expects, so the null is rejected. The solid '
        'filled ellipse adds an intrinsic window-to-window scatter of '
        '$\\sigma_\\mathrm{int} = %.3f$ and %.3f dex in the two ratios in '
        'quadrature with those uncertainties; it is the only contour directly '
        'comparable with the plotted points, which carry both terms, and it '
        'contains %d of %d (%.0f%%). The intrinsic scatter is thus a small '
        'fraction of a decade: the windows differ measurably in shape, but '
        'around one shape rather than across the plane.\n\n'
        'Of %d windows fitted, %d returned a usable fit and %d of those '
        'converged to a non-degenerate ellipsoid; axes are additionally clipped '
        'below %.2f, where the ratios are not trusted, leaving %d drawn. '
        'Legend counts follow the drawing; the maximum-likelihood shape uses '
        'every usable window in the highest quartile (%d), while the coverage '
        'fractions above are measured on the %d of those that converged to a '
        'non-degenerate ellipsoid.%s\n'
        % (S['k'], snr_txt, c21, c32, me21, me32,
           S['cov_meas'][0], S['cov_meas'][1], S['cov_meas'][2], EXPECTED,
           s21, s32,
           S['cov_tot'][0], S['cov_tot'][1], S['cov_tot'][2],
           S['n_fitted'], S['n_usable'], len(S['d']), S['floor'], n_draw,
           S['n_q4_fit'], S['n_q4_cov'], off))


def main(k=mtf.K, floor=mtf.FLOOR, ncol=2, outdir=None, variant=None):
    import matplotlib
    matplotlib.use('Agg')
    outdir = outdir or os.path.join(ROOT, 'results', 'figures')
    os.makedirs(outdir, exist_ok=True)
    # Resolve BEFORE use: `variant` lands in the output filenames below.
    if variant is None:
        variant = mtf.default_variant()

    apply_science_style()
    S = shape_plane_data(k=k, floor=floor, variant=variant)
    fig, ax, leg = fig_shape_plane_science(S, ncol=ncol)

    # The variant rides on the filename so the arcsinh and raw-flux versions of
    # the paper figure (and their captions) cannot overwrite each other.
    base = os.path.join(outdir, 'shape_plane_science' + variant)
    # PDF is the deliverable (Science wants vector); PNG is for screen checks.
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=600)
    cap = os.path.join(outdir, 'shape_plane_science%s_caption.txt' % variant)
    with open(cap, 'w') as fh:
        fh.write(caption(S))
    return base + '.pdf', base + '.png', cap, fig, S


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--k', type=int, default=mtf.K)
    ap.add_argument('--floor', type=float, default=mtf.FLOOR)
    ap.add_argument('--ncol', type=int, default=2, choices=(1, 2, 3),
                    help='Science column width: 5.7 / 12.1 / 18.4 cm')
    ap.add_argument('--outdir', default=None)
    # Follows the pipeline-wide default (raw flux) via mtf.default_variant();
    # CASA_ARCSINH_UNITS=1 selects arcsinh.  See analysis/scale_split.py.
    ap.add_argument('--variant', default=None,
                    help="'_linear' = raw flux (default), '' = arcsinh run")
    a = ap.parse_args()
    for p in main(k=a.k, floor=a.floor, ncol=a.ncol, outdir=a.outdir,
                  variant=a.variant)[:3]:
        print(os.path.relpath(p, ROOT))
