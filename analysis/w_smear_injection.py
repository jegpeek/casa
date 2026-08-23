#!/usr/bin/env python
"""Does unmodelled smearing along W manufacture the in-plane axis preference?

W is the axis whose resolution comes from epoch spacing rather than pixel
scale: in a typical window the in-plane lag grid steps at 0.0016 ly while the
W lags run 0, 0.0224 ... 0.2142 ly, so W is sampled ~14x more coarsely over
about a third of the in-plane lever arm.  That asymmetry motivated a standing
caveat that the measured preference of a1 for the echo plane might be a
sampling artifact.

The caveat had the SIGN BACKWARDS, and this script is the test that shows it.
Smearing the field along W is a convolution along W: it ADDS correlation
length there, lengthening the recovered ellipsoid along W.  It cannot shorten
it.  So W-smearing rotates the recovered long axis TOWARD W, away from the
plane -- the systematic works against the in-plane preference rather than
producing it, which makes the measured median angle(a1, W) a conservative
floor.

Method.  A noise-free ellipsoidal S2 is evaluated on a real window's lag grid
(real coverage pattern preserved, as in tests/test_band_recovery.py), smeared
along W, and refit with the standard band fitter.  Smearing is applied to the
COVARIANCE, not to S2 directly: if the field is convolved with kernel K, the
covariance is convolved with K*K -- Gaussian of width sqrt(2)*sigma_W -- and
S2 is then rebuilt as 2*(C_s(0) - C_s(d)).  Convolving S2 itself is wrong and
does not preserve S2(0) = 0.

Outputs (written to results/):
    w_smear_injection.csv   recovered angle + axes vs (true tilt, sigma_W)
    w_smear_inflation.csv   per-axis inflation, split by W-alignment
    w_smear_injection.png   two-panel figure

Usage:
    python analysis/w_smear_injection.py
    python analysis/w_smear_injection.py --row 3200 --col 2400 --stride 2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import structure_function as sf          # noqa: E402
import scale_split as ss                 # noqa: E402
import summarize_scale_split as sss      # noqa: E402

# Truth ellipsoid for the sweep: a clearly prolate object, so that a rotation
# of the recovered long axis is unambiguous.
TRUTH_AXES = dict(a1=0.30, a2=0.15, a3=0.15)
ALPHA, BETA, VAR_INF = 0.67, 1.5, 0.001

# sigma_W grid [ly].  The top of the range is ~2x the smallest non-zero W lag,
# i.e. a deliberately severe test rather than a plausible residual.
SIGMA_W = (0.0, 0.010, 0.020, 0.030, 0.050)
TILTS_DEG = (0, 30, 45, 60, 90)

# Fits use the Weibull here, not the canonical power law: the injected truth is
# a Weibull and this test is about geometry recovery, not profile choice.  The
# power law has no saturation scale and so cannot represent the injected field.
FIT_PROFILE = 'weibull'

# Threading (figure-style 4.1): one colour for the W-aligned axis throughout.
C_FOCAL = '#0F6FC6'
C_META = '#8A8A8A'
BASE, MID, SMALL = 8, 7, 6


def _model_params(truth):
    return dict(sf.params_from_principal_axes(**truth),
                alpha=ALPHA, beta=BETA, var_inf=VAR_INF)


def _s2_at(params, DU, DV, dw):
    """Unsmeared model S2 on a (DV, DU) grid at one fixed W lag."""
    lags = np.stack([DU.ravel(), DV.ravel(), np.full(DU.size, dw)], axis=-1)
    return (10.0 ** sf.log_s2_model(params, lags,
                                    profile=sf.weibull_log_s2)).reshape(DU.shape)


def _cov_at(params, DU, DV, dw):
    """Covariance implied by the structure function: C = (sill - S2)/2."""
    return 0.5 * (VAR_INF - _s2_at(params, DU, DV, dw))


def smeared_s2(s2_real, truth, sigma_w_ly, n_quad=13, inner_pix=None):
    """S2 of the truth field after Gaussian smearing along W.

    Only the inner |du|, |dv| <= inner_pix region is smeared.  That is the
    region the fitter uses (structure_function.fit_s2 masks to
    inner_uv_pixels), and the quadrature is the expensive part, so evaluating
    the outer grid unsmeared costs nothing and keeps the array shape intact.
    """
    inner_pix = ss.INNER_UV + 10 if inner_pix is None else inner_pix
    params = _model_params(truth)
    du = np.asarray(s2_real['lag_du'], float)
    dv = np.asarray(s2_real['lag_dv'], float)
    dw = np.asarray(s2_real['lag_dw'], float)

    iu = np.abs(du) <= inner_pix * abs(du[1] - du[0])
    iv = np.abs(dv) <= inner_pix * abs(dv[1] - dv[0])
    DVi, DUi = np.meshgrid(dv[iv], du[iu], indexing='ij')
    DVf, DUf = np.meshgrid(dv, du, indexing='ij')
    zero = np.zeros((1, 1))

    if sigma_w_ly <= 0:
        cov = lambda w: _cov_at(params, DUi, DVi, w)          # noqa: E731
        c0 = _cov_at(params, zero, zero, 0.0)[0, 0]
    else:
        # Field kernel K has width sigma_W, so the covariance kernel K*K has
        # width sqrt(2)*sigma_W; Gauss-Hermite nodes carry the extra sqrt(2).
        x, wt = hermgauss(n_quad)
        nodes = np.sqrt(2.0) * (np.sqrt(2.0) * sigma_w_ly) * x
        wt = wt / np.sqrt(np.pi)
        cov = lambda w: sum(wt[i] * _cov_at(params, DUi, DVi, w - nodes[i])  # noqa: E731
                            for i in range(n_quad))
        c0 = sum(wt[i] * _cov_at(params, zero, zero, -nodes[i])[0, 0]
                 for i in range(n_quad))

    out = np.empty((dw.size, dv.size, du.size), float)
    for k, w in enumerate(dw):
        full = _s2_at(params, DUf, DVf, w)
        full[np.ix_(iv, iu)] = 2.0 * (c0 - cov(w))
        out[k] = full

    # Keep the real coverage: where there was no data, there is none now.
    out = np.where(np.isfinite(np.asarray(s2_real['s2'], float)), out, np.nan)
    d = dict(s2_real)
    d['s2'] = out
    return d


def angle_a1_w(rec):
    """Unsigned angle [deg] between the fitted long axis and W.

    Via the stored Euler angles -- NOT via an eigendecomposition of the raw
    shape matrix, which is easy to get wrong in a way that looks plausible.
    """
    n1, _, _ = sss.axes_from_angles(rec['theta'], rec['phi'], rec['psi'])
    return float(np.degrees(np.arccos(min(abs(float(n1[2])), 1.0))))


def axes_by_w_alignment(rec):
    """The three (|n_i . W|, a_i) pairs, most W-aligned first."""
    n1, n2, n3 = sss.axes_from_angles(rec['theta'], rec['phi'], rec['psi'])
    pairs = [(abs(float(n[2])), float(rec[k]))
             for n, k in ((n1, 'a1'), (n2, 'a2'), (n3, 'a3'))]
    return sorted(pairs, key=lambda t: -t[0])


def run(row=3200, col=2400, stride=2, data_dir=None):
    data_dir = data_dir or os.path.join(_ROOT, 'data')
    # NB sf.read_window signature is (row0, col0, nrows, ncols) -- row FIRST.
    data = sf.read_window(row, col, 400, 400, data_dir=data_dir, **ss.READ_KW)
    s2_real = sf.compute_s2(data, **ss.COMPUTE_KW)

    sweep, inflation = [], []
    for tilt in TILTS_DEG:
        truth = dict(TRUTH_AXES, theta=np.radians(tilt), phi=0.0, psi=0.0)
        n1t, _, _ = sss.axes_from_angles(tilt, 0.0, 0.0)
        true_ang = float(np.degrees(np.arccos(min(abs(float(n1t[2])), 1.0))))
        ctrl = None
        for sig in SIGMA_W:
            rec = ss._fit_one(smeared_s2(s2_real, truth, sig), stride,
                              profile=FIT_PROFILE)
            ok = ('error' not in rec) and bool(rec.get('fit_success'))
            sweep.append(dict(row=row, col=col, true_tilt=tilt,
                              true_ang=true_ang, sigma_w=sig, ok=ok,
                              ang=angle_a1_w(rec) if ok else np.nan,
                              a1=rec.get('a1', np.nan), a2=rec.get('a2', np.nan),
                              a3=rec.get('a3', np.nan),
                              a2a1=(rec['a2'] / rec['a1']) if ok else np.nan,
                              alpha=rec.get('alpha', np.nan),
                              beta=rec.get('beta', np.nan)))
            if not ok:
                continue
            pr = axes_by_w_alignment(rec)
            w_axis, in_plane = pr[0][1], float(np.mean([pr[1][1], pr[2][1]]))
            if sig == 0:
                ctrl = (w_axis, in_plane)
            if ctrl is not None:
                inflation.append(dict(true_tilt=tilt, sigma_w=sig,
                                      w_align_cos=pr[0][0],
                                      infl_w=w_axis / ctrl[0],
                                      infl_plane=in_plane / ctrl[1]))
    return pd.DataFrame(sweep), pd.DataFrame(inflation)


def make_figure(sweep, inflation, out_png):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    mpl.rcParams.update({'font.size': BASE, 'axes.titlesize': BASE,
                         'axes.labelsize': BASE, 'legend.fontsize': SMALL,
                         'xtick.labelsize': SMALL, 'ytick.labelsize': SMALL,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))
    sig_pos = [s for s in SIGMA_W if s > 0]

    ax = axes[0]
    ax.plot([0, 90], [0, 90], ls=':', lw=1.0, color='0.55', zorder=1)
    ax.annotate('unbiased', xy=(76, 76), xytext=(78, 69), fontsize=SMALL,
                color='0.45', rotation=45, rotation_mode='anchor',
                ha='center', va='center')
    cmap = plt.get_cmap('viridis')
    for k, sig in enumerate(sig_pos):
        sub = sweep[(sweep.sigma_w == sig) & sweep.ok].sort_values('true_ang')
        ax.plot(sub.true_ang, sub.ang, 'o-', ms=3.4, lw=1.4, zorder=3,
                color=cmap(0.12 + 0.72 * k / max(1, len(sig_pos) - 1)),
                label='%.0f' % (sig * 1000))
    sub0 = sweep[(sweep.sigma_w == 0) & sweep.ok].sort_values('true_ang')
    ax.plot(sub0.true_ang, sub0.ang, 's--', ms=3.4, lw=1.4, color=C_FOCAL,
            zorder=4, label='0 (control)')
    ax.set_xlabel('True angle between long axis and W  [deg]')
    ax.set_ylabel('Recovered angle  [deg]')
    ax.set_xticks([0, 30, 45, 60, 90])
    ax.set_yticks([0, 30, 45, 60, 90])
    ax.set_xlim(-5, 95)
    ax.set_ylim(-5, 95)
    ax.set_title('Recovered orientation is pulled toward W', loc='left')
    leg = ax.legend(title='$\\sigma_W$ [10$^{-3}$ ly]', frameon=False,
                    fontsize=SMALL, title_fontsize=SMALL, loc='upper left',
                    bbox_to_anchor=(0.015, 1.005), labelspacing=0.22,
                    handlelength=1.6)
    leg._legend_box.align = 'left'
    worst = sweep[(sweep.true_tilt == 45) & (sweep.sigma_w == max(SIGMA_W))]
    if len(worst):
        ax.annotate('bias is toward 0$\\degree$,\naway from the plane',
                    xy=(45, float(worst.ang.iloc[0])), xytext=(64, 16),
                    fontsize=SMALL, color='0.30', ha='center',
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='0.45',
                                    shrinkA=2, shrinkB=3))

    ax = axes[1]
    styles = {0: ('--', 's'), 90: ('-', 'o')}
    for tilt, (ls, mk) in styles.items():
        s = inflation[inflation.true_tilt == tilt].sort_values('sigma_w')
        if not len(s):
            continue
        ax.plot(s.sigma_w * 1e3, s.infl_w, ls=ls, marker=mk, ms=3.6, lw=1.5,
                color=C_FOCAL, zorder=4)
        ax.plot(s.sigma_w * 1e3, s.infl_plane, ls=ls, marker=mk, ms=3.6,
                lw=1.5, color=C_META, zorder=3)
    ax.axhline(1.0, ls=':', lw=1.0, color='0.55', zorder=1)
    ax.annotate('no inflation', xy=(1, 1.0), xytext=(1, 0.985),
                fontsize=SMALL, color='0.45', va='top')
    ax.set_xlabel('W-smearing $\\sigma_W$  [10$^{-3}$ ly]')
    ax.set_ylabel('Recovered axis length / control')
    ax.set_title('The W-aligned axis inflates most', loc='left')
    ax.margins(0.07)
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=C_FOCAL, lw=1.5, label='axis nearest W'),
        mlines.Line2D([], [], color=C_META, lw=1.5,
                      label='the two in-plane axes'),
        mlines.Line2D([], [], color='0.35', lw=1.2, ls='-', marker='o',
                      ms=3.4, label='truth: long axis in plane'),
        mlines.Line2D([], [], color='0.35', lw=1.2, ls='--', marker='s',
                      ms=3.4, label='truth: long axis along W'),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=SMALL,
              loc='upper left', labelspacing=0.22, handlelength=1.9)

    fig.tight_layout(pad=0.5, w_pad=1.9)
    fig.savefig(out_png, dpi=300)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--row', type=int, default=3200)
    ap.add_argument('--col', type=int, default=2400)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out-dir', default=os.path.join(_ROOT, 'results'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sweep, inflation = run(args.row, args.col, args.stride, args.data_dir)
    sweep.to_csv(os.path.join(args.out_dir, 'w_smear_injection.csv'), index=False)
    inflation.to_csv(os.path.join(args.out_dir, 'w_smear_inflation.csv'),
                     index=False)
    make_figure(sweep, inflation,
                os.path.join(args.out_dir, 'w_smear_injection.png'))

    piv = sweep[sweep.ok].pivot_table(index='true_ang', columns='sigma_w',
                                      values='ang')
    print(piv.round(2).to_string())
    ctrl = sweep[(sweep.sigma_w == 0) & (sweep.true_tilt == 0) & sweep.ok]
    if len(ctrl):
        r = ctrl.iloc[0]
        print('\ncontrol at sigma_W=0, W-aligned truth: a2/a1 = %.3f (true %.3f)'
              % (r.a2a1, TRUTH_AXES['a2'] / TRUTH_AXES['a1']))


if __name__ == '__main__':
    main()
