"""Can a narrow lag band recover the shape it was given?

The continuous-scale measurement (analysis/scale_profile.py) fits the ellipsoid
in overlapping bands of log lag radius and reports triaxiality T as a function
of band centre.  That is only meaningful if a band actually CONSTRAINS the axis
ratios -- and there is a specific reason to doubt it at small radius: dW is
quantised by the epoch spacing, so the innermost bands contain only a handful of
distinct |dW| values (3 of 11 in the smallest 0.6 dex band).  a3 is largely the
out-of-plane axis, so if any band cannot recover a3 the T(scale) curve there is
an artefact.

This test injects a KNOWN ellipsoid into the real lag geometry (noise-free,
using the real lag grid, counts and mask), refits it band by band, and asserts
the ratios come back.  It is the calibration that licenses the T(scale) curve.

Run:  pytest -q tests/test_band_recovery.py
      python tests/test_band_recovery.py     # prints the recovery table
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

import structure_function as sf  # noqa: E402
import scale_split as ss  # noqa: E402

TRUTH = dict(a1=0.30, a2=0.15, a3=0.05, theta=0.7, phi=0.4, psi=0.3)
ALPHA, BETA, VAR_INF = 0.67, 1.5, 1.0e-3
BAND_DEX = 0.6


def _synth_s2(s2_real, truth=TRUTH):
    """Replace s2 values with a noise-free ellipsoidal model on the real grid."""
    geom = sf.params_from_principal_axes(**truth)
    params = dict(geom, alpha=ALPHA, beta=BETA, var_inf=VAR_INF)
    DV, DU = np.meshgrid(s2_real['lag_dv'], s2_real['lag_du'], indexing='ij')
    dw = np.asarray(s2_real['lag_dw'], dtype=float)
    lags = np.stack([np.broadcast_to(DU, (dw.size,) + DU.shape),
                     np.broadcast_to(DV, (dw.size,) + DV.shape),
                     np.broadcast_to(dw[:, None, None], (dw.size,) + DU.shape)],
                    axis=-1).reshape(-1, 3)
    logs2 = sf.log_s2_model(params, lags, profile=sf.weibull_log_s2)
    out = dict(s2_real)
    model = (10.0 ** logs2).reshape(np.asarray(s2_real['s2']).shape)
    # keep the real coverage pattern: where there was no data, there is none now
    model = np.where(np.isfinite(np.asarray(s2_real['s2'], dtype=float)), model, np.nan)
    out['s2'] = model
    return out


def have_bulk_data(root=None):
    """True when the input arrays are present -- tier B in REPRODUCING.md.

    A clone carries the fit OUTPUTS, not the input arrays, so the tests that
    re-fit real windows cannot run from a bare checkout.  Returning False lets
    them report a skip; a bare FileNotFoundError reads like a broken repo and
    would send a new user hunting for a bug that isn't there.
    """
    d = os.path.join(root or _ROOT, 'data')
    return all(os.path.exists(os.path.join(d, f)) for f in
               ('resampled_epochs_noclip.npy', 'U_grid.npy', 'V_grid.npy'))


def band_recovery(row=3200, col=2400, stride=2, band_dex=BAND_DEX):
    # NB sf.read_window signature is (row0, col0, nrows, ncols) -- row FIRST.
    data = sf.read_window(row, col, 400, 400, data_dir=os.path.join(_ROOT, 'data'),
                          **ss.READ_KW)
    s2_real = sf.compute_s2(data, **ss.COMPUTE_KW)
    s2 = _synth_s2(s2_real)
    r_grid = ss._lag_radius_grid(s2)
    radii, _ = ss._fit_point_radii(s2, stride)
    lo0, hi0 = np.log10(radii.min()), np.log10(radii.max())

    t_true = triaxiality(TRUTH['a1'], TRUTH['a2'], TRUTH['a3'])
    rows = []
    for lo in np.arange(lo0, hi0 - band_dex / 2, band_dex / 2):
        hi = min(lo + band_dex, hi0 + 1e-9)
        rec = ss._fit_one(ss._banded(s2, r_grid, 10 ** lo, 10 ** hi), stride)
        if 'error' in rec or not rec.get('fit_success'):
            rows.append(dict(lo=10 ** lo, hi=10 ** hi, ok=False))
            continue
        a1, a2, a3 = rec['a1'], rec['a2'], rec['a3']
        rows.append(dict(lo=10 ** lo, hi=10 ** hi, ok=True,
                         a2a1=a2 / a1, a3a2=a3 / a2,
                         T=triaxiality(a1, a2, a3),
                         alpha=rec.get('alpha', np.nan)))
    return rows, t_true


def triaxiality(a1, a2, a3):
    """T = (a1^2 - a2^2) / (a1^2 - a3^2).  T->0 oblate, T->1 prolate."""
    return (a1 ** 2 - a2 ** 2) / (a1 ** 2 - a3 ** 2)


def triaxiality_from_ratios(a2a1, a3a2):
    return triaxiality(1.0, a2a1, a2a1 * a3a2)


def test_bands_recover_injected_ratios():
    if not have_bulk_data():
        print('SKIP test_bands_recover_injected_ratios: input arrays absent')
        return
    rows, t_true = band_recovery()
    good = [r for r in rows if r['ok']]
    assert len(good) >= 4, f'too few bands converged: {len(good)} of {len(rows)}'
    # The widest-radius bands must recover the truth well.
    for r in good[2:]:
        assert abs(r['a2a1'] - TRUTH['a2'] / TRUTH['a1']) < 0.10, r
        assert abs(r['T'] - t_true) < 0.15, r


if __name__ == '__main__':
    rows, t_true = band_recovery()
    print('injected: a2/a1=%.3f a3/a2=%.3f T=%.3f alpha=%.2f'
          % (TRUTH['a2'] / TRUTH['a1'], TRUTH['a3'] / TRUTH['a2'], t_true, ALPHA))
    print('%9s %9s  %7s %7s %7s %7s' % ('lo', 'hi', 'a2a1', 'a3a2', 'T', 'alpha'))
    for r in rows:
        if not r['ok']:
            print('%9.4f %9.4f  FIT FAILED' % (r['lo'], r['hi']))
        else:
            print('%9.4f %9.4f  %7.3f %7.3f %7.3f %7.3f'
                  % (r['lo'], r['hi'], r['a2a1'], r['a3a2'], r['T'], r['alpha']))
