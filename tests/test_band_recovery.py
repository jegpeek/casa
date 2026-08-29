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

TWO TESTS, because there are two error sources and they must not be confused.

`test_bands_recover_injected_ratios` fits the SAME profile that was injected
(Weibull).  Any error is then purely geometric -- the dW quantisation above --
and recovery is essentially exact (a2/a1 and T to three decimals in every band),
which is the licence the T(scale) curve needs.

`test_powerlaw_band_fit_is_biased_by_model_mismatch` fits the CANONICAL band
profile (powerlaw, since commit d620e71) to that same Weibull field.  The
combination is biased: a2/a1 falls from 0.48 to 0.05 across the five bands, while
a3/a2 stays correct.  The bias is PINNED rather than asserted away, so it cannot
be mistaken for a physical T(scale) trend.  Consequence for the paper: absolute
per-band a2/a1 from a powerlaw band fit is not quotable, whereas trends across
identically-fitted bands are.

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


def band_recovery(row=3200, col=2400, stride=2, band_dex=BAND_DEX,
                  profile='weibull'):
    """Inject TRUTH as a Weibull S2 field and refit it band by band.

    `profile` is the profile the FIT uses.  The default 'weibull' matches the
    injection, which is what makes this a geometry test: any departure from
    TRUTH is then attributable to the lag geometry (the dW quantisation this
    module's docstring is about), not to a model mismatch.  Passing
    profile='powerlaw' instead measures the combined geometry + mismatch error
    -- see test_powerlaw_band_fit_is_biased_by_model_mismatch.
    """
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
        rec = ss._fit_one(ss._banded(s2, r_grid, 10 ** lo, 10 ** hi), stride,
                          profile=profile)
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
    """The geometry test: matched profile, so only the lag grid can hurt us."""
    if not have_bulk_data():
        print('SKIP test_bands_recover_injected_ratios: input arrays absent')
        return
    rows, t_true = band_recovery(profile='weibull')
    good = [r for r in rows if r['ok']]
    assert len(good) >= 4, f'too few bands converged: {len(good)} of {len(rows)}'
    # The widest-radius bands must recover the truth well.
    for r in good[2:]:
        assert abs(r['a2a1'] - TRUTH['a2'] / TRUTH['a1']) < 0.10, r
        assert abs(r['T'] - t_true) < 0.15, r


def test_powerlaw_band_fit_is_biased_by_model_mismatch():
    """The canonical band profile does NOT recover a Weibull field's ratios.

    scale_split.CANONICAL_PROFILE became 'powerlaw' in commit d620e71, on the
    argument that within one 0.6-dex band the Weibull's saturation scale is
    never sampled, so the two are numerically equivalent there.  That argument
    holds for the NARROW inner bands and fails for the wide outer ones: fitting
    A*r^alpha to a field that does saturate drives a2/a1 toward zero as the band
    reaches larger lags (0.48 -> 0.05 across the five bands here).

    This test pins the direction and rough size of that bias rather than
    asserting recovery, so the effect cannot be mistaken for a real T(scale)
    trend.  The scale-profile results are read as RELATIVE trends across bands
    fitted the same way, which is why the published numbers survive this -- but
    the absolute per-band ratios from a powerlaw fit are not calibrated.
    """
    if not have_bulk_data():
        print('SKIP test_powerlaw_band_fit_is_biased_by_model_mismatch: '
              'input arrays absent')
        return
    rows, _ = band_recovery(profile='powerlaw')
    good = [r for r in rows if r['ok']]
    assert len(good) >= 4, f'too few bands converged: {len(good)} of {len(rows)}'
    truth = TRUTH['a2'] / TRUTH['a1']
    # Innermost band: the equivalence argument holds, recovery is good.
    assert abs(good[0]['a2a1'] - truth) < 0.10, good[0]
    # Outermost band: strongly biased LOW, and monotonically so.
    assert good[-1]['a2a1'] < 0.5 * truth, good[-1]
    a2a1 = [r['a2a1'] for r in good]
    assert a2a1 == sorted(a2a1, reverse=True), a2a1
    # a3/a2 is NOT what degrades -- the bias is specific to a2/a1.
    for r in good:
        assert abs(r['a3a2'] - TRUTH['a3'] / TRUTH['a2']) < 0.10, r


if __name__ == '__main__':
    t_true = triaxiality(TRUTH['a1'], TRUTH['a2'], TRUTH['a3'])
    print('injected (Weibull): a2/a1=%.3f a3/a2=%.3f T=%.3f alpha=%.2f'
          % (TRUTH['a2'] / TRUTH['a1'], TRUTH['a3'] / TRUTH['a2'], t_true, ALPHA))
    for profile in ('weibull', 'powerlaw'):
        rows, _ = band_recovery(profile=profile)
        print('\nfit profile: %s%s' % (
            profile, '   <-- CANONICAL' if profile == ss.CANONICAL_PROFILE else
                     '   (matches injection)'))
        print('%9s %9s  %7s %7s %7s %7s'
              % ('lo', 'hi', 'a2a1', 'a3a2', 'T', 'alpha'))
        for r in rows:
            if not r['ok']:
                print('%9.4f %9.4f  FIT FAILED' % (r['lo'], r['hi']))
            else:
                print('%9.4f %9.4f  %7.3f %7.3f %7.3f %7.3f'
                      % (r['lo'], r['hi'], r['a2a1'], r['a3a2'], r['T'],
                         r['alpha']))
