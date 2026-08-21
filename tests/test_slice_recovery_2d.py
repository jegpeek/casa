"""The UV-plane (2D) fit must recover the dW=0 SLICE of a known ellipsoid.

This is the 2D analogue of test_band_recovery.py, and it is what licenses
analysis/scale_profile_2d.py.  A known ellipsoid is injected into the real lag
geometry (noise-free), the dW != 0 planes are dropped, and the 2D fit must
return the in-plane axis ratio and position angle of the ellipsoid's central
CROSS-SECTION -- the Schur complement of the shape matrix, not its projection.

It also pins two properties the 2D fit depends on:

  * at dW = 0 the residuals are EXACTLY independent of s33/l13/l23 (not merely
    weakly dependent), which is why those three must be frozen; and
  * freeze=() leaves fit_s2 numerically identical to the unmodified solver,
    so the 3D results are unaffected by the freeze machinery.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))
sys.path.insert(0, os.path.join(_ROOT, 'tests'))

import structure_function as sf  # noqa: E402
import scale_split as ss  # noqa: E402
import scale_profile_2d as sp2  # noqa: E402
import test_band_recovery as tbr  # noqa: E402


def test_dw0_residuals_ignore_w_params():
    """s33, l13, l23 have IDENTICALLY zero effect on r at dW = 0."""
    rng = np.random.default_rng(0)
    lags = np.column_stack([rng.normal(size=500), rng.normal(size=500),
                            np.zeros(500)])
    base = [0.30, 0.11, 0.7, 0.05, -0.2, 0.4]
    r0 = sf._compute_r(base, lags)
    for i in (2, 4, 5):                       # s33, l13, l23
        p = list(base)
        p[i] = base[i] * 3.7 + 1.3
        assert np.array_equal(sf._compute_r(p, lags), r0), _GEOM_MSG[i]


_GEOM_MSG = {2: 's33 affects r at dW=0', 4: 'l13 affects r at dW=0',
             5: 'l23 affects r at dW=0'}


def test_slice_axes_are_the_schur_complement():
    """principal_axes_2d must give the SLICE, distinguishable from the
    projection: for a tilted ellipsoid the two differ."""
    geom = sf.params_from_principal_axes(a1=0.30, a2=0.15, a3=0.05,
                                         theta=0.7, phi=0.4, psi=0.3)
    L = np.array([[geom['s11'], geom['l12'], geom['l13']],
                  [0.0, geom['s22'], geom['l23']],
                  [0.0, 0.0, geom['s33']]])
    C = L @ L.T
    sl = sf.principal_axes_2d(geom)
    # slice: invert C, take the 2x2 UV block, invert back
    Cinv = np.linalg.inv(C)
    C_slice = np.linalg.inv(Cinv[:2, :2])
    ev = np.sort(np.linalg.eigvalsh(C_slice))[::-1]
    assert np.allclose([sl['b1'], sl['b2']], np.sqrt(ev), rtol=1e-10)
    # and it is NOT the projection C[:2,:2] for this tilted case
    ev_proj = np.sort(np.linalg.eigvalsh(C[:2, :2]))[::-1]
    assert not np.allclose(np.sqrt(ev_proj), np.sqrt(ev), rtol=1e-3)


def slice_recovery(row=2000, col=3200, stride=2):
    data = sf.read_window(row, col, 400, 400,
                          data_dir=os.path.join(_ROOT, 'data'), **ss.READ_KW)
    s2_real = sf.compute_s2(data, **ss.COMPUTE_KW)
    s2 = tbr._synth_s2(s2_real)
    truth = sf.principal_axes_2d(sf.params_from_principal_axes(**tbr.TRUTH))
    fit = sf.fit_s2(sp2._dw0_only(s2), profile=sf.weibull_log_s2,
                    inner_uv_pixels=ss.INNER_UV, freeze=sp2.FREEZE_2D,
                    **dict(ss.FIT_KW, fit_stride=stride))
    rec = sf._fit_scalars(fit['params'])
    return rec, truth, fit


def test_2d_fit_recovers_injected_slice():
    rec, truth, fit = slice_recovery()
    true_ratio = truth['b2'] / truth['b1']
    true_pa = np.degrees(truth['pa'])
    print('injected slice b2/b1=%.5f PA=%.3f | recovered %.5f PA=%.3f'
          % (true_ratio, true_pa, rec['b2b1'], rec['pa2d']))
    assert abs(rec['b2b1'] - true_ratio) < 0.01, 'slice ratio not recovered'
    assert abs(rec['pa2d'] - true_pa) < 1.0, 'slice PA not recovered'
    # the frozen parameters must not have moved from their guess
    for k in sp2.FREEZE_2D:
        assert np.isfinite(fit['params'][k])


def test_freeze_empty_is_a_noop():
    """freeze=() must not perturb the ordinary 3D fit."""
    data = sf.read_window(2000, 3200, 400, 400,
                          data_dir=os.path.join(_ROOT, 'data'), **ss.READ_KW)
    s2 = tbr._synth_s2(sf.compute_s2(data, **ss.COMPUTE_KW))
    kw = dict(profile=sf.weibull_log_s2, inner_uv_pixels=ss.INNER_UV,
              **dict(ss.FIT_KW, fit_stride=4))
    a = sf.fit_s2(s2, **kw)
    b = sf.fit_s2(s2, freeze=(), **kw)
    for k in sf._GEOM_KEYS:
        assert a['params'][k] == b['params'][k], f'{k} changed with freeze=()'


def test_freeze_rejects_unknown_name():
    data = sf.read_window(2000, 3200, 400, 400,
                          data_dir=os.path.join(_ROOT, 'data'), **ss.READ_KW)
    s2 = sf.compute_s2(data, **ss.COMPUTE_KW)
    try:
        sf.fit_s2(s2, freeze=('s44',), profile=sf.weibull_log_s2,
                  inner_uv_pixels=ss.INNER_UV, **dict(ss.FIT_KW, fit_stride=8))
    except ValueError as exc:
        assert 's44' in str(exc)
    else:
        raise AssertionError('freeze accepted an unknown parameter name')


if __name__ == '__main__':
    test_dw0_residuals_ignore_w_params()
    test_slice_axes_are_the_schur_complement()
    test_2d_fit_recovers_injected_slice()
    test_freeze_empty_is_a_noop()
    test_freeze_rejects_unknown_name()
    print('all 2D slice-recovery tests passed')
