"""Test whether one ellipsoidal structure function describes all scales.

The analysis so far fits a single ellipsoid to every lag simultaneously, which
ASSUMES the shape and orientation are scale-independent.  This script tests that
assumption directly: split each window's lag points at the median 3D lag radius
(so equal numbers of points fall on each side), fit the two halves
independently, and ask whether the axis ratios and the principal-axis
orientation agree.

What is and is not measurable in a narrow band
----------------------------------------------
At small lag the profile reduces to S2 -> var_inf * |L^-1 . dr|^alpha, so
scaling L -> c L is exactly absorbed by var_inf -> var_inf * c^alpha.  The
absolute axis length a1 is therefore weakly determined (near-degenerate) in the
inner band, and a1 must NOT be compared between bands.  The axis RATIOS a2/a1,
a3/a2 and the orientation of the principal axes are not degenerate, and those
are what this test compares.

Method
------
Splitting is done by NaN-ing s2 outside the band before fit_s2 sees it:
_make_fit_data selects on np.isfinite(plane), so this restricts the fit without
modifying any shared code path.  The split radius is computed ONCE per window
from the full-band fit point set and then held FIXED across the jackknife
refits, so the two bins stay comparable when a spatial block is deleted.

Errors come from the same 2x2 spatial block jackknife used everywhere else in
this project (delete one image quadrant, refit, spread over the 4 refits).

Orientation is compared as the UNSIGNED angle between principal-axis
directions, arccos|n_A . n_B|, because eigenvectors carry an arbitrary sign and
the raw Euler angles flip discontinuously (the axis-flip caveat in
_jackknife_fit's own docstring).

Split modes
-----------
'median'  equal NUMBER of fit points each side (r_split = median radius).  This
          is what was asked for first, but note it is very unequal in LOG
          radius: the inner band spans ~1.45 dex and the outer only ~0.27 dex,
          because most lag pixels sit at large radius.  A profile slope alpha
          cannot be determined over 0.27 dex, so the outer band is poorly
          conditioned and alpha there runs wild.
'logmid'  equal LOG-radius coverage each side (r_split = geometric mean of
          r_min and r_max).  Balances conditioning at the price of very unequal
          counts.  This is the CONTROL for whether any shape difference found
          under 'median' is physical or an artefact of the narrow outer band.

Usage:  python analysis/scale_split.py [n_workers] [stride] [windows] [mode]
          defaults: 8 2 q4 median
One JSON per window in data/scale_split_<mode>_s<stride>/, so runs at different
split modes can never be mixed.  Runs are resumable.
"""
import json
import multiprocessing as mp
import os
import sys
import time
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import structure_function as sf

K = 2  # 2x2 image blocks -> 4 delete-one-block jackknife samples

# Identical settings to jackknife_noise.py / bootstrap_windows.py.  A mismatch
# here would make the full-band control fit disagree with the committed runs,
# which is the canary that these have drifted apart.
COMPUTE_KW = dict(background=0.03, arcsinh_scale=0.03, assume_stationary=True)
FIT_KW = dict(max_nfev=None, weighting='1/r', min_n_fraction=0.1)
READ_KW = dict(edge_mask_radius=50, min_coverage=0.25)
INNER_UV = 200


def _lag_radius_grid(s2):
    """|(dU, dV, dW)| on the full (n_pairs, n_lag_v, n_lag_u) lag grid."""
    DV, DU = np.meshgrid(s2['lag_dv'], s2['lag_du'], indexing='ij')
    r_uv2 = (DU ** 2 + DV ** 2)[None, :, :]
    dw2 = np.asarray(s2['lag_dw'], dtype=float)[:, None, None] ** 2
    return np.sqrt(r_uv2 + dw2)


def _fit_point_radii(s2, stride):
    """3D lag radii of exactly the points fit_s2 would use (full band)."""
    _, _, lags_flat, _ = sf._make_fit_data(
        s2, INNER_UV, FIT_KW.get('min_same_epoch_lag_pix', 4),
        10 ** -3.75, min_n_fraction=FIT_KW['min_n_fraction'],
        fit_stride=stride)
    return np.linalg.norm(lags_flat, axis=1), lags_flat


def _banded(s2, r_grid, lo, hi):
    """Shallow copy of s2 with S2 NaN outside [lo, hi)."""
    out = dict(s2)
    arr = np.array(s2['s2'], dtype=s2['s2'].dtype, copy=True)
    arr[~((r_grid >= lo) & (r_grid < hi))] = np.nan
    out['s2'] = arr
    return out


#: Profiles usable for a single narrow band of lags.  'weibull' is the
#: historical default (3 profile params: alpha, beta, var_inf); 'powerlaw' is
#: the 2-parameter form, and MUST be fit with freeze=('A',) because
#: A * r^alpha is invariant under r -> k r, an exact flat direction against the
#: ellipsoid size (see power_law_log_s2's docstring).  Axis RATIOS -- the only
#: thing the scale profile reads -- are unaffected by that degeneracy.
BAND_PROFILES = {
    'weibull':  dict(profile_fn='weibull_log_s2',  freeze=()),
    'powerlaw': dict(profile_fn='power_law_log_s2', freeze=('A',)),
}


def _fit_one(s2_band, stride, profile='weibull'):
    try:
        spec = BAND_PROFILES[profile]
        fit = sf.fit_s2(s2_band, profile=getattr(sf, spec['profile_fn']),
                        freeze=spec['freeze'],
                        inner_uv_pixels=INNER_UV,
                        **dict(FIT_KW, fit_stride=stride))
        rec = sf._fit_scalars(fit['params'])
        rec['fit_success'] = bool(getattr(fit.get('fit'), 'success', True))
        res = fit.get('fit')
        if res is not None and getattr(res, 'fun', None) is not None:
            rec['rms_resid'] = float(np.sqrt(np.mean(np.asarray(res.fun) ** 2)))
            rec['n_fit'] = int(np.asarray(res.fun).size)
        return rec
    except Exception as exc:
        return {'error': repr(exc)}


def _delete_block(data, i, j, k):
    """Copy of `data` with spatial block (i, j) of a k x k grid set to NaN.

    This is the project's standard jackknife blocking: delete one rectangular
    patch of the image (all epochs), refit, and take the spread over the k^2
    refits.  Factored out here so scale_split and scale_profile use exactly the
    same block definition.
    """
    flux = data['flux_epochs']
    _, ny, nx = flux.shape
    r_edges = np.linspace(0, ny, k + 1).astype(int)
    c_edges = np.linspace(0, nx, k + 1).astype(int)
    d2 = dict(data)
    f2 = np.array(flux, copy=True)
    f2[:, r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]] = np.nan
    d2['flux_epochs'] = f2
    return d2


def _fit_bands(data, r_split, stride, r_grid=None):
    """Fit inner band, outer band, and the full band on one image realisation."""
    s2 = sf.compute_s2(data, **COMPUTE_KW)
    if r_grid is None:
        r_grid = _lag_radius_grid(s2)
    inf = np.inf
    return dict(
        inner=_fit_one(_banded(s2, r_grid, 0.0, r_split), stride),
        outer=_fit_one(_banded(s2, r_grid, r_split, inf), stride),
        full=_fit_one(s2, stride),
    ), r_grid


def _split_radius(radii, mode):
    """Band boundary: equal counts ('median') or equal log-coverage ('logmid')."""
    if mode == 'median':
        return float(np.median(radii))
    if mode == 'logmid':
        pos = radii[radii > 0]
        return float(np.sqrt(pos.min() * pos.max()))
    raise ValueError(f'unknown split mode {mode!r}')


def _one_window(spec):
    row, col, size, stride, out_dir, mode = spec
    out_fn = f'{out_dir}/ss_r{row}_c{col}_s{size}.json'
    if os.path.exists(out_fn):
        return out_fn, 'cached'

    t0 = time.time()
    data = sf.read_window(row, col, size, size, data_dir='data', **READ_KW)

    # Split radius from the FULL-band fit point set: median => equal counts.
    s2_full = sf.compute_s2(data, **COMPUTE_KW)
    r_grid = _lag_radius_grid(s2_full)
    radii, lags_flat = _fit_point_radii(s2_full, stride)
    r_split = _split_radius(radii, mode)
    n_in = int(np.sum(radii < r_split))
    n_out = int(np.sum(radii >= r_split))
    # the 1/r weighting means equal COUNTS is not equal WEIGHT; record both
    w = 1.0 / np.maximum(np.hypot(lags_flat[:, 0], lags_flat[:, 1]),
                         float(abs(s2_full['lag_du'][1] - s2_full['lag_du'][0])))
    w_frac_inner = float(w[radii < r_split].sum() / w.sum())

    central, _ = _fit_bands(data, r_split, stride, r_grid=r_grid)

    samples = []
    for i in range(K):
        for j in range(K):
            d2 = _delete_block(data, i, j, K)
            rec, _ = _fit_bands(d2, r_split, stride)
            rec['block'] = [i, j]
            samples.append(rec)

    out = dict(row=row, col=col, size=size, k=K, fit_stride=stride,
               profile='weibull', method=f'scale_split_{mode}_r3', split_mode=mode,
               r_split=r_split, n_inner=n_in, n_outer=n_out,
               w_frac_inner=w_frac_inner,
               r_min=float(radii.min()), r_max=float(radii.max()),
               central=central, samples=samples,
               wall_s=time.time() - t0)
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_fn + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh)
    os.replace(tmp, out_fn)
    return out_fn, 'done'


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    which = sys.argv[3] if len(sys.argv) > 3 else 'q4'
    mode = sys.argv[4] if len(sys.argv) > 4 else 'median'

    wl = json.load(open(f'{_ROOT}/handoff/{which}_windows.json'))
    specs_raw = wl['specs'] if isinstance(wl, dict) else wl
    suffix = '' if mode == 'median' else f'_{mode}'
    out_dir = f'{_ROOT}/data/scale_split{suffix}_s{stride}'
    os.makedirs(out_dir, exist_ok=True)
    specs = [(int(r), int(c), int(s), stride, out_dir, mode) for r, c, s in specs_raw]

    print(f'{len(specs)} windows, stride {stride}, mode {mode}, '
          f'{n_workers} workers -> {out_dir}', flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(n_workers) as pool:
        for n, (fn, status) in enumerate(
                pool.imap_unordered(_one_window, specs), 1):
            print(f'[{n}/{len(specs)}] {status} {os.path.basename(fn)} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
