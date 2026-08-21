"""In-plane (UV) axis ratio as a continuous function of scale -- the 2D twin of
analysis/scale_profile.py.

Purpose
-------
The 2D images show a wide range of apparent anisotropy from window to window.
Some of that range is REAL variation in the underlying 3D structures; some is
pure geometry, because a fixed 3D ellipsoid sliced at different orientations
presents a different in-plane ellipse.  This script measures the 2D anisotropy
directly so the two can be separated: the observed spread of the 2D ratio is
compared against the spread PREDICTED by the 3D fits' own implied slices.

Why this is a subspace of the 3D fit, not a new model
----------------------------------------------------
The ellipsoidal radius is r = |L^-1 dr| with L upper-triangular.  Setting
dW = 0 kills the xw term, so r depends ONLY on s11, s22, l12 -- the upper-left
2x2 block L2 of L.  s33, l13, l23 have IDENTICALLY zero gradient (verified to
0.0, not just to round-off) and must therefore be frozen, or least_squares
wanders in a flat 3-dimensional null space.  The in-plane shape matrix is
exactly C2 = L2 L2^T, which is the central SLICE of the 3D ellipsoid through
dW = 0 -- the Schur complement of C, NOT the projection of C onto the plane.
The slice is what a single-epoch image sees.  Consequently:

  * the 2D fit uses the SAME model, SAME profile, SAME weighting, SAME band
    machinery and SAME jackknife as the 3D run -- only the dW=0 planes are
    selected and three parameters are frozen;
  * sf._fit_scalars reports b1, b2, b2b1, pa2d for BOTH runs, so the 3D fit's
    implied slice and the 2D fit's measurement are the same quantity computed
    the same way, and can be differenced directly.

What restricting to dW = 0 costs
--------------------------------
Only the 5 same-epoch planes survive of the 15 epoch pairs, so ~1/3 of the lag
pairs remain.  _make_fit_data already excludes the central
+-min_same_epoch_lag_pix pixels of the dW=0 planes (correlated-noise spike), so
the innermost few pixels are unavailable in 2D exactly as they are in 3D.  The
band radius on a dW=0 plane IS the in-plane radius |(dU, dV)|, so the same
_lag_radius_grid / band_edges code gives in-plane bands with no change.

Method
------
Identical to scale_profile.py: bands BAND_DEX wide in log10 radius stepped
BAND_DEX/2, edges computed once per window from the full-band fit point set and
held FIXED across the 2x2 spatial block jackknife.

Usage:  python analysis/scale_profile_2d.py [n_workers] [stride] [windows] [band_dex]
          defaults: 8 2 q4 0.6
One JSON per window in data/scale_profile_2d_d<band_dex>_s<stride>/, resumable.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

import structure_function as sf  # noqa: E402
import scale_split as ss  # noqa: E402
import scale_profile as sp  # noqa: E402

K = 2                       # 2x2 spatial block jackknife, as everywhere else
BAND_DEX = 0.6
# The three parameters with identically zero gradient at dW = 0.
FREEZE_2D = ('s33', 'l13', 'l23')


def _dw0_only(s2):
    """Shallow copy of s2 with every dW != 0 plane NaN-ed out.

    Selects by lag_dw == 0 rather than by epoch_pairs so it stays correct if
    the pair ordering ever changes.  Uses the same NaN-masking trick as
    scale_split._banded: _make_fit_data selects on np.isfinite, so no shared
    code path changes.
    """
    dw = np.asarray(s2['lag_dw'], dtype=float)
    keep = (dw == 0.0)
    if not keep.any():
        raise ValueError('no dW=0 planes in this s2 result')
    out = dict(s2)
    arr = np.array(s2['s2'], dtype=s2['s2'].dtype, copy=True)
    arr[~keep] = np.nan
    out['s2'] = arr
    return out


def _fit_one_2d(s2_band, stride):
    """fit_s2 with the W-only geometry parameters frozen."""
    try:
        fit = sf.fit_s2(s2_band, profile=sf.weibull_log_s2,
                        inner_uv_pixels=ss.INNER_UV, freeze=FREEZE_2D,
                        **dict(ss.FIT_KW, fit_stride=stride))
        rec = sf._fit_scalars(fit['params'])
        rec['fit_success'] = bool(getattr(fit.get('fit'), 'success', True))
        res = fit.get('fit')
        if res is not None and getattr(res, 'fun', None) is not None:
            rec['rms_resid'] = float(np.sqrt(np.mean(np.asarray(res.fun) ** 2)))
            rec['n_fit'] = int(np.asarray(res.fun).size)
        return rec
    except Exception as exc:
        return {'error': repr(exc)}


def _fit_profile_2d(data, edges, stride, r_grid=None):
    """Fit every in-plane band on one image realisation."""
    s2 = _dw0_only(sf.compute_s2(data, **ss.COMPUTE_KW))
    if r_grid is None:
        r_grid = ss._lag_radius_grid(s2)
    rows = []
    for lo, hi in edges:
        rec = _fit_one_2d(ss._banded(s2, r_grid, lo, hi), stride)
        if 'error' not in rec and rec.get('fit_success'):
            rec['b2b1'] = rec['b2'] / rec['b1'] if rec['b1'] > 0 else float('nan')
        rec['r_lo'], rec['r_hi'] = float(lo), float(hi)
        rec['r_mid'] = float(np.sqrt(lo * hi))
        rows.append(rec)
    return rows, r_grid


def _one_window(spec):
    row, col, size, stride, out_dir, band_dex = spec
    path = os.path.join(out_dir, f'sp2_r{row}_c{col}_s{size}.json')
    if os.path.exists(path):
        return path, 'cached'
    t0 = time.time()
    # NB sf.read_window signature is (row0, col0, nrows, ncols) -- row FIRST.
    data = sf.read_window(row, col, size, size,
                          data_dir=os.path.join(_ROOT, 'data'), **ss.READ_KW)
    s2_full = _dw0_only(sf.compute_s2(data, **ss.COMPUTE_KW))
    radii, _ = ss._fit_point_radii(s2_full, stride)
    edges = sp.band_edges(radii, band_dex)
    del s2_full

    central, r_grid = _fit_profile_2d(data, edges, stride)
    samples = []
    for i in range(K):
        for j in range(K):
            d2 = ss._delete_block(data, i, j, K)
            rows, _ = _fit_profile_2d(d2, edges, stride, r_grid=r_grid)
            samples.append(dict(block=[i, j], bands=rows))

    out = dict(row=row, col=col, size=size, k=K, fit_stride=stride,
               profile='weibull', method='scale_profile_2d', band_dex=band_dex,
               frozen=list(FREEZE_2D), n_bands=len(edges),
               bands=central, samples=samples,
               slopes={k: dict(zip(('slope', 'intercept', 'n'),
                                   sp._slope(central, k)))
                       for k in ('b2b1', 'b1', 'pa2d', 'alpha')},
               elapsed_s=round(time.time() - t0, 1))
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh)
    os.replace(tmp, path)
    return path, 'done'


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    which = sys.argv[3] if len(sys.argv) > 3 else 'q4'
    band_dex = float(sys.argv[4]) if len(sys.argv) > 4 else BAND_DEX

    wl = json.load(open(f'{_ROOT}/handoff/{which}_windows.json'))
    specs_raw = wl['specs'] if isinstance(wl, dict) else wl
    out_dir = f'{_ROOT}/data/scale_profile_2d_d{band_dex:g}_s{stride}'
    os.makedirs(out_dir, exist_ok=True)
    specs = [(int(r), int(c), int(s), stride, out_dir, band_dex)
             for r, c, s in specs_raw]

    print(f'{len(specs)} windows, stride {stride}, band {band_dex} dex, '
          f'{n_workers} workers, UV-plane only -> {out_dir}', flush=True)
    with Pool(n_workers) as pool:
        for n, (path, how) in enumerate(pool.imap_unordered(_one_window, specs), 1):
            print(f'[{n}/{len(specs)}] {how} {os.path.basename(path)}', flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
