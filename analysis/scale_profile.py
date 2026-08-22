"""Oblate-vs-prolate as a continuous function of scale.

The two-band test (analysis/scale_split.py) showed the principal-axis
ORIENTATION and the ratio a2/a1 are scale-invariant, while a3/a2 is not.  This
script replaces that two-point comparison with a continuous one: fit the
ellipsoid independently in overlapping bands of log lag radius and report the
triaxiality

    T = (a1^2 - a2^2) / (a1^2 - a3^2)      T -> 0 oblate (pancake)
                                           T -> 1 prolate (cigar)

as a function of band centre, plus the log-slope dT/dlog r.

Why T alone is not enough here
------------------------------
This cloud is strongly elongated: the committed full-band fits give a2/a1 ~
0.29 and T ~ 0.94 in every window, i.e. firmly on the prolate side.  T
SATURATES in that regime.  At the observed shape, T moves only 0.014 across the
entire observed range of a3/a2 (0.32 -> 0.52) but 0.052 across a plausible
range of a2/a1 (0.25 -> 0.35) -- about 4x more sensitive to a2/a1 than to
a3/a2.  Since a2/a1 is the scale-INVARIANT ratio and a3/a2 is the
scale-DEPENDENT one, T is expected to look nearly flat even though the shape
genuinely changes.  T is therefore reported together with:

    a3/a1 = (a3/a2)(a2/a1)    overall flattening, does not saturate
    a2/a1, a3/a2             the two ratios separately

so a flat T curve can be distinguished from "nothing changes".

What is and is not measurable in a band
---------------------------------------
At small lag S2 -> var_inf * |L^-1 . dr|^alpha, so rescaling L is exactly
absorbed by var_inf: the ABSOLUTE axis lengths are near-degenerate within a
narrow band and must not be compared between bands.  Axis RATIOS, T, and
orientation are not degenerate.  tests/test_band_recovery.py injects a known
ellipsoid into the real lag geometry and confirms every band recovers the
injected ratios (T to <0.01), including the innermost band where dW quantisation
leaves only 3 distinct |dW| values -- that test is what licenses this curve.

Method
------
Bands are BAND_DEX wide in log10 radius, stepped BAND_DEX/2, so consecutive
bands overlap by half and the curve is smooth but each point is
half-independent.  Band edges are computed from the full-band fit point set of
each window and held FIXED across the jackknife refits, so bands stay
comparable when a spatial block is deleted.  Errors come from the same 2x2
spatial block jackknife used everywhere else in this project, applied to the
per-band quantity AND to the fitted slope, so the slope error accounts for the
correlation between bands (they share the same images).

Bands are masked by NaN-ing s2 outside the band before fit_s2 sees it:
_make_fit_data selects on np.isfinite, so no shared code path changes.

Usage:  python analysis/scale_profile.py [n_workers] [stride] [windows] [band_dex]
          defaults: 8 2 q4 0.6
One JSON per window in data/scale_profile_d<band_dex>_s<stride>/, resumable.
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

K = 2                      # 2x2 spatial block jackknife, as everywhere else
BAND_DEX = 0.6
_GEOM = ('a1', 'a2', 'a3', 'theta', 'phi', 'psi')


def triaxiality(a1, a2, a3):
    """T = (a1^2-a2^2)/(a1^2-a3^2).  0 = oblate (pancake), 1 = prolate (cigar)."""
    denom = a1 ** 2 - a3 ** 2
    if denom <= 0:
        return float('nan')
    return float((a1 ** 2 - a2 ** 2) / denom)


def prolateness(a1, a2, a3):
    """p = ln(a1 a3 / a2^2) = ln(a1/a2) - ln(a2/a3).

    A better-conditioned oblate/prolate coordinate than T for this cloud:

      p > 0   prolate  (a2 sits close to a3: cigar)
      p = 0   maximally triaxial (a2 = sqrt(a1 a3))
      p < 0   oblate   (a2 sits close to a1: pancake)

    Unlike T = (a1^2-a2^2)/(a1^2-a3^2), p does not saturate.  T is pinned near 1
    for this cloud (median 0.94, no oblate window in the sample) and at the
    observed elongation responds ~4x more strongly to a2/a1 -- the ratio that is
    scale-INVARIANT here -- than to a3/a2, the one that is not.  p weights the
    two log ratios equally and is symmetric between the oblate and prolate
    limits, so a change in either ratio shows up undiluted.
    """
    if not (a1 > 0 and a2 > 0 and a3 > 0):
        return float('nan')
    return float(np.log(a1) + np.log(a3) - 2.0 * np.log(a2))


def band_edges(radii, band_dex):
    """Overlapping [lo, hi) band edges in log10 radius, stepped band_dex/2."""
    lo0 = np.log10(radii[radii > 0].min())
    hi0 = np.log10(radii.max())
    out = []
    lo = lo0
    while lo < hi0 - band_dex / 2:
        out.append((10 ** lo, 10 ** min(lo + band_dex, hi0 + 1e-12)))
        lo += band_dex / 2
    return out


def _fit_profile(data, edges, stride, r_grid=None, profile='weibull'):
    """Fit every band on one image realisation.  Returns list of per-band dicts.

    `profile` selects the 1D form fit inside each band (see
    scale_split.BAND_PROFILES).  'weibull' is the historical default; note that
    within a single 0.6-dex band the Weibull's saturation scale is never
    sampled (a1 exceeds the band's own outer radius in >99% of fits), so its
    beta pins to a bound in ~63% of bands.  'powerlaw' fits the small-r limit
    the Weibull is reducing to anyway, with one free profile parameter.
    """
    s2 = sf.compute_s2(data, **ss.COMPUTE_KW)
    if r_grid is None:
        r_grid = ss._lag_radius_grid(s2)
    rows = []
    for lo, hi in edges:
        rec = ss._fit_one(ss._banded(s2, r_grid, lo, hi), stride, profile=profile)
        if 'error' not in rec and rec.get('fit_success'):
            a1, a2, a3 = rec['a1'], rec['a2'], rec['a3']
            rec['a2a1'] = a2 / a1
            rec['a3a2'] = a3 / a2
            rec['a3a1'] = a3 / a1
            rec['T'] = triaxiality(a1, a2, a3)
            rec['p'] = prolateness(a1, a2, a3)
        rec['r_lo'], rec['r_hi'] = float(lo), float(hi)
        rec['r_mid'] = float(np.sqrt(lo * hi))
        rows.append(rec)
    return rows, r_grid


def _slope(rows, key):
    """Least-squares slope of key vs log10(r_mid) over converged bands."""
    x, y = [], []
    for r in rows:
        if r.get('fit_success') and np.isfinite(r.get(key, np.nan)):
            x.append(np.log10(r['r_mid']))
            y.append(r[key])
    if len(x) < 3:
        return float('nan'), float('nan'), 0
    A = np.polyfit(x, y, 1)
    return float(A[0]), float(A[1]), len(x)


def _one_window(spec):
    row, col, size, stride, out_dir, band_dex, profile = spec
    path = os.path.join(out_dir, f'sp_r{row}_c{col}_s{size}.json')
    if os.path.exists(path):
        return path, 'cached'
    t0 = time.time()
    # NB sf.read_window signature is (row0, col0, nrows, ncols) -- row FIRST.
    data = sf.read_window(row, col, size, size,
                          data_dir=os.path.join(_ROOT, 'data'), **ss.READ_KW)
    s2_full = sf.compute_s2(data, **ss.COMPUTE_KW)
    radii, _ = ss._fit_point_radii(s2_full, stride)
    edges = band_edges(radii, band_dex)
    del s2_full

    central, r_grid = _fit_profile(data, edges, stride, profile=profile)
    samples = []
    for i in range(K):
        for j in range(K):
            d2 = ss._delete_block(data, i, j, K)
            rows, _ = _fit_profile(d2, edges, stride, r_grid=r_grid,
                                   profile=profile)
            samples.append(dict(block=[i, j], bands=rows))

    out = dict(row=row, col=col, size=size, k=K, fit_stride=stride,
               profile=profile, method='scale_profile', band_dex=band_dex,
               n_bands=len(edges), bands=central, samples=samples,
               slopes={k: dict(zip(('slope', 'intercept', 'n'), _slope(central, k)))
                       for k in ('T', 'p', 'a2a1', 'a3a2', 'a3a1', 'alpha')},
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
    profile = sys.argv[5] if len(sys.argv) > 5 else 'weibull'
    if profile not in ss.BAND_PROFILES:
        raise SystemExit('profile must be one of %s' % sorted(ss.BAND_PROFILES))

    wl = json.load(open(f'{_ROOT}/handoff/{which}_windows.json'))
    specs_raw = wl['specs'] if isinstance(wl, dict) else wl
    # The default output tree keeps its historical name so existing caches and
    # the tracked result CSVs stay valid; a non-default profile gets a suffix.
    suffix = '' if profile == 'weibull' else f'_{profile}'
    out_dir = f'{_ROOT}/data/scale_profile_d{band_dex:g}_s{stride}{suffix}'
    os.makedirs(out_dir, exist_ok=True)
    specs = [(int(r), int(c), int(s), stride, out_dir, band_dex, profile)
             for r, c, s in specs_raw]

    print(f'{len(specs)} windows, stride {stride}, band {band_dex} dex, '
          f'profile {profile}, {n_workers} workers -> {out_dir}', flush=True)
    with Pool(n_workers) as pool:
        for n, (path, how) in enumerate(pool.imap_unordered(_one_window, specs), 1):
            print(f'[{n}/{len(specs)}] {how} {os.path.basename(path)}', flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
