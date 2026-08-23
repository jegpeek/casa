"""Single-band (r < R_CUT) ellipsoid fits with jackknife errors, power law vs Weibull.

Motivation
----------
analysis/scale_profile.py fits overlapping 0.6-dex bands out to the largest lag
available.  The structure function turns over near 0.1 ly, and above that the
outer bands are fitting a regime the ellipsoid model does not describe.  This
driver instead fits ONE band, from the smallest lag up to R_CUT = 0.1 ly.

On that range the Weibull profile is over-parameterised.  Since
1 - exp(-r^beta) -> r^beta as r -> 0, weibull_log_s2 tends to var_inf * r^alpha
with NO beta dependence, so beta and var_inf are not separately constrained
unless the saturation scale a1 is actually sampled -- and a1 exceeds its own
band's outer radius in 144 of 145 band fits.  sf.power_law_log_s2 is that limit
with two fewer parameters.

Both profiles are fitted here, in 3D and on the dW=0 slice, each with a 2x2
block jackknife, so the profile choice can be judged against measurement error
rather than against chi^2 (whose reduced value is ~6, i.e. the errors are
underestimated and AIC is not trustworthy at face value).

The power law carries an exact scale degeneracy (all axes by k, A by k^alpha),
so A is frozen -- see the power_law_log_s2 docstring.  Read axis RATIOS and
angles from these fits, never the absolute a1.

Usage
-----
    python analysis/singleband_powerlaw.py [--rcut 0.1] [--stride 2]
                                           [--size 400] [--procs 6]

Writes results/singleband_powerlaw_r<rcut>_s<stride>.csv -- one row per
(window, mode, profile) with the central value and jackknife SE of every
tracked quantity.
"""
import argparse
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

K_DEFAULT = 2               # 2x2 spatial block jackknife, as everywhere else
FREEZE_2D = ('s33', 'l13', 'l23')

# Quantities carried through the jackknife.  Ratios and angles only for the
# power law -- a1/a2/a3 are defined only up to a common factor there.
TRACK_3D = ('a2a1', 'a3a2', 'a3a1', 'T', 'p', 'b2b1', 'pa2d',
            'theta', 'phi', 'psi', 'incl', 'alpha')
TRACK_2D = ('b2b1', 'pa2d', 'alpha')

#: Profile-specific parameters, exported only on rows of that profile.
#: The Weibull's `beta` and `var_inf` are FIT and stored in the per-window
#: JSONs, but were not being written to the summary CSV, so a reader could not
#: see that beta sits at a bound in most windows -- the very reason the power
#: law was adopted as canonical.  They are the diagnostic, so they are exported.
TRACK_PROFILE = {'weibull': ('beta', 'var_inf'), 'powerlaw': ()}

PROFILES = {'powerlaw': sf.power_law_log_s2, 'weibull': sf.weibull_log_s2}


def _derived(rec):
    """Add ratios, shape coordinates and inclination to a _fit_scalars dict."""
    a1, a2, a3 = rec.get('a1'), rec.get('a2'), rec.get('a3')
    if a1 and a2 and a3 and a1 > 0 and a2 > 0 and a3 > 0:
        rec['a2a1'] = a2 / a1
        rec['a3a2'] = a3 / a2
        rec['a3a1'] = a3 / a1
        rec['T'] = sp.triaxiality(a1, a2, a3)
        rec['p'] = sp.prolateness(a1, a2, a3)
    # theta is the polar angle of a1 from W, so the inclination of the long
    # axis out of the echo plane is 90 - theta, folded to [0, 90].
    th = rec.get('theta')
    if th is not None and np.isfinite(th):
        rec['incl'] = float(90.0 - abs(90.0 - (float(th) % 180.0)))
    return rec


def _fit_band(s2_band, stride, profile, freeze):
    """One fit_s2 call, flattened to scalars.  Never raises."""
    try:
        fit = sf.fit_s2(s2_band, profile=profile, freeze=freeze,
                        inner_uv_pixels=ss.INNER_UV,
                        **dict(ss.FIT_KW, fit_stride=stride))
        rec = _derived(sf._fit_scalars(fit['params']))
        rec['fit_success'] = bool(getattr(fit.get('fit'), 'success', True))
        res = fit.get('fit')
        if res is not None and getattr(res, 'fun', None) is not None:
            fun = np.asarray(res.fun)
            rec['rms_resid'] = float(np.sqrt(np.mean(fun ** 2)))
            rec['n_fit'] = int(fun.size)
            rec['chi2'] = float(np.sum(fun ** 2))
        return rec
    except Exception as exc:
        return {'error': repr(exc), 'fit_success': False}


def _fit_all_modes(data, rcut, stride, r_grid=None):
    """Fit {mode x profile} on one image realisation.

    Modes: '3d' (full lag cube) and '2d' (dW=0 slice, W-only geometry frozen).
    """
    s2 = sf.compute_s2(data, **ss.COMPUTE_KW)
    if r_grid is None:
        r_grid = ss._lag_radius_grid(s2)
    out = {}
    band3 = ss._banded(s2, r_grid, 0.0, rcut)
    band2 = ss._banded(_dw0(s2), r_grid, 0.0, rcut)
    for name, profile in PROFILES.items():
        extra = ('A',) if name == 'powerlaw' else ()
        out[('3d', name)] = _fit_band(band3, stride, profile, extra)
        out[('2d', name)] = _fit_band(band2, stride, profile,
                                      tuple(FREEZE_2D) + extra)
    return out, r_grid


def _dw0(s2):
    """Shallow copy of s2 with every dW != 0 plane NaN-ed out."""
    dw = np.asarray(s2['lag_dw'], dtype=float)
    keep = (dw == 0.0)
    if not keep.any():
        raise ValueError('no dW=0 planes in this s2 result')
    out = dict(s2)
    arr = np.array(s2['s2'], dtype=s2['s2'].dtype, copy=True)
    arr[~keep] = np.nan
    out['s2'] = arr
    return out


def _one_window(spec):
    row, col, size, stride, rcut, out_dir, k_blocks = spec
    path = os.path.join(out_dir, f'sb_r{row}_c{col}_s{size}.json')
    if os.path.exists(path):
        return path, 'cached'
    t0 = time.time()
    # NB sf.read_window signature is (row0, col0, nrows, ncols) -- row FIRST.
    data = sf.read_window(row, col, size, size,
                          data_dir=os.path.join(_ROOT, 'data'), **ss.READ_KW)
    central, r_grid = _fit_all_modes(data, rcut, stride)
    samples = []
    for i in range(k_blocks):
        for j in range(k_blocks):
            d2 = ss._delete_block(data, i, j, k_blocks)
            fits, _ = _fit_all_modes(d2, rcut, stride, r_grid=r_grid)
            rec = {'%s|%s' % kk: v for kk, v in fits.items()}
            rec['block'] = [i, j]
            samples.append(rec)

    out = dict(row=row, col=col, size=size, k=k_blocks, fit_stride=stride,
               rcut=rcut, frozen_2d=list(FREEZE_2D),
               central={'%s|%s' % k: v for k, v in central.items()},
               samples=samples, seconds=time.time() - t0)
    os.makedirs(out_dir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh)
    os.replace(tmp, path)
    return path, 'ok'


def _jk_se(values):
    """Delete-one-block jackknife SE from N block-deleted estimates."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size < 2:
        return float('nan')
    n = v.size
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def summarize(paths, csv_path):
    import csv
    keys = {'3d': TRACK_3D, '2d': TRACK_2D}
    rows = []
    for p in paths:
        with open(p) as fh:
            rec = json.load(fh)
        for tag, cen in rec['central'].items():
            mode, profile = tag.split('|')
            r = dict(chunk='r%d_c%d' % (rec['row'], rec['col']),
                     row=rec['row'], col=rec['col'], mode=mode,
                     profile=profile, rcut=rec['rcut'],
                     fit_success=cen.get('fit_success', False),
                     error=cen.get('error', ''),
                     rms_resid=cen.get('rms_resid', float('nan')),
                     n_fit=cen.get('n_fit', 0),
                     chi2=cen.get('chi2', float('nan')))
            for q in keys[mode] + TRACK_PROFILE.get(profile, ()):
                r[q] = cen.get(q, float('nan'))
                r['se_' + q] = _jk_se([s.get(tag, {}).get(q)
                                       for s in rec['samples']])
            # a1 is reported for the Weibull only -- degenerate for the power law
            r['a1'] = cen.get('a1', float('nan')) if profile == 'weibull' \
                else float('nan')
            rows.append(r)
    cols = sorted({k for r in rows for k in r})
    head = ['chunk', 'row', 'col', 'mode', 'profile']
    cols = head + [c for c in cols if c not in head]
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rcut', type=float, default=0.1,
                    help='outer radius of the single band [LIGHT YEARS]')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--size', type=int, default=400)
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--k', type=int, default=K_DEFAULT,
                    help='jackknife blocking: k x k spatial blocks, k^2 refits '
                         'per window. NB the output directory encodes k, so '
                         'different k never share a cache.')
    ap.add_argument('--windows', default=None,
                    help='JSON file with [[row, col], ...]; default = the '
                         'top-SNR-quartile set used elsewhere')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    if args.windows:
        with open(args.windows) as fh:
            obj = json.load(fh)
        # Accept either a bare [[row, col], ...] list or {'specs': [[row, col,
        # size], ...]} as written by the window-selection step.  Only the first
        # two entries are used; --size sets the window size.
        raw = obj['specs'] if isinstance(obj, dict) else obj
        specs = [(int(x[0]), int(x[1])) for x in raw]
    else:
        raise SystemExit('--windows is required (pass the window list as JSON)')

    tag = 'r%g_s%d' % (args.rcut, args.stride)
    if args.k != K_DEFAULT:
        tag += '_k%d' % args.k
    out_dir = args.out_dir or os.path.join(_ROOT, 'results',
                                           'singleband_%s' % tag)
    jobs = [(r, c, args.size, args.stride, args.rcut, out_dir, args.k)
            for r, c in specs]
    t0 = time.time()
    paths = []
    with Pool(args.procs) as pool:
        for i, (path, status) in enumerate(
                pool.imap_unordered(_one_window, jobs)):
            paths.append(path)
            print('  %2d/%d %s %s  %.0fs' % (i + 1, len(jobs),
                                             os.path.basename(path), status,
                                             time.time() - t0), flush=True)
    csv_path = os.path.join(_ROOT, 'results',
                            'singleband_powerlaw_%s.csv' % tag)
    rows = summarize(sorted(paths), csv_path)
    print('wrote %s (%d rows)' % (csv_path, len(rows)))


if __name__ == '__main__':
    main()
