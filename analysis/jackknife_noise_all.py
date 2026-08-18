"""2x2 block jackknife on the top-SNR-quartile windows, with a choice of profile.

Same windows, same fit settings, and the same delete-one-block scheme as
jackknife_q4.py, but the profile is selectable so weibull_log_s2 (3 param) and
weibull_noise_log_s2 (4 param, fitted noise pedestal) can be compared under
identical conditions.  This is the controlled test of whether fitting the noise
floor removes the geometry bias.

Unlike jackknife_q4.py this also records the CENTRAL (all-data) fit in the same
JSON, so each window's file is self-contained: central fit + K*K jackknife
samples.

Usage:  python analysis/jackknife_noise.py <profile> [n_workers]
          profile = 'weibull' | 'noise'

One output JSON per window in data/jk_<profile>/, so the run is resumable.
"""
import json
import os
import sys
import time

import numpy as np

import structure_function as sf

K = 2  # 2x2 blocks -> N = 4 delete-one-block samples

COMPUTE_KW = dict(background=0.03, arcsinh_scale=0.03, assume_stationary=True)
FIT_KW = dict(max_nfev=None, weighting='1/r', min_n_fraction=0.1, fit_stride=1)
READ_KW = dict(edge_mask_radius=50, min_coverage=0.25)

PROFILES = {'weibull': sf.weibull_log_s2, 'noise': sf.weibull_noise_log_s2}


def _fit_one(data, profile):
    """Fit and flatten to scalars, or return the error."""
    try:
        s2 = sf.compute_s2(data, **COMPUTE_KW)
        fit = sf.fit_s2(s2, profile=profile, **FIT_KW)
        rec = sf._fit_scalars(fit['params'])
        # keep the fitted noise floor and a goodness measure for diagnostics
        rec['fit_success'] = bool(getattr(fit.get('fit'), 'success', True))
        res = fit.get('fit')
        if res is not None and getattr(res, 'fun', None) is not None:
            rec['rms_resid'] = float(np.sqrt(np.mean(np.asarray(res.fun) ** 2)))
        return rec
    except Exception as exc:
        return {'error': repr(exc)}


def _one_window(spec):
    row, col, size, tag = spec
    profile = PROFILES[tag]
    out_dir = f'data/jk_{tag}'
    out_fn = f'{out_dir}/jk_r{row}_c{col}_s{size}.json'
    if os.path.exists(out_fn):
        return out_fn, 'cached'

    t0 = time.time()
    data = sf.read_window(row, col, size, size, data_dir='data', **READ_KW)
    flux = data['flux_epochs']
    _, ny, nx = flux.shape
    r_edges = np.linspace(0, ny, K + 1).astype(int)
    c_edges = np.linspace(0, nx, K + 1).astype(int)

    central = _fit_one(data, profile)

    samples = []
    for i in range(K):
        for j in range(K):
            d = dict(data)
            f = flux.copy()
            f[:, r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]] = np.nan
            d['flux_epochs'] = f
            rec = _fit_one(d, profile)
            rec['block'] = [i, j]
            samples.append(rec)

    rec = dict(row=row, col=col, size=size, k=K, profile=tag,
               central=central, samples=samples, wall_s=time.time() - t0)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_fn, 'w') as fh:
        json.dump(rec, fh)
    return out_fn, f'{time.time() - t0:.0f}s'


if __name__ == '__main__':
    tag = sys.argv[1]
    if tag not in PROFILES:
        raise SystemExit(f'profile must be one of {sorted(PROFILES)}')
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    specs = [tuple(s) + (tag,)
             for s in json.load(open('handoff/all115_windows.json'))['specs']]
    os.makedirs(f'data/jk_{tag}', exist_ok=True)
    print(f'{len(specs)} windows x (1 central + {K*K} jackknife) '
          f'refits, profile={tag}, {n_workers} workers', flush=True)

    # multiprocessing.Pool (fork), matching process_chunks -- the stdlib
    # ProcessPoolExecutor probes SC_SEM_NSEMS_MAX at startup, which this
    # sandbox denies with PermissionError.
    import multiprocessing

    done = 0
    with multiprocessing.Pool(n_workers) as pool:
        for spec, res in zip(specs, pool.imap_unordered(_one_window, specs)):
            done += 1
            fn, msg = res
            print(f'[{done}/{len(specs)}] {os.path.basename(fn)} {msg}',
                  flush=True)
    print('all done', flush=True)
