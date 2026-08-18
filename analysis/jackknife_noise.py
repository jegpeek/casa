"""2x2 block jackknife on the top-SNR-quartile windows, with a choice of profile.

Same windows, same fit settings, and the same delete-one-block scheme as
jackknife_q4.py, but the profile is selectable so weibull_log_s2 (3 param) and
weibull_noise_log_s2 (4 param, fitted noise pedestal) can be compared under
identical conditions.  This is the controlled test of whether fitting the noise
floor removes the geometry bias.

Unlike jackknife_q4.py this also records the CENTRAL (all-data) fit in the same
JSON, so each window's file is self-contained: central fit + K*K jackknife
samples.

Usage:  python analysis/jackknife_noise.py <profile> [n_workers] [stride] [windows]
          profile = 'weibull' | 'noise'
          stride  = fit_stride (default 1); >1 subsamples lag pixels
          windows = 'q4' (29 top-SNR) | 'all' (115)

One output JSON per window in data/jk_<profile>[_s<stride>]/, so the run is
resumable.  The stride is encoded in the directory name so that runs at
different strides can never be mixed in the same analysis.
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


def _fit_one(data, profile, stride=1):
    """Fit and flatten to scalars, or return the error.

    `stride` is passed in per-call rather than read from the module-level
    FIT_KW because multiprocessing uses the *spawn* start method on macOS:
    workers re-import this module and would see the unmodified default.
    """
    try:
        s2 = sf.compute_s2(data, **COMPUTE_KW)
        fit = sf.fit_s2(s2, profile=profile, **dict(FIT_KW, fit_stride=stride))
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
    row, col, size, tag, suffix, stride = spec
    profile = PROFILES[tag]
    out_dir = f'data/jk_{tag}{suffix}'
    out_fn = f'{out_dir}/jk_r{row}_c{col}_s{size}.json'
    if os.path.exists(out_fn):
        return out_fn, 'cached'

    t0 = time.time()
    data = sf.read_window(row, col, size, size, data_dir='data', **READ_KW)
    flux = data['flux_epochs']
    _, ny, nx = flux.shape
    r_edges = np.linspace(0, ny, K + 1).astype(int)
    c_edges = np.linspace(0, nx, K + 1).astype(int)

    central = _fit_one(data, profile, stride)

    samples = []
    for i in range(K):
        for j in range(K):
            d = dict(data)
            f = flux.copy()
            f[:, r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]] = np.nan
            d['flux_epochs'] = f
            rec = _fit_one(d, profile, stride)
            rec['block'] = [i, j]
            samples.append(rec)

    rec = dict(row=row, col=col, size=size, k=K, profile=tag,
               fit_stride=stride,
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
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    which = sys.argv[4] if len(sys.argv) > 4 else 'q4'

    suffix = '' if stride == 1 else f'_s{stride}'
    win_file = {'q4': 'handoff/q4_windows.json',
                'all': 'handoff/all115_windows.json'}[which]

    specs = [tuple(s) + (tag, suffix, stride)
             for s in json.load(open(win_file))['specs']]
    os.makedirs(f'data/jk_{tag}{suffix}', exist_ok=True)
    print(f'{len(specs)} windows x (1 central + {K*K} jackknife) refits, '
          f'profile={tag}, stride={stride}, windows={which}, '
          f'{n_workers} workers -> data/jk_{tag}{suffix}/', flush=True)

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
