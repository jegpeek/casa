"""2x2 block jackknife on the top-SNR-quartile windows.

Differs from structure_function._jackknife_fit in one way that matters here: it
KEEPS the per-sample parameter vectors instead of collapsing them to a
per-quantity stderr.  Errors on ratios (a2/a1, a3/a2) cannot be recovered from
marginal stderrs on a1/a2/a3 -- the axes are strongly covariant, so the ratio
error needs the samples themselves.  Keeping them also allows a 2x2 covariance
per window, i.e. an error ellipse rather than an independent-axes cross.

Fit parameters replicate run 1 exactly (background=0.03, arcsinh_scale=0.03,
weighting='1/r', min_n_fraction=0.1, edge_mask_radius=50, min_coverage=0.25).

One output JSON per window in data/jk_q4/, so the run is resumable.
"""
import json
import os
import sys
import time


import numpy as np

import structure_function as sf

OUT_DIR = 'data/jk_q4'
K = 2  # 2x2 blocks -> N = 4 delete-one-block samples

COMPUTE_KW = dict(background=0.03, arcsinh_scale=0.03, assume_stationary=True)
FIT_KW = dict(profile=None, max_nfev=None, weighting='1/r',
              min_n_fraction=0.1, fit_stride=1)
READ_KW = dict(edge_mask_radius=50, min_coverage=0.25)


def _one_window(spec):
    """Delete each of the K*K blocks in turn, refit, and record every sample."""
    row, col, size = spec
    out_fn = f'{OUT_DIR}/jk_r{row}_c{col}_s{size}.json'
    if os.path.exists(out_fn):
        return out_fn, 'cached'

    t0 = time.time()
    data = sf.read_window(row, col, size, size, data_dir='data', **READ_KW)
    flux = data['flux_epochs']
    _, ny, nx = flux.shape
    r_edges = np.linspace(0, ny, K + 1).astype(int)
    c_edges = np.linspace(0, nx, K + 1).astype(int)

    samples = []
    for i in range(K):
        for j in range(K):
            d = dict(data)
            f = flux.copy()
            f[:, r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]] = np.nan
            d['flux_epochs'] = f
            try:
                s2 = sf.compute_s2(d, **COMPUTE_KW)
                fit = sf.fit_s2(s2, **FIT_KW)
                rec = sf._fit_scalars(fit['params'])
                rec['block'] = [i, j]
                rec['fit_success'] = bool(getattr(fit.get('fit'), 'success',
                                                  True))
                samples.append(rec)
            except Exception as exc:            # keep going; record the gap
                samples.append({'block': [i, j], 'error': repr(exc)})

    rec = dict(row=row, col=col, size=size, k=K,
               samples=samples, wall_s=time.time() - t0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_fn, 'w') as fh:
        json.dump(rec, fh)
    return out_fn, f'{time.time() - t0:.0f}s'


if __name__ == '__main__':
    specs = json.load(open('handoff/q4_windows.json'))['specs']
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'{len(specs)} windows x {K*K} refits, {n_workers} workers',
          flush=True)

    # multiprocessing.Pool (fork), matching process_chunks -- the stdlib
    # ProcessPoolExecutor probes SC_SEM_NSEMS_MAX at startup, which this
    # sandbox denies with PermissionError.
    import multiprocessing

    done = 0
    with multiprocessing.Pool(n_workers) as pool:
        for spec, res in zip(specs, pool.imap(_one_window, specs)):
            done += 1
            fn, msg = res
            print(f'[{done}/{len(specs)}] r{spec[0]}_c{spec[1]} {msg}',
                  flush=True)
    print('all done', flush=True)
