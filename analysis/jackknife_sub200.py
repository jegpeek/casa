"""200px subwindow fits + 2x2 block jackknife, inside the high-SNR 400px windows.

Each of the 29 top-SNR-quartile 400px windows is split into four 200px
quadrants (116 subwindows).  For each subwindow we run:
  - one full fit on all its data   -> the central value
  - four delete-one-block refits   -> the 2x2 jackknife (100px blocks)

so 5 fits per subwindow.  Unlike jackknife_q4.py (which reused run 1's
full-window fit as the central value) the central fit is computed here,
because no 200px fits exist yet.

Per-sample parameter vectors are kept, not collapsed to a stderr, so that
errors on the RATIOS a2/a1 and a3/a2 -- and their covariance -- can be formed.

Fit parameters replicate run 1 exactly.  One JSON per subwindow, so resumable.

CAVEAT recorded here because it governs interpretation: the same-epoch S2 radial
profile of the parent 400px windows has only reached a median 75% of its
(plateau - floor) amplitude by 60-90px lag.  A 200px window therefore does not
span the correlation length, and var_inf is extrapolated rather than measured.
Axis RATIOS are the robust output; absolute axes are not.
"""
import json
import os
import sys
import time

import numpy as np

import structure_function as sf

OUT_DIR = 'data/jk_sub200'
K = 2
SIZE = 200

COMPUTE_KW = dict(background=0.03, arcsinh_scale=0.03, assume_stationary=True)
# inner_uv_pixels=100 -> fit lags out to half the window, the same PROPORTION
# run 1 used (200 of 399 lag pixels on a 400px window).  This matters: letting
# the fit use the near-full 200px lag extent instead sends a1 to 17 ly on the
# first subwindow tested, versus 0.24 ly at half-extent.
FIT_KW = dict(profile=None, max_nfev=None, weighting='1/r',
              min_n_fraction=0.1, fit_stride=1, inner_uv_pixels=100)
READ_KW = dict(edge_mask_radius=50, min_coverage=0.25)


def _scalars(fit):
    rec = sf._fit_scalars(fit['params'])
    rec['fit_success'] = bool(getattr(fit.get('fit'), 'success', True))
    return rec


def _one_subwindow(spec):
    row, col, size = spec
    out_fn = f'{OUT_DIR}/jk_r{row}_c{col}_s{size}.json'
    if os.path.exists(out_fn):
        return out_fn, 'cached'

    t0 = time.time()
    data = sf.read_window(row, col, size, size, data_dir='data', **READ_KW)

    # --- central fit on the whole subwindow
    try:
        s2_full = sf.compute_s2(data, **COMPUTE_KW)
        fit_full = sf.fit_s2(s2_full, **FIT_KW)
        central = _scalars(fit_full)
        # _summary_record returns a TUPLE matching _summary_dtype(profile), not
        # a dict — so pull chi2_dof out by its field position.
        dt = sf._summary_dtype(sf.weibull_log_s2)
        rec_t = sf._summary_record(s2_full, fit_full, sf.weibull_log_s2,
                                   'data', 'data')
        central['chi2_dof'] = float(
            np.array([tuple(rec_t)], dtype=dt)['chi2_dof'][0])
    except Exception as exc:
        central = {'error': repr(exc)}

    # --- 2x2 delete-one-block jackknife
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
                rec = _scalars(sf.fit_s2(sf.compute_s2(d, **COMPUTE_KW),
                                         **FIT_KW))
                rec['block'] = [i, j]
                samples.append(rec)
            except Exception as exc:
                samples.append({'block': [i, j], 'error': repr(exc)})

    rec = dict(row=row, col=col, size=size, k=K, central=central,
               samples=samples, wall_s=time.time() - t0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_fn, 'w') as fh:
        json.dump(rec, fh)
    return out_fn, f'{time.time() - t0:.0f}s'


if __name__ == '__main__':
    specs = json.load(open('handoff/sub200_windows.json'))['specs']
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'{len(specs)} subwindows x {1 + K*K} fits, {n_workers} workers',
          flush=True)

    # multiprocessing.Pool (fork): ProcessPoolExecutor is denied in this sandbox.
    import multiprocessing

    done = 0
    with multiprocessing.Pool(n_workers) as pool:
        for spec, res in zip(specs, pool.imap(_one_subwindow, specs)):
            done += 1
            fn, msg = res
            print(f'[{done}/{len(specs)}] r{spec[0]}_c{spec[1]} {msg}',
                  flush=True)
    print('all done', flush=True)
