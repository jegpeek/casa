"""Block bootstrap of the 3D structure function fit, k x k blocks per window.

Replaces the delete-one-block jackknife (which yields only k*k resamples and
forces a normal-theory error bar) with a proper bootstrap: blocks are resampled
WITH replacement, so we can draw as many replicates as we can afford and read
off percentile intervals, covariances and distribution shape directly.

See block_bootstrap.py for the resampling rule and the algebra that makes each
replicate cheap.  The short version: S2's numerator and denominator are both
bilinear in the block multiplicities, so the FFT work is done ONCE per window
and every replicate is a weighted sum of precomputed block-pair arrays.  The
cost per replicate is then dominated entirely by fit_s2.

Parallelism is over WINDOWS, not replicates: the precomputed block-pair arrays
are ~360 MB at k=3 and cannot be shared between processes (macOS spawns rather
than forks), so each worker builds one set and reuses it for all B replicates.
Peak RSS is ~1 GB per worker at k=3; do not exceed 8 workers on a 16 GB machine.

Usage:  python analysis/bootstrap_windows.py [n_boot] [n_workers] [stride] [windows] [k]
          n_boot    number of bootstrap replicates per window (default 100)
          n_workers worker processes (default 6)
          stride    fit_stride (default 2)
          windows   'q4' (29 top-SNR) | 'all' (115)  (default q4)
          k         blocks per side (default 3)

One JSON per window in data/bs_k<k>_B<n_boot>_s<stride>/, so the run is
resumable and runs at different settings can never be mixed.
"""
import json
import os
import sys
import time
import zlib

import numpy as np

# Run from anywhere: the repo root holds structure_function/block_bootstrap and
# is not on sys.path when this is invoked as analysis/bootstrap_windows.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structure_function as sf
import block_bootstrap as bb


READ_KW = dict(edge_mask_radius=50, min_coverage=0.25)
# These are written out in full but are VALUE-IDENTICAL to jackknife_noise.py's
# settings: that driver leaves clip_percentiles / fill_nans / subtract_mean at
# their defaults, and this one leaves max_nfev / weighting at theirs.  Verified
# empirically -- the central fit for r2800_c2000 reproduces the completed
# jackknife run to 4 decimals in both axis ratios.  Keep them in sync.
COMPUTE_KW = dict(clip_percentiles=(0.002, 0.998), fill_nans=False,
                  assume_stationary=True,
                  background=0.03, arcsinh_scale=0.03, subtract_mean='global')
# precompute_blocks implements the stationary formula unconditionally (it is the
# only mode the profiles use), so it takes no assume_stationary argument.
PRE_KW = {k: v for k, v in COMPUTE_KW.items() if k != 'assume_stationary'}
FIT_KW = dict(inner_uv_pixels=200, min_same_epoch_lag_pix=4,
              min_n_fraction=0.1, max_nfev=None, weighting='1/r')
# All windows are 400 px, so 200 is the half-width used by every prior run.
# inner_uv_pixels must scale WITH window size (it is a strong lever on the fit,
# not a detail): a 200 px window would need 100 here.
INNER_UV = 200


def _scalars(fit):
    """Flatten a fit to the scalars we track, or return the error."""
    rec = sf._fit_scalars(fit['params'])
    rec['fit_success'] = bool(getattr(fit.get('fit'), 'success', True))
    res = fit.get('fit')
    if res is not None and getattr(res, 'fun', None) is not None:
        rec['rms_resid'] = float(np.sqrt(np.mean(np.asarray(res.fun) ** 2)))
    return rec


def _one_window(spec):
    """Bootstrap one window.  All parameters travel in `spec`.

    Nothing is read from module-level state that the caller may have mutated:
    multiprocessing uses the *spawn* start method on macOS, so workers re-import
    this module and would see the original defaults, silently producing results
    labelled with settings they were not run at.
    """
    row, col, size, n_boot, stride, k, seed, out_dir = spec
    out_fn = f'{out_dir}/bs_r{row}_c{col}_s{size}.json'
    if os.path.exists(out_fn):
        return out_fn, 'cached'

    t0 = time.time()
    profile = sf.weibull_log_s2
    fitkw = dict(FIT_KW, fit_stride=stride)

    data = sf.read_window(row, col, size, size, data_dir='data', **READ_KW)

    # Central fit uses the ORIGINAL compute_s2, not the block machinery, so the
    # central value is directly comparable to every earlier run.
    try:
        central = _scalars(sf.fit_s2(sf.compute_s2(data, **COMPUTE_KW),
                                     profile=profile, **fitkw))
    except Exception as exc:
        central = {'error': repr(exc)}

    pre = bb.precompute_blocks(data, k=k, inner_uv_pixels=INNER_UV, **PRE_KW)
    del data

    rng = np.random.default_rng(seed)
    samples = []
    for b in range(n_boot):
        m = bb.draw_multiplicities(pre['K'], rng)
        try:
            rep = bb.s2_from_weights(pre, m)
            rec = _scalars(sf.fit_s2(rep, profile=profile, **fitkw))
        except Exception as exc:
            rec = {'error': repr(exc)}
        rec['m'] = [int(x) for x in m]
        samples.append(rec)

    out = dict(row=row, col=col, size=size, k=k, n_boot=n_boot,
               profile='weibull', fit_stride=stride, seed=seed,
               method='block_bootstrap_multinomial',
               central=central, samples=samples, wall_s=time.time() - t0)
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_fn + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh)
    os.replace(tmp, out_fn)          # atomic: readers never see a partial file
    return out_fn, f'{time.time() - t0:.0f}s'


def main():
    import multiprocessing as mp

    n_boot    = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    stride    = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    which     = sys.argv[4] if len(sys.argv) > 4 else 'q4'
    k         = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    win_file = {'q4': 'handoff/q4_windows.json',
                'all': 'handoff/all115_windows.json'}[which]
    with open(win_file) as fh:
        wins = json.load(fh)['specs']       # list of [row, col, size]

    out_dir = f'data/bs_k{k}_B{n_boot}_s{stride}'
    os.makedirs(out_dir, exist_ok=True)

    # Per-window seeds derived from position: reproducible, and independent of
    # how the work happens to be scheduled across workers.  zlib.crc32 rather
    # than hash(), which is salted per interpreter and so is NOT reproducible
    # across runs.
    specs = [(row, col, size, n_boot, stride, k,
              zlib.crc32(f'{row}_{col}_{k}_{n_boot}'.encode()), out_dir)
             for row, col, size in wins]

    print(f'{len(specs)} windows x {n_boot} bootstrap replicates, k={k}, '
          f'stride={stride}, {n_workers} workers -> {out_dir}/', flush=True)

    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers) as pool:
        for i, (fn, msg) in enumerate(pool.imap_unordered(_one_window, specs), 1):
            el = time.time() - t0
            print(f'[{i}/{len(specs)}] {os.path.basename(fn)} {msg} '
                  f'(elapsed {el/60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
