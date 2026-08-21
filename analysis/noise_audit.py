"""Measure the per-window noise floor of S2 and relate it to fit quality.

Independent per-pixel noise contributes a constant 2*sigma_n^2 to S2 at every
non-zero lag, while the ISM signal contribution falls to zero as lag -> 0.  So
the same-epoch (dW=0) structure function at a 1-pixel lag is an empirical
estimate of the noise floor, and its rise to the large-lag plateau measures the
signal.  Both are in arcsinh-transformed units, i.e. the units actually fitted.

Writes noise_audit.csv with one row per window.
"""
import glob
import os

import h5py
import numpy as np

INNER_UV = 200          # fit_s2's inner_uv_pixels default
R_FLOOR = (0.9, 1.8)    # pixel-radius band taken as the noise floor
R_PLATEAU = (150, 200)  # pixel-radius band taken as the signal plateau


def radial_profile(plane, weights, rpix, nbins_edges):
    """Count-weighted mean of `plane` in the given radial bins."""
    idx = np.digitize(rpix.ravel(), nbins_edges) - 1
    good = np.isfinite(plane.ravel()) & (weights.ravel() > 0) & (idx >= 0) \
        & (idx < len(nbins_edges) - 1)
    i = idx[good]
    num = np.bincount(i, weights=(plane.ravel() * weights.ravel())[good],
                      minlength=len(nbins_edges) - 1)
    den = np.bincount(i, weights=weights.ravel()[good],
                      minlength=len(nbins_edges) - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def audit_s2(s2d, row, col, size_ly=None, flux_mad_std=np.nan, chunk_id=-1):
    """Audit an in-memory compute_s2() result, no HDF5 cache required.

    `audit_file` below reads the same quantities out of a `data/sf_fits/*.h5`
    cache of `compute_s2` outputs.  That cache is not part of the fork's tracked
    data, so this entry point exists to let the SNR tiering be regenerated from
    the raw arrays via read_window + compute_s2.  Verified to reproduce
    `results/noise_audit_table.csv` to 5+ decimals on spot-checked windows.
    """
    pairs = np.asarray(s2d['epoch_pairs'])
    same = np.flatnonzero(pairs[:, 0] == pairs[:, 1])
    s2 = np.asarray(s2d['s2'][same.min():same.max() + 1], dtype=np.float64)
    nc = np.asarray(s2d['n_counts'][same.min():same.max() + 1],
                    dtype=np.float64)
    lag_du = np.asarray(s2d['lag_du'])
    return _audit_planes(s2, nc, lag_du, row, col, chunk_id, size_ly,
                         flux_mad_std)


def _audit_planes(s2, nc, lag_du, row, col, cid, size_ly, mad):
    """Shared core: count-weighted dW=0 mean -> radial profile -> floor/plateau.

    s2 = dsq/N per plane, so sum(s2*N)/sum(N) = sum(dsq)/sum(N).
    """
    num = np.nansum(np.where(np.isfinite(s2), s2 * nc, 0.0), axis=0)
    den = np.nansum(np.where(np.isfinite(s2), nc, 0.0), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        s2_avg = num / den
    s2_avg[den <= 0] = np.nan

    ny, nx = s2_avg.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rpix = np.hypot(yy - ny // 2, xx - nx // 2)

    edges = np.concatenate([[0.0], np.geomspace(0.9, INNER_UV, 40)])
    prof = radial_profile(s2_avg, den, rpix, edges)
    centres = np.sqrt(edges[:-1] * np.maximum(edges[1:], 1e-9))

    def band(lo, hi):
        m = (centres >= lo) & (centres <= hi) & np.isfinite(prof)
        return float(np.nanmean(prof[m])) if m.any() else np.nan

    floor = band(*R_FLOOR)
    plateau = band(*R_PLATEAU)
    # Signal-to-noise of the *structure*: excess over the floor, per floor.
    snr2 = (plateau - floor) / floor if floor and np.isfinite(floor) else np.nan
    pix_ly = float(abs(lag_du[1] - lag_du[0]))

    return dict(row=row, col=col, chunk_id=cid, size_ly=size_ly,
                pix_ly=pix_ly, flux_mad_std=mad,
                s2_floor=floor, s2_plateau=plateau,
                snr2=snr2, snr=np.sqrt(snr2) if snr2 > 0 else np.nan,
                floor_frac=floor / plateau if plateau else np.nan)


def audit_window(spec):
    """Pool worker: audit one window from the raw arrays, with a JSON cache.

    spec = (row, col, size, data_dir, cache_dir).  Module level and taking a
    single picklable tuple because macOS multiprocessing uses `spawn`, which
    re-imports the worker by qualified name -- a closure or a function defined
    in a notebook cell cannot be dispatched to a Pool.
    """
    import json
    import structure_function as sf
    import scale_split as ss

    row, col, size, data_dir, cache_dir = spec
    cache = os.path.join(cache_dir, 'audit_r%d_c%d.json' % (row, col))
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh), 'cached'
    data = sf.read_window(row, col, size, size, data_dir=data_dir,
                          **ss.READ_KW)
    rec = audit_s2(sf.compute_s2(data, **ss.COMPUTE_KW), row, col)
    rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in rec.items()}
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache, 'w') as fh:
        json.dump(rec, fh)
    return rec, 'computed'


def audit_file(path):
    with h5py.File(path, 'r') as f:
        row = int(f.attrs['row'])
        col = int(f.attrs['col'])
        cid = int(f.attrs['chunk_id'])
        pairs = f['sf/epoch_pairs'][:]
        same = np.flatnonzero(pairs[:, 0] == pairs[:, 1])
        # Count-weighted mean over the same-epoch (dW=0) planes:
        # s2 = dsq/N per plane, so sum(s2*N)/sum(N) = sum(dsq)/sum(N).
        s2 = f['sf/s2'][same.min():same.max() + 1].astype(np.float64)
        nc = f['sf/n_counts'][same.min():same.max() + 1].astype(np.float64)
        mad = float(f['sf'].attrs['flux_mad_std'])
        size_ly = float(f['sf'].attrs['size_ly'])
        lag_du = f['sf/lag_du'][:]

    return _audit_planes(s2, nc, lag_du, row, col, cid, size_ly, mad)


if __name__ == '__main__':
    paths = sorted(glob.glob('data/sf_fits/sf_fit_r*_c*_s*.h5'))
    print(f'{len(paths)} result files')
    rows = []
    for n, p in enumerate(paths, 1):
        try:
            rows.append(audit_file(p))
        except Exception as exc:
            print(f'FAILED {os.path.basename(p)}: {exc}')
        if n % 20 == 0:
            print(f'  {n}/{len(paths)}')

    keys = list(rows[0])
    with open('noise_audit.csv', 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(f'{r[k]:.6g}' if isinstance(r[k], float)
                              else str(r[k]) for k in keys) + '\n')
    print(f'wrote noise_audit.csv with {len(rows)} rows')
