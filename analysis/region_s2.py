"""3D structure function averaged over a user-defined interior region.

Reads a polygon (data/interior_region.json, written by pick_region.py), tiles
its bounding box into 400-px blocks (the normal window size), masks each block
to the polygon (out-of-region pixels -> NaN), runs compute_s2 on all 5 epochs
(the 5 W-slices), and accumulates the summed squared-differences and pair
counts per lag across blocks.  Because compute_s2 stores s2 = dsq / N per lag,
the accumulation is  sum(s2 * N) / sum(N)  -- an N-weighted average over the
whole region, reproducing the per-window lag treatment exactly.

Results are binned by 3D separation r = sqrt(dU^2 + dV^2 + dW^2) out to the
same cap the normal-window profiles use (R_HI_SAMPLED = 0.5 ly), with a
separate in-plane-only curve (same-epoch pairs, dW = 0).
"""
import json, os
import numpy as np
from matplotlib.path import Path

import structure_function as sf

RMAX = 0.5007          # ly — matches R_HI_SAMPLED in make_turnover_profiles.py
RMIN = 8e-3            # ly — matches phys_profile rmin
NBIN = 14             # matches phys_profile nbin
BLOCK = 400           # px — normal window size
N_EPOCHS = 5
S2_KW = dict(background=0.03, arcsinh_scale=0.03,
             subtract_mean='global', assume_stationary=True,
             clip_percentiles=(0.002, 0.998))


def load_polygon(path):
    d = json.load(open(path))
    v = np.asarray(d['vertices_col_row'], float)   # [col, row]
    return v, d


def block_grid(vcol, vrow, block=BLOCK):
    """Non-overlapping block origins (row0, col0) tiling the polygon bbox."""
    r0, r1 = int(np.floor(vrow.min())), int(np.ceil(vrow.max()))
    c0, c1 = int(np.floor(vcol.min())), int(np.ceil(vcol.max()))
    rows = list(range(r0, r1, block))
    cols = list(range(c0, c1, block))
    return rows, cols, (r0, r1, c0, c1)


def accumulate(poly_path, data_dir, block=BLOCK, min_frac=0.02, verbose=True):
    """Tile the region, run compute_s2 per block, accumulate dsq & N per lag.

    Returns a dict with the accumulated s2 grid (n_pairs, 2b-1, 2b-1), n_counts,
    lag_du, lag_dv, lag_dw, epoch_pairs, plus bookkeeping.
    """
    verts, meta = load_polygon(poly_path)
    vcol, vrow = verts[:, 0], verts[:, 1]
    path = Path(verts)                              # (col, row) order
    H, W = meta['field_shape_HxW']

    rows, cols, bbox = block_grid(vcol, vrow, block)
    acc_dsq = acc_N = None
    lag_du = lag_dv = lag_dw = epoch_pairs = None
    n_blocks_used = 0
    total_pix = 0

    for r0 in rows:
        r1 = min(r0 + block, H)
        for c0 in cols:
            c1 = min(c0 + block, W)
            # in-polygon mask for this block, in absolute pixel coords
            cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
            inside = path.contains_points(
                np.column_stack([cc.ravel(), rr.ravel()])
            ).reshape(rr.shape)
            npix = int(inside.sum())
            if npix < min_frac * (block * block):
                continue

            data = sf._read_noclip_region(
                data_dir, slice(r0, r1), slice(c0, c1),
                tuple(range(N_EPOCHS)), stride=1)
            flux = data['flux_epochs']              # (5, nr, nc)
            # blank everything outside the polygon
            flux[:, ~inside] = np.nan
            data['flux_epochs'] = flux

            res = sf.compute_s2(data, **S2_KW)
            s2, N = res['s2'], res['n_counts']
            dsq = np.where(N > 0, s2 * N, 0.0)

            if acc_dsq is None:
                acc_dsq = np.zeros_like(dsq, dtype=np.float64)
                acc_N = np.zeros_like(N, dtype=np.int64)
                lag_du, lag_dv = res['lag_du'], res['lag_dv']
                lag_dw, epoch_pairs = res['lag_dw'], res['epoch_pairs']
            # blocks can differ in size at the field edge; only accumulate
            # matching-shape blocks (interior blocks are full BLOCK x BLOCK)
            if dsq.shape == acc_dsq.shape:
                acc_dsq += dsq
                acc_N += N
                n_blocks_used += 1
                total_pix += npix
            if verbose:
                print(f'  block r{r0} c{c0}: {npix} in-poly px', flush=True)

    return dict(dsq=acc_dsq, n_counts=acc_N, lag_du=lag_du, lag_dv=lag_dv,
                lag_dw=lag_dw, epoch_pairs=epoch_pairs,
                n_blocks=n_blocks_used, n_pix=total_pix, bbox=bbox, meta=meta)


def bin_by_separation(acc, rmin=RMIN, rmax=RMAX, nbin=NBIN):
    """N-weighted bin of accumulated dsq by 3D separation and by in-plane-only.

    Returns dict of arrays (bin centers + S2 + N) for 'all3d' and 'inplane'.
    """
    du, dv, dw = acc['lag_du'], acc['lag_dv'], acc['lag_dw']
    pairs = acc['epoch_pairs']
    DU, DV = np.meshgrid(du, dv)                    # (nrow, ncol) lag planes

    edges = np.geomspace(rmin, rmax, nbin + 1)
    ctr = np.sqrt(edges[:-1] * edges[1:])

    sum_dsq_3d = np.zeros(nbin); sum_N_3d = np.zeros(nbin)
    sum_dsq_ip = np.zeros(nbin); sum_N_ip = np.zeros(nbin)

    for k, (i, j) in enumerate(pairs):
        dsq_k = acc['dsq'][k]; N_k = acc['n_counts'][k]
        good = N_k > 0
        r3d = np.sqrt(DU**2 + DV**2 + dw[k]**2)
        idx = np.digitize(r3d, edges) - 1
        sel = good & (idx >= 0) & (idx < nbin)
        np.add.at(sum_dsq_3d, idx[sel], dsq_k[sel])
        np.add.at(sum_N_3d, idx[sel], N_k[sel])
        if i == j:                                 # same-epoch => dW = 0 (in-plane)
            rip = np.sqrt(DU**2 + DV**2)
            idxi = np.digitize(rip, edges) - 1
            seli = good & (idxi >= 0) & (idxi < nbin)
            np.add.at(sum_dsq_ip, idxi[seli], dsq_k[seli])
            np.add.at(sum_N_ip, idxi[seli], N_k[seli])

    with np.errstate(invalid='ignore', divide='ignore'):
        s2_3d = sum_dsq_3d / sum_N_3d
        s2_ip = sum_dsq_ip / sum_N_ip
    return dict(r=ctr, edges=edges,
                s2_3d=s2_3d, N_3d=sum_N_3d,
                s2_ip=s2_ip, N_ip=sum_N_ip)
