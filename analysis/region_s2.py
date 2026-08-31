"""3D structure function over a user-defined interior region.

Reads a polygon (data/interior_region.json, written by pick_region.py), loads
its whole bounding box across all 5 epochs (the 5 W-slices) in one piece, masks
every pixel outside the polygon to NaN, and runs compute_s2 ONCE on the whole
masked region.  Because compute_s2's FFT cross-correlation is global, this
single pass counts EVERY valid pixel pair inside the clicked domain out to the
requested lag -- including pairs that straddle any arbitrary sub-block boundary,
which an earlier block-tiled version silently dropped.  It also uses a single
global mean and variance over the region rather than a per-block one.

compute_s2(..., maxlag_px=) crops the returned lag plane to |dU|,|dV| <= maxlag
so the large-field call returns a small array (a full uncropped lag plane for a
multi-Mpix region would be many GB, virtually all of it lags we never use).

Results are binned by 3D separation r = sqrt(dU^2 + dV^2 + dW^2) out to the same
cap the normal-window profiles use (R_HI_SAMPLED = 0.5 ly), with a separate
in-plane-only curve (same-epoch pairs, dW = 0).
"""
import json, os
import numpy as np
from matplotlib.path import Path

import structure_function as sf

RMAX = 0.5007          # ly — matches R_HI_SAMPLED in make_turnover_profiles.py
RMIN = 8e-3            # ly — matches phys_profile rmin
NBIN = 14             # matches phys_profile nbin
MAXLAG_PX = 320       # px — lag crop; 0.5 ly / 0.0016 ly/pix = 312.5, +margin
N_EPOCHS = 5
S2_KW = dict(background=0.03, arcsinh_scale=0.03,
             subtract_mean='global', assume_stationary=True,
             clip_percentiles=(0.002, 0.998))


def load_polygon(path):
    d = json.load(open(path))
    v = np.asarray(d['vertices_col_row'], float)   # [col, row]
    return v, d


def accumulate(poly_path, data_dir, maxlag_px=MAXLAG_PX, verbose=True):
    """Load the whole polygon bbox, mask to the polygon, run compute_s2 once.

    Returns a dict with the region s2 grid (n_pairs, 2*ml+1, 2*ml+1), n_counts,
    lag_du, lag_dv, lag_dw, epoch_pairs, plus bookkeeping.  For downstream
    binning the key 'dsq' = s2 * n_counts is provided (so bin_by_separation is
    shared with the old tiled path unchanged).
    """
    verts, meta = load_polygon(poly_path)
    vcol, vrow = verts[:, 0], verts[:, 1]
    path = Path(verts)                              # (col, row) order
    H, W = meta['field_shape_HxW']

    r0, r1 = int(np.floor(vrow.min())), int(np.ceil(vrow.max()))
    c0, c1 = int(np.floor(vcol.min())), int(np.ceil(vcol.max()))
    r1 = min(r1, H); c1 = min(c1, W)
    bbox = (r0, r1, c0, c1)
    if verbose:
        print(f'  bbox rows(V) {r0}:{r1} ({r1 - r0}) '
              f'cols(U) {c0}:{c1} ({c1 - c0})', flush=True)

    # in-polygon mask over the whole bbox, in absolute pixel coords
    cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    inside = path.contains_points(
        np.column_stack([cc.ravel(), rr.ravel()])
    ).reshape(rr.shape)
    total_pix = int(inside.sum())
    if verbose:
        print(f'  in-polygon pixels: {total_pix} '
              f'({100 * total_pix / inside.size:.1f}% of bbox)', flush=True)

    data = sf._read_noclip_region(
        data_dir, slice(r0, r1), slice(c0, c1),
        tuple(range(N_EPOCHS)), stride=1)
    flux = data['flux_epochs']                      # (5, nr, nc)
    flux[:, ~inside] = np.nan                        # blank outside the polygon
    data['flux_epochs'] = flux

    res = sf.compute_s2(data, maxlag_px=maxlag_px, **S2_KW)
    s2, N = res['s2'], res['n_counts']
    dsq = np.where(N > 0, s2 * N, 0.0)

    return dict(dsq=dsq, n_counts=N,
                lag_du=res['lag_du'], lag_dv=res['lag_dv'],
                lag_dw=res['lag_dw'], epoch_pairs=res['epoch_pairs'],
                n_blocks=1, n_pix=total_pix, bbox=bbox, meta=meta)


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
