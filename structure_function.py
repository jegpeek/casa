"""
Compute 3D structure functions S2(dU, dV, dW) from UVW-chunked HDF5 files.

S2(lag) = (1/N) * sum_x [ I(x + lag) - I(x) ]^2

where the sum is over all pixel pairs that are both non-NaN, and N counts
those pairs.  NaN pixels and off-edge lags are excluded; N is adjusted
accordingly so S2 is never biased by missing data.

Output shape: (n_pairs, 2*n_rows-1, 2*n_cols-1)
  axis 0 : epoch pair — same-epoch pairs (i,i) first (dW=0), then cross-epoch
            pairs (i,j) with i<j in lexicographic order
  axis 1 : V-direction lag index, centered: index 0 = most-negative lag,
            middle index = lag 0, last index = most-positive lag
  axis 2 : U-direction lag index, same convention

lag_dv and lag_du carry the corresponding physical lags in light-years,
monotonically increasing from negative to positive.
"""

import sys
import os
import re
sys.path.insert(0, os.path.expanduser('~/projects/util_efs/python'))

import numpy as np
import h5py
from scipy.fft import rfft2, irfft2, next_fast_len
from itertools import combinations
import scipy.linalg
import scipy.ndimage
from scipy.optimize import least_squares
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import util_efs

LY_PER_PC = 3.2616          # light-years per parsec
ARCSINH_SCALE = 0.03        # default arcsinh / background noise scale [flux units]

# Per-epoch sky background subtracted from resampled_epochs_noclip.npy when the
# uvw_chunk_*_products.h5 files were built (a single scalar per epoch — recovered
# exactly, i.e. noclip_slice - chunk has zero spatial variance).  Subtracting
# these reproduces the chunk flux from the fullsky noclip map to float round-off;
# earlier 3-decimal values left a ~5e-4 offset.
NOCLIP_BACKGROUNDS = np.array([0.34245232539640424, 0.313639571526185,
                               0.311547736890913, 0.3160668003620078,
                               0.3930647945949676])

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

# The 115 official chunks are named windows into the full image, defined by
# data/chunk_windows.csv (chunk_id, row, col, size) rather than by the retired
# uvw_chunk_*_products.h5 files.  chunk_id -> (row, col, size); a chunk is the
# size x size pixel block at (row, col).  size is stored per chunk so the tile
# size is free to change.

def _chunk_id(chunk):
    """Chunk id as int from an id, a bare id string, or a legacy
    uvw_chunk_<id>_products.h5 path/name."""
    if isinstance(chunk, (int, np.integer)):
        return int(chunk)
    s = str(chunk)
    m = re.search(r'chunk_(\d+)', s)
    return int(m.group(1)) if m else int(s)


def _load_chunk_windows(data_dir='data'):
    """Parse data/chunk_windows.csv into {chunk_id: (row, col, size)}."""
    tbl = {}
    with open(f'{data_dir}/chunk_windows.csv') as fh:
        next(fh)                                    # header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cid, row, col, size = (int(x) for x in line.split(','))
            tbl[cid] = (row, col, size)
    return tbl


def chunk_window(chunk, data_dir='data'):
    """(row, col, size) of a chunk from data/chunk_windows.csv."""
    return _load_chunk_windows(data_dir)[_chunk_id(chunk)]


def chunk_ids(data_dir='data'):
    """Sorted list of the official chunk ids (the rows of chunk_windows.csv)."""
    return sorted(_load_chunk_windows(data_dir))


def chunk_path(chunk_id, data_dir='data'):
    """Canonical name/key for a chunk: data/uvw_chunk_<id>_products.h5.  Kept as
    a stable string key for results/plots even though no such file is read."""
    return f'{data_dir}/uvw_chunk_{chunk_id}_products.h5'


# --- Windows: (row, col, size) is the primary identity for SF analysis. -----
# The official 115 chunks are one set of windows (chunk_windows.csv); the batch
# pipeline also runs arbitrary grids (e.g. overlapping 400px windows on a 200px
# stride).  Results are keyed / named by geometry, not by chunk index.

def _window_spec(item, data_dir='data'):
    """Normalise a window spec to (row, col, size).  Accepts a (row, col, size)
    triple directly, or a chunk id / name / legacy path (looked up in the CSV)."""
    if isinstance(item, (tuple, list)) and len(item) == 3:
        return (int(item[0]), int(item[1]), int(item[2]))
    return chunk_window(item, data_dir)


def window_result_path(row, col, size, data_dir='data'):
    """Geometry-based output path for a window's SF fit (replaces the old
    uvw_chunk_<id>_sf_fit.h5 naming)."""
    return f'{data_dir}/sf_fit_r{row}_c{col}_s{size}.h5'


def window_chunk_id(row, col, size, data_dir='data'):
    """Official chunk id whose window equals (row, col, size), or -1 if none —
    a convenience so one can still refer to a window by its old chunk index."""
    for cid, rcs in _load_chunk_windows(data_dir).items():
        if rcs == (row, col, size):
            return cid
    return -1


def window_coverage(row, col, size, data_dir='data', edge_mask_radius=50):
    """Fraction of finite pixels across all epochs in a window (the cov_all cut
    used to select windows).  Equals the mean of the fullmap edge mask over the
    window, since read_window's finite pixels are exactly the mask-valid ones."""
    mask = _fullsky_edge_mask(data_dir, edge_mask_radius)
    return float(np.asarray(mask[:, row:row + size, col:col + size]).mean())


def official_windows(data_dir='data'):
    """(row, col, size) for the 115 official chunks, in chunk-id order."""
    return [chunk_window(i, data_dir) for i in chunk_ids(data_dir)]


def window_grid(size=400, stride=200, coverage_min=0.62, data_dir='data',
                edge_mask_radius=50):
    """Windows of `size` px on a `stride` px grid over the full image, keeping
    those with all-epoch coverage > coverage_min.  Default 400px/200px reproduces
    the old footprint ~4x over (each region seen by up to four windows); the
    0.62 cut reproduces the official 115 on the non-overlapping 400px grid."""
    mask = _fullsky_edge_mask(data_dir, edge_mask_radius)
    _, H, W = mask.shape
    specs = []
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            if np.asarray(mask[:, r:r + size, c:c + size]).mean() > coverage_min:
                specs.append((r, c, size))
    return specs


def chunk_corners(chunk_id, coords='uv', data_dir='data'):
    """
    Return the corner coordinates of a chunk in the full image.

    Parameters
    ----------
    chunk_id : int, str, or legacy uvw_chunk_<id>_products.h5 path.
    coords : 'uv' or 'pixel'
        'uv'    — (u_lo, u_hi, v_lo, v_hi) in light-years
        'pixel' — (row_lo, row_hi, col_lo, col_hi) as integer slice bounds
                  into the fullsky arrays (row_hi and col_hi are exclusive)
    data_dir : str
        Directory containing chunk_windows.csv and U_grid.npy / V_grid.npy.
    """
    row, col, size = chunk_window(chunk_id, data_dir)
    row_lo, row_hi, col_lo, col_hi = row, row + size, col, col + size

    if coords == 'pixel':
        return row_lo, row_hi, col_lo, col_hi

    U = np.load(f'{data_dir}/U_grid.npy', mmap_mode='r')
    V = np.load(f'{data_dir}/V_grid.npy', mmap_mode='r')
    u_lo, u_hi = float(U[row_lo, col_lo]), float(U[row_lo, col_hi - 1])
    v_lo, v_hi = float(V[row_lo, col_lo]), float(V[row_hi - 1, col_lo])
    return u_lo, u_hi, v_lo, v_hi


def _epoch_idx(epochs, n_all):
    """Normalise an epochs selector (None / int / iterable) to a list of ints."""
    if epochs is None:
        return list(range(n_all))
    if isinstance(epochs, (int, np.integer)):
        return [int(epochs)]
    return list(epochs)


def _fullsky_edge_mask(data_dir='data', edge_mask_radius=50, rebuild=False):
    """Valid-pixel mask (n_epochs, H, W) bool for the full noclip map: True
    where a pixel survives edge erosion, False for bad pixels (NaN/zero) and
    everything within edge_mask_radius of them.

    Computed once on the whole map so the mask is window-invariant — a pixel's
    fate no longer depends on which window views it (the reason to mask the
    fullmap rather than each chunk).  The erosion removes the interpolation-
    contaminated band near the fullmap boundary, where resampling filled edge
    pixels with near-duplicate nearest-edge values.  Cached to
    data/edge_mask_r<radius>.npy and mmap-reused; pass rebuild=True after the
    noclip map or the radius definition changes.

    Per-window low-coverage blanking is NOT here (it is regional by nature — see
    _blank_low_coverage / read_window).
    """
    cache = f'{data_dir}/edge_mask_r{edge_mask_radius}.npy'
    if not rebuild and os.path.exists(cache):
        return np.load(cache, mmap_mode='r')

    flux = np.load(f'{data_dir}/resampled_epochs_noclip.npy', mmap_mode='r')
    mask = np.empty(flux.shape, dtype=bool)
    for ep in range(flux.shape[0]):
        f = np.asarray(flux[ep])
        bad = ~np.isfinite(f) | (f == 0)
        if edge_mask_radius > 0 and bad.any():
            dist = scipy.ndimage.distance_transform_edt(~bad)
            mask[ep] = dist > edge_mask_radius
        else:
            mask[ep] = ~bad if edge_mask_radius > 0 else True
    np.save(cache, mask)
    return np.load(cache, mmap_mode='r')


def _blank_low_coverage(flux, min_coverage=0.25):
    """Blank (set NaN) in place any epoch whose valid fraction within this window
    is below min_coverage.  Regional by nature, so applied per window rather than
    on the fullmap (unlike edge erosion; see _fullsky_edge_mask)."""
    if min_coverage <= 0:
        return flux
    total = flux.shape[1] * flux.shape[2]
    for ep in range(flux.shape[0]):
        if np.sum(np.isfinite(flux[ep])) / total < min_coverage:
            flux[ep] = np.nan
    return flux


def read_chunk(chunk, edge_mask_radius: int = 50,
               min_coverage: float = 0.25, data_dir=None) -> dict:
    """
    Read one official chunk as a window into the fullsky noclip map.

    `chunk` is a chunk id (int), a bare id string, or a legacy
    uvw_chunk_<id>_products.h5 path (the id is parsed from it; the file is not
    read).  The chunk's (row, col, size) comes from data/chunk_windows.csv and
    the pixels from resampled_epochs_noclip.npy — see read_window, which this
    wraps.  data_dir defaults to the chunk path's directory, else 'data'.

    Returns a dict with keys:
      flux_epochs : (n_epochs, n_rows, n_cols) float64  — may contain NaNs
      U_grid      : (n_rows, n_cols) float64            — U coords [ly]
      V_grid      : (n_rows, n_cols) float64            — V coords [ly]
      W_values    : (n_epochs,) float64                 — W coord per epoch [ly]

    Edge erosion (edge_mask_radius) is applied on the fullmap once and cut to
    the window; epochs with fewer than min_coverage * n_pixels valid pixels in
    the window are then blanked entirely.
    """
    if data_dir is None:
        data_dir = (os.path.dirname(chunk) if isinstance(chunk, str)
                    and os.path.dirname(chunk) else 'data')
    row, col, size = chunk_window(chunk, data_dir)
    return read_window(row, col, size, size, data_dir=data_dir,
                       edge_mask_radius=edge_mask_radius,
                       min_coverage=min_coverage)


def read_fullmap(data_dir='data', epochs=None, stride=1) -> dict:
    """
    Read the full-image noclip data and return a dict matching read_chunk().

    Loads resampled_epochs_noclip.npy, U_grid.npy, V_grid.npy, and W_values
    (epoch_mean_w.npy).  Subtracts NOCLIP_BACKGROUNDS so the flux scale matches
    the chunk files (background ~ 0 in off-cloud regions).

    Parameters
    ----------
    epochs : int, list of int, or None
        Epoch indices to load.  None (default) loads all 5 epochs.  Pass a
        single int or a list to reduce memory and limit compute_s2 to only
        the selected epoch pairs (e.g. epochs=2 for just the middle epoch).
    stride : int
        Spatial downsampling factor applied to both axes (default 1 = full
        resolution).  stride=2 halves each dimension, reducing FFT cost
        ~16× and making compute_s2 ~15× faster at the cost of discarding
        sub-pixel lags below stride * pixel_ly.

    Returns
    -------
    dict with keys flux_epochs, U_grid, V_grid, W_values  (same as read_chunk)
    """
    return _read_noclip_region(data_dir, slice(None), slice(None),
                               epochs, stride)


def _read_noclip_region(data_dir, row_slice, col_slice, epochs, stride):
    """Cut a region out of the fullsky noclip map and return a read_chunk-style
    dict, WITHOUT edge masking (read_window applies it; read_fullmap wants none).

    Slices via memory-map so a window materialises only its own pixels.  The
    per-epoch sky background (NOCLIP_BACKGROUNDS) is subtracted so the flux
    matches the old chunk files, and W_values come from epoch_mean_w.npy (equal
    to every chunk's stored W).  Striding is applied last, after any masking a
    caller does, matching read_chunk (which does not stride).
    """
    flux_mm = np.load(f'{data_dir}/resampled_epochs_noclip.npy', mmap_mode='r')
    idx = _epoch_idx(epochs, flux_mm.shape[0])
    flux = np.asarray(flux_mm[:, row_slice, col_slice], dtype=float)[idx]
    flux -= NOCLIP_BACKGROUNDS[idx][:, None, None]

    U_grid = np.asarray(np.load(f'{data_dir}/U_grid.npy',
                                mmap_mode='r')[row_slice, col_slice])
    V_grid = np.asarray(np.load(f'{data_dir}/V_grid.npy',
                                mmap_mode='r')[row_slice, col_slice])
    W_values = np.load(f'{data_dir}/epoch_mean_w.npy')[idx]

    if stride != 1:
        flux   = flux[:, ::stride, ::stride]
        U_grid = U_grid[::stride, ::stride]
        V_grid = V_grid[::stride, ::stride]

    return {
        "flux_epochs": flux,
        "U_grid":      U_grid,
        "V_grid":      V_grid,
        "W_values":    W_values,
    }


def read_window(row0, col0, nrows=400, ncols=400, data_dir='data',
                epochs=None, stride=1, edge_mask_radius=50,
                min_coverage=0.25) -> dict:
    """Cut a window out of the fullsky noclip map, as a read_chunk-style dict.

    This is the fullsky + window replacement for the fixed uvw_chunk_*.h5 files:
    for a 400x400 window at a chunk's (row0, col0) it reproduces that chunk's
    raw_data flux/U/V/W from the fullsky map to floating-point round-off.  Edge
    erosion is taken from the window-invariant fullmap mask (_fullsky_edge_mask)
    rather than eroded per window, then low-coverage epochs are blanked for this
    window.  Windows may be any size/placement, not just the historical grid.

    row0, col0 : top-left pixel of the window in the full 6326x6274 image.
    nrows, ncols : window size in pixels (default 400x400, the chunk size).
    See read_chunk for edge_mask_radius / min_coverage and read_fullmap for
    epochs / stride.
    """
    row_slice, col_slice = slice(row0, row0 + nrows), slice(col0, col0 + ncols)
    d = _read_noclip_region(data_dir, row_slice, col_slice, epochs, stride=1)
    flux = d['flux_epochs']
    if edge_mask_radius > 0:
        mask = _fullsky_edge_mask(data_dir, edge_mask_radius)
        idx = _epoch_idx(epochs, mask.shape[0])
        valid = np.asarray(mask[:, row_slice, col_slice])[idx]
        flux[~valid] = np.nan
    _blank_low_coverage(flux, min_coverage)
    if stride != 1:
        d['flux_epochs'] = flux[:, ::stride, ::stride]
        d['U_grid'] = d['U_grid'][::stride, ::stride]
        d['V_grid'] = d['V_grid'][::stride, ::stride]
    # window geometry (pixel origin + side) so results can key on it rather than
    # a chunk index; ncols==nrows in the analysis pipeline (square windows)
    d['row'], d['col'], d['size'] = row0, col0, nrows
    return d


# ---------------------------------------------------------------------------
# FFT cross-correlation helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _fft_xcorr_full(a_fft: np.ndarray, b_fft: np.ndarray,
                    fft_shape: tuple,
                    n_rows: int, n_cols: int) -> np.ndarray:
    """
    Full cross-correlation C[dr, dc] = sum_{r,c} a[r,c] * b[r+dr, c+dc].

    Output shape (2*n_rows-1, 2*n_cols-1) in FFT-natural order:
      [0, 0]               = lag (0, 0)
      [1..n_rows-1, ...]   = positive dV lags
      [n_rows..end, ...]   = negative dV lags -(n_rows-1)..-1  (same convention for dU)

    Inputs are pre-computed rfft2 arrays zero-padded to fft_shape, which must
    satisfy fft_shape >= (2*n_rows-1, 2*n_cols-1) to avoid circular aliasing.
    """
    c = irfft2(np.conj(a_fft) * b_fft, s=fft_shape)
    fr, fc = fft_shape
    out = np.empty((2 * n_rows - 1, 2 * n_cols - 1))

    # Quadrant layout (dV_sign, dU_sign) → source and destination slices
    # (+dV, +dU): rows 0..n_rows-1,       cols 0..n_cols-1
    out[:n_rows,  :n_cols]  = c[:n_rows,               :n_cols]
    # (-dV, +dU): rows n_rows..2n_rows-2, cols 0..n_cols-1
    out[n_rows:,  :n_cols]  = c[fr - n_rows + 1:fr,    :n_cols]
    # (+dV, -dU): rows 0..n_rows-1,       cols n_cols..2n_cols-2
    out[:n_rows,  n_cols:]  = c[:n_rows,               fc - n_cols + 1:fc]
    # (-dV, -dU): rows n_rows..2n_rows-2, cols n_cols..2n_cols-2
    out[n_rows:,  n_cols:]  = c[fr - n_rows + 1:fr,    fc - n_cols + 1:fc]

    return out


def _sum_diff_sq_full(m_i_fft, mf_i_fft, mf2_i_fft,
                      m_j_fft, mf_j_fft, mf2_j_fft,
                      fft_shape, n_rows, n_cols):
    """
    Return (sum_diff_sq, N_pairs) for epoch pair (i, j), full lag plane.

    sum_diff_sq[dr,dc] = sum_{valid pairs} (f_j(x+lag) - f_i(x))^2
    N_pairs[dr,dc]     = number of valid pairs at that lag
    """
    xcorr = lambda a, b: _fft_xcorr_full(a, b, fft_shape, n_rows, n_cols)
    N       = xcorr(m_i_fft,   m_j_fft)
    sfi2    = xcorr(mf2_i_fft, m_j_fft)
    sfj2    = xcorr(m_i_fft,   mf2_j_fft)
    scross  = xcorr(mf_i_fft,  mf_j_fft)
    return sfi2 + sfj2 - 2.0 * scross, N


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_s2(data: dict,
               clip_percentiles: tuple | None = (0.002, 0.998),
               fill_nans: bool = False,
               assume_stationary: bool = True,
               background: float | np.ndarray | None = 0.03,
               arcsinh_scale: float | None = 0.03,
               subtract_mean: str = 'global') -> dict:
    """
    Compute the 3D structure function for one chunk.

    Parameters
    ----------
    data             : dict returned by read_chunk()
    clip_percentiles : (low, high) percentiles in [0, 1] used to clip each
                       epoch's flux before computing S2, masking outlier pixels
                       as NaN.  E.g. (0.01, 0.99) clips the top and bottom 1%.
                       Pass None to skip clipping.
    fill_nans        : if True, fill NaN pixels with the per-epoch nanmean
                       before computing S2 (matches the old get_sf_2d behaviour).
                       Default False (NaN pixels are excluded and N is adjusted).
    assume_stationary: if True, use the stationary formula S2 = 2*Var - 2*Cov(r),
                       with variance pooled globally across all epochs.
                       Default True.
    background       : if not None, subtract from each epoch's flux before any
                       other processing.  Scalar: same value applied to every
                       epoch.  1-D array of length n_epochs: per-epoch values.
    arcsinh_scale    : if not None, apply arcsinh(flux / arcsinh_scale) after
                       background subtraction and clipping but before mean
                       subtraction.  Sets the transition scale between linear
                       (noise-dominated) and logarithmic (signal-dominated)
                       behaviour; typically set to the per-pixel noise level.
    subtract_mean    : how to subtract the mean from the arcsinh-transformed
                       field before computing S2.
                       'global' (default): subtract a single mean pooled across
                           all epochs and pixels; avoids catastrophic cancellation
                           in the stationary formula without removing epoch-to-epoch
                           structure.
                       'epoch': subtract the per-epoch nanmean; equivalent to
                           the old subtract_epoch_mean=True behaviour.
                       'none': no mean subtraction.

    Returns
    -------
    dict with keys:
      s2          : (n_pairs, 2*n_rows-1, 2*n_cols-1) — structure function
      n_counts    : same shape — number of contributing pixel pairs per lag
      lag_dv      : (2*n_rows-1,) — V-direction physical lags [ly], FFT order
      lag_du      : (2*n_cols-1,) — U-direction physical lags [ly], FFT order
      lag_dw      : (n_pairs,)    — W physical lags [ly]
      epoch_pairs : list of (i, j) tuples matching axis 0 of s2/n_counts

    Lag arrays use FFT-natural ordering: index 0 = lag 0, positive lags
    first, then negative lags.  Apply numpy.fft.fftshift on axes 1 & 2
    (and the corresponding lag arrays) to get a centered representation.

    Same-epoch pairs (i, i) with dW=0 are listed first (indices 0..n_epochs-1),
    followed by cross-epoch pairs (i, j) with i<j in lexicographic order.
    Post-hoc zero-W-lag average:
        s2_avg = sum(s2[:n_epochs], axis=0) / sum(n_counts[:n_epochs], axis=0)
    """
    flux = data["flux_epochs"].copy()   # (n_epochs, n_rows, n_cols)

    if background is not None:
        bg = np.broadcast_to(np.asarray(background, dtype=float),
                             (flux.shape[0],))
        for e in range(flux.shape[0]):
            flux[e] -= bg[e]

    if clip_percentiles is not None:
        lo_frac, hi_frac = clip_percentiles
        for e in range(flux.shape[0]):
            plane = flux[e]
            valid = plane[np.isfinite(plane)]
            if valid.size == 0:
                continue
            lo = np.percentile(valid, lo_frac * 100)
            hi = np.percentile(valid, hi_frac * 100)
            flux[e] = np.where((plane < lo) | (plane > hi), np.nan, plane)

    if arcsinh_scale is not None:
        flux = np.arcsinh(flux / arcsinh_scale)

    all_valid = np.concatenate([flux[e][np.isfinite(flux[e])] for e in range(flux.shape[0])])
    if all_valid.size > 0:
        med = float(np.median(all_valid))
        flux_mad_std = float(np.median(np.abs(all_valid - med)) * 1.4826)
    else:
        flux_mad_std = np.nan

    if subtract_mean == 'global':
        all_vals = np.concatenate([flux[e][np.isfinite(flux[e])] for e in range(flux.shape[0])])
        global_mean = float(all_vals.mean()) if all_vals.size > 0 else 0.0
        flux -= global_mean
    elif subtract_mean == 'epoch':
        for e in range(flux.shape[0]):
            flux[e] -= np.nanmean(flux[e])
    elif subtract_mean != 'none':
        raise ValueError(f"subtract_mean must be 'global', 'epoch', or 'none'; got {subtract_mean!r}")

    if fill_nans:
        for e in range(flux.shape[0]):
            plane = flux[e]
            flux[e] = np.where(np.isfinite(plane), plane, np.nanmean(plane))

    U_grid = data["U_grid"]
    V_grid = data["V_grid"]
    W      = data["W_values"]

    n_epochs, n_rows, n_cols = flux.shape

    # Physical lag step (assumes uniform grid; U varies along cols, V along rows)
    du_step = float(U_grid[0, 1] - U_grid[0, 0])
    dv_step = float(V_grid[1, 0] - V_grid[0, 0])

    # FFT-natural lag arrays: [0, step, ..., +max, -max, ..., -step]
    lag_du = np.empty(2 * n_cols - 1)
    lag_du[:n_cols] = np.arange(n_cols) * du_step
    lag_du[n_cols:] = np.arange(-(n_cols - 1), 0) * du_step

    lag_dv = np.empty(2 * n_rows - 1)
    lag_dv[:n_rows] = np.arange(n_rows) * dv_step
    lag_dv[n_rows:] = np.arange(-(n_rows - 1), 0) * dv_step

    # Epoch pairs: same-epoch first, then cross-epoch (i < j)
    same_pairs  = [(e, e) for e in range(n_epochs)]
    cross_pairs = list(combinations(range(n_epochs), 2))
    epoch_pairs = same_pairs + cross_pairs
    n_pairs     = len(epoch_pairs)

    lag_dw = np.array([float(W[j] - W[i]) for i, j in epoch_pairs])

    # FFT padding: need >= 2*N - 1 in each dimension to avoid aliasing
    fft_shape = (
        next_fast_len(2 * n_rows - 1),
        next_fast_len(2 * n_cols - 1),
    )

    # Pre-compute per-epoch FFTs
    masks = np.isfinite(flux).astype(np.float64)
    f_    = np.where(masks.astype(bool), flux, 0.0)

    M_fft = [rfft2(masks[e], s=fft_shape) for e in range(n_epochs)]

    if assume_stationary:
        # S2 = 2*(E[f²] - xcorr(fM,fM)/N).
        # 'global': pool all epochs for one E[f²]; with subtract_mean='global'
        #   the field is near-zero-mean so E[f²] ≈ Var(f), avoiding catastrophic
        #   cancellation from large μ².
        # 'epoch'/'none': per-epoch E[f²] (preserves old per-epoch behaviour).
        valid = [masks[e].astype(bool) for e in range(n_epochs)]
        evars = [float((f_[e][valid[e]]**2).mean()) if valid[e].any() else 0.0
                 for e in range(n_epochs)]
        MF_fft  = [rfft2(masks[e] * f_[e],   s=fft_shape) for e in range(n_epochs)]
        MF2_fft = [evars[e] * M_fft[e]                    for e in range(n_epochs)]
    else:
        MF_fft  = [rfft2(masks[e] * f_[e],    s=fft_shape) for e in range(n_epochs)]
        MF2_fft = [rfft2(masks[e] * f_[e]**2, s=fft_shape) for e in range(n_epochs)]

    out_shape = (n_pairs, 2 * n_rows - 1, 2 * n_cols - 1)
    s2       = np.full(out_shape, np.nan, dtype=np.float32)
    n_counts = np.zeros(out_shape, dtype=np.int32)

    for k, (i, j) in enumerate(epoch_pairs):
        dsq, N = _sum_diff_sq_full(
            M_fft[i], MF_fft[i], MF2_fft[i],
            M_fft[j], MF_fft[j], MF2_fft[j],
            fft_shape, n_rows, n_cols,
        )
        valid = N >= 0.5
        s2[k][valid]  = dsq[valid] / N[valid]
        n_counts[k]   = N

    s2       = np.fft.fftshift(s2,       axes=(1, 2))
    n_counts = np.fft.fftshift(n_counts, axes=(1, 2))
    lag_du   = np.fft.fftshift(lag_du)
    lag_dv   = np.fft.fftshift(lag_dv)

    size_px = int(data.get('size', -1))
    du_pix  = (float(np.median(np.abs(np.diff(U_grid[0]))))
               if U_grid.shape[1] > 1 else np.nan)
    size_ly = size_px * du_pix if size_px > 0 else np.nan

    return {
        "s2":          s2,
        "n_counts":    n_counts,
        "lag_du":      lag_du,
        "lag_dv":      lag_dv,
        "lag_dw":      lag_dw,
        "epoch_pairs": epoch_pairs,
        "u_mean":       float(np.mean(U_grid)),
        "v_mean":       float(np.mean(V_grid)),
        "w_mean":       float(np.mean(W)),
        "w_values":     W.copy(),
        "flux_mad_std": flux_mad_std,
        "row":          int(data.get('row', -1)),
        "col":          int(data.get('col', -1)),
        "size":         size_px,
        "size_ly":      size_ly,
    }

# old code
def get_sf_2d(image):
    if np.isnan(image).any():
        image = image.copy()
        image[np.isnan(image)] = np.nanmean(image)
    
    ny, nx = image.shape
    var = np.var(image)
    mean_val = np.mean(image)
    img_centered = image - mean_val
    
    pad_y, pad_x = ny, nx
    padded = np.pad(img_centered, ((0, pad_y), (0, pad_x)), mode='constant')
    
    fft_img = fft2(padded)
    psd = np.abs(fft_img)**2
    full_ac = fftshift(np.real(ifft2(psd)))
    
    ones_pad = np.pad(np.ones_like(img_centered), ((0, pad_y), (0, pad_x)), constant_values=0)
    fft_ones = fft2(ones_pad)
    overlap_count = fftshift(np.real(ifft2(np.abs(fft_ones)**2)))
    overlap_count[overlap_count < 1e-5] = 1
    normalized_ac = full_ac / overlap_count
    
    cy, cx = normalized_ac.shape[0] // 2, normalized_ac.shape[1] // 2
    start_y = cy - (ny // 2)
    start_x = cx - (nx // 2)
    ac_crop = normalized_ac[start_y : start_y + ny, start_x : start_x + nx]
    
    if ac_crop.shape != (ny, nx):
        ac_crop = normalized_ac[start_y : start_y + ny, start_x : start_x + nx]

    s2_map = 2 * (var - ac_crop)
    my, mx = s2_map.shape[0] // 2, s2_map.shape[1] // 2
    s2_map[my, mx] = 0
    return s2_map


# ---------------------------------------------------------------------------
# Structure function fitting — 1D profile functions
# ---------------------------------------------------------------------------
# Each profile is a callable: profile(r, params_array) -> log10(S2)
# with attached metadata: .n_params, .param_names, .default_guess
# (None entries in default_guess are replaced by a data-driven amplitude
# estimate inside fit_s2).
#
# The full parameter vector used by the optimizer is always:
#   [s11, s22, s33, l12, l13, l23,  <-- 6 geometric (L-matrix) params
#    *profile_params]                <-- profile.n_params params

def weibull_log_s2(r, params):
    """S2 = var_inf * (1 - exp(-r^beta))^(alpha/beta)"""
    alpha, beta, var_inf = params
    weibull = np.maximum(-np.expm1(-r ** beta), 1e-300)
    return np.log10(var_inf) + (alpha / beta) * np.log10(weibull)

weibull_log_s2.n_params      = 3
weibull_log_s2.param_names   = ['alpha', 'beta', 'var_inf']
weibull_log_s2.default_guess = [0.4, 2.0, None]   # None → data-driven
weibull_log_s2.param_bounds  = [(1e-3, np.inf), (1.0, 10.0), (1e-6, 4.0)]


def broken_pl_log_s2(r, params):
    """S2 = A * (r/r_b)^alpha1 / (1 + (r/r_b)^(alpha1-alpha2))

    Steep power law alpha1 at small r, flat power law alpha2 at large r,
    transitioning around r_b.  S2(r_b) = A/2.
    """
    alpha1, alpha2, r_b, A = params
    log_r_over_rb = np.log(np.maximum(r, 1e-100) / r_b)
    # log(1 + (r/r_b)^(alpha1-alpha2)) computed stably via logaddexp
    log_denom = np.logaddexp(0.0, (alpha1 - alpha2) * log_r_over_rb)
    return (np.log10(A) + alpha1 * log_r_over_rb / np.log(10)
            - log_denom / np.log(10))

broken_pl_log_s2.n_params      = 4
broken_pl_log_s2.param_names   = ['alpha1', 'alpha2', 'r_b', 'A']
broken_pl_log_s2.default_guess = [1.0, 0.4, 0.1, 1.0]
broken_pl_log_s2.param_bounds  = [(1e-3, np.inf), (1e-3, np.inf),
                                   (1e-4, np.inf), (1e-6, np.inf)]


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

_GEOM_KEYS = ['s11', 's22', 's33', 'l12', 'l13', 'l23']
_N_GEOM    = 6


def _compute_r(geom_params, lags):
    """Ellipsoidal radius r = |L⁻¹ lag| for each row of lags (N, 3)."""
    s11, s22, s33, l12, l13, l23 = geom_params
    # Manual back-substitution for the fixed 3×3 upper-triangular L avoids
    # scipy overhead (validation, dispatch, allocation) on every residual call.
    xw = lags[:, 2] / s33
    xv = (lags[:, 1] - l23 * xw) / s22
    xu = (lags[:, 0] - l12 * xv - l13 * xw) / s11
    return np.maximum(np.sqrt(xu*xu + xv*xv + xw*xw), 1e-100)


def log_s2_model(params, lags, profile=None):
    """
    Evaluate log10 S2 on lag vectors using a pluggable 1D profile.

    Parameters
    ----------
    params  : array-like [s11, s22, s33, l12, l13, l23, *profile_params]
              or a params dict as returned by fit_s2(...)['params']
    lags    : (N, 3) array of [dU, dV, dW] lag vectors [ly]
    profile : 1D profile callable (default: weibull_log_s2)

    Returns
    -------
    (N,) array of log10 S2 values
    """
    if profile is None:
        profile = weibull_log_s2
    if isinstance(params, dict):
        geom   = [params[k] for k in _GEOM_KEYS]
        prof   = [params[k] for k in profile.param_names]
    else:
        geom   = params[:_N_GEOM]
        prof   = params[_N_GEOM:]
    r = _compute_r(geom, lags)
    return profile(r, prof)


def predict_s2(params, lag_du, lag_dv, lag_dw, profile=None):
    """
    Evaluate the S2 model on a full (n_pairs, n_lag_v, n_lag_u) grid.

    Parameters
    ----------
    params  : array-like or dict of model parameters (see log_s2_model)
    lag_du  : (n_lag_u,) U lag coordinates [ly]
    lag_dv  : (n_lag_v,) V lag coordinates [ly]
    lag_dw  : (n_pairs,) W lag per plane [ly]
    profile : 1D profile callable (default: weibull_log_s2)

    Returns
    -------
    (n_pairs, n_lag_v, n_lag_u) array of predicted S2 values
    """
    if profile is None:
        profile = weibull_log_s2
    DV, DU = np.meshgrid(lag_dv, lag_du, indexing='ij')
    du_flat = DU.ravel()
    dv_flat = DV.ravel()
    n_lag_v, n_lag_u = DV.shape
    n_pairs = len(lag_dw)
    s2_pred = np.empty((n_pairs, n_lag_v, n_lag_u), dtype=np.float32)
    for k, dw in enumerate(lag_dw):
        lags = np.column_stack([du_flat, dv_flat, np.full(du_flat.size, dw)])
        s2_pred[k] = (10 ** log_s2_model(params, lags, profile=profile)
                      ).reshape(n_lag_v, n_lag_u)
    return s2_pred


def params_from_principal_axes(a1, a2, a3, theta, phi, psi):
    """
    Build geometry params dict from principal-axis ellipsoid description.

    The ellipsoid covariance C = L L^T has semi-axes a1 >= a2 >= a3 (in
    light-years) oriented by the rotation matrix Q whose columns are the
    principal-axis unit vectors.

    Parameters
    ----------
    a1, a2, a3 : float
        Semi-axis lengths [ly] along the three principal axes.
        a1 is the longest axis.
    theta : float [rad]
        Polar angle of a1 from the W axis (theta=0: along W,
        theta=pi/2: in UV plane).
    phi : float [rad]
        Azimuthal angle of a1 in the UV plane, measured from U toward V.
    psi : float [rad]
        Roll of a2/a3 around a1.  psi=0: a2 lies in the plane spanned by
        a1 and W (or, if a1 || W, a2 is along U).

    Returns
    -------
    dict with keys s11, s22, s33, l12, l13, l23
    """
    # Longest-axis unit vector  (U, V, W components)
    n1 = np.array([np.sin(theta) * np.cos(phi),
                   np.sin(theta) * np.sin(phi),
                   np.cos(theta)])

    # Two perpendicular vectors before psi roll
    n2_0 = np.array([np.cos(theta) * np.cos(phi),
                     np.cos(theta) * np.sin(phi),
                     -np.sin(theta)])
    n3_0 = np.array([-np.sin(phi),
                     np.cos(phi),
                     0.0])

    # Apply psi roll around n1
    n2 = np.cos(psi) * n2_0 + np.sin(psi) * n3_0
    n3 = -np.sin(psi) * n2_0 + np.cos(psi) * n3_0

    # Q columns are principal axes; C = Q diag(a^2) Q^T
    Q = np.column_stack([n1, n2, n3])
    C = Q @ np.diag([a1**2, a2**2, a3**2]) @ Q.T

    # Upper-triangular Cholesky: C = L L^T  (L upper triangular)
    s33 = np.sqrt(C[2, 2])
    l23 = C[1, 2] / s33
    l13 = C[0, 2] / s33
    s22 = np.sqrt(C[1, 1] - l23**2)
    l12 = (C[0, 1] - l23 * l13) / s22
    s11 = np.sqrt(C[0, 0] - l12**2 - l13**2)

    return dict(s11=s11, s22=s22, s33=s33, l12=l12, l13=l13, l23=l23)


def params_from_uvshift(du_per_dw, dv_per_dw, scale, elongation,
                        psi=0.0, axis_ratio=1.0):
    """
    Build geometry params from eyeball-observable quantities.

    Parameters
    ----------
    du_per_dw : float
        Arrow U-component: dU shift of S2 minimum per unit dW.
        Equals l13/s33 in the L-matrix parameterization.
    dv_per_dw : float
        Arrow V-component: dV shift of S2 minimum per unit dW.
        Equals l23/s33 in the L-matrix parameterization.
    scale : float
        Overall size [ly]: the minor semi-axis length a2 (= a3 for prolate).
    elongation : float
        Axis ratio a1/a2 (>= 1).  elongation=1 gives a sphere.
    psi : float [rad]
        Roll of a2/a3 around a1 (default 0, relevant only when
        axis_ratio != 1).
    axis_ratio : float
        a3/a2 ratio (default 1 = prolate; < 1 gives a triaxial ellipsoid).

    Returns
    -------
    dict with keys s11, s22, s33, l12, l13, l23

    Notes
    -----
    phi is taken directly from the arrow direction.  theta is solved
    numerically from the prolate-ellipsoid formula relating the arrow
    magnitude to elongation and theta:

        |arrow|^2 = sin^2(theta) cos^2(theta) (e^2 - 1)^2
                    / (e^2 cos^2(theta) + sin^2(theta))^2

    where e = a1/a2 = elongation.  In the degenerate case elongation=1
    (sphere) or |arrow|=0, theta defaults to pi/2.
    """
    phi = np.arctan2(dv_per_dw, du_per_dw)
    arrow_mag = np.sqrt(du_per_dw**2 + dv_per_dw**2)

    e = elongation

    if e <= 1.0 or arrow_mag == 0.0:
        theta = np.pi / 2.0
    else:
        # Solve for theta in [0, pi/2]:
        #   f(theta) = sin(theta)*cos(theta)*(e^2-1) / (e^2*cos^2+sin^2) - arrow_mag
        # which is equivalent to the arrow-magnitude formula.
        from scipy.optimize import brentq

        def f(th):
            s, c = np.sin(th), np.cos(th)
            denom = e**2 * c**2 + s**2
            return s * c * (e**2 - 1.0) / denom - arrow_mag

        # f(0)=0, f(pi/2)=0, maximum somewhere in between.
        # The function peaks at some theta_peak; if arrow_mag exceeds the
        # peak there is no solution — clamp to theta_peak.
        th_grid = np.linspace(1e-6, np.pi / 2 - 1e-6, 1000)
        f_vals = np.vectorize(f)(th_grid)
        peak_idx = np.argmax(f_vals)

        if f_vals[peak_idx] <= 0:
            # arrow_mag exceeds g_max — no exact solution, clamp to theta_peak.
            theta = th_grid[peak_idx]
        else:
            # Two solutions exist; pick the one in (0, theta_peak).
            theta = brentq(f, 1e-6, th_grid[peak_idx])

    a1 = scale * elongation
    a2 = scale
    a3 = scale * axis_ratio

    return params_from_principal_axes(a1, a2, a3, theta, phi, psi)


def estimate_uvshift(sf, inner_uv_pixels=200):
    """
    Estimate du_per_dw and dv_per_dw from the S2 minimum in cross-epoch slices.

    The minimum of S2 at dW lies at (l13/s33*dW, l23/s33*dW), so dividing by
    dW gives the shift rates directly.  Sub-pixel accuracy is obtained by
    parabolic interpolation around the argmin in each direction independently.

    Pairs are tried in order of increasing |dW| (excluding same-epoch pairs).
    The first pair whose minimum is not at the edge of the search window is
    used.  Raises ValueError if no usable pair is found.

    Parameters
    ----------
    sf : dict
        Output of compute_s2.
    inner_uv_pixels : int
        Half-width of the central search region in pixels (default 200).

    Returns
    -------
    du_per_dw, dv_per_dw : float
    """
    lag_du = sf['lag_du']
    lag_dv = sf['lag_dv']
    p  = inner_uv_pixels
    cu = len(lag_du) // 2
    cv = len(lag_dv) // 2
    u_sl = slice(max(cu - p, 0), min(cu + p + 1, len(lag_du)))
    v_sl = slice(max(cv - p, 0), min(cv + p + 1, len(lag_dv)))

    def _parabolic_refine(arr, idx):
        if idx == 0 or idx == len(arr) - 1:
            return float(idx)
        a, b, c = arr[idx - 1], arr[idx], arr[idx + 1]
        if not (np.isfinite(a) and np.isfinite(c)):
            return float(idx)
        denom = 2.0 * (a - 2.0 * b + c)
        return idx - (c - a) / denom if denom != 0.0 else float(idx)

    # Sort candidate pairs by |dW|, shortest non-zero spacing first
    dws = sf['lag_dw']
    order = sorted(
        (k for k, dw in enumerate(dws) if dw != 0.0),
        key=lambda k: abs(dws[k]),
    )
    if not order:
        raise ValueError("No cross-epoch pairs with non-zero dW")

    last_exc = None
    for k in order:
        dw = dws[k]
        s2 = sf['s2'][k][v_sl, u_sl].copy()
        s2[~np.isfinite(s2)] = np.inf
        iv_crop, iu_crop = np.unravel_index(np.argmin(s2), s2.shape)
        if (iu_crop == 0 or iu_crop == s2.shape[1] - 1 or
                iv_crop == 0 or iv_crop == s2.shape[0] - 1):
            last_exc = ValueError(
                f"S2 minimum at edge of search window for pair "
                f"{sf['epoch_pairs'][k]} (dW={dw:.3f})")
            continue
        iu_fine = _parabolic_refine(s2[iv_crop, :], iu_crop)
        iv_fine = _parabolic_refine(s2[:, iu_crop], iv_crop)
        du_min = lag_du[u_sl][0] + iu_fine * (lag_du[1] - lag_du[0])
        dv_min = lag_dv[v_sl][0] + iv_fine * (lag_dv[1] - lag_dv[0])
        return du_min / dw, dv_min / dw

    raise last_exc


def principal_axes_from_params(params):
    """
    Decompose geometry params into principal-axis ellipsoid description.

    Inverse of params_from_principal_axes.

    Parameters
    ----------
    params : dict with keys s11, s22, s33, l12, l13, l23
             or array-like [s11, s22, s33, l12, l13, l23, ...]

    Returns
    -------
    dict with keys:
      a1, a2, a3 : float  -- semi-axis lengths [ly], sorted descending
      theta      : float  -- polar angle of a1 from W axis [rad]
      phi        : float  -- azimuthal angle of a1 from U in UV plane [rad]
      psi        : float  -- roll of a2/a3 around a1 [rad]
    """
    if not isinstance(params, dict):
        s11, s22, s33, l12, l13, l23 = params[:6]
    else:
        s11 = params['s11']; s22 = params['s22']; s33 = params['s33']
        l12 = params['l12']; l13 = params['l13']; l23 = params['l23']

    L = np.array([[s11, l12, l13],
                  [0.0, s22, l23],
                  [0.0, 0.0, s33]])
    C = L @ L.T

    # Eigendecompose C = Q diag(a^2) Q^T, eigenvalues descending
    eigvals, Q = np.linalg.eigh(C)          # eigh returns ascending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    Q = Q[:, idx]

    # Ensure right-handed frame (det Q = +1)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    a1, a2, a3 = np.sqrt(np.maximum(eigvals, 0.0))

    # Longest-axis unit vector n1 = Q[:,0]  (U, V, W components)
    n1 = Q[:, 0]
    theta = np.arccos(np.clip(n1[2], -1.0, 1.0))
    phi = np.arctan2(n1[1], n1[0])

    # Reference perpendicular frame at (theta, phi)
    n2_0 = np.array([np.cos(theta) * np.cos(phi),
                     np.cos(theta) * np.sin(phi),
                     -np.sin(theta)])
    n3_0 = np.array([-np.sin(phi), np.cos(phi), 0.0])

    # psi: angle of Q[:,1] relative to n2_0 in the n2_0/n3_0 plane
    n2 = Q[:, 1]
    psi = np.arctan2(n2 @ n3_0, n2 @ n2_0)

    return dict(a1=a1, a2=a2, a3=a3, theta=theta, phi=phi, psi=psi)


def _make_fit_data(sf, inner_uv_pixels, min_same_epoch_lag_pix, s2_floor,
                   min_n_fraction=0.1, fit_stride=1):
    """Select lag points for fitting; shared by fit_s2 and build_fit_result.

    min_n_fraction : exclude lags where N < min_n_fraction * max(N) for that
        pair.  Removes severely under-sampled lags caused by masked image
        regions without requiring a hand-tuned absolute threshold.
    fit_stride     : take every fit_stride-th lag pixel in U and V.  Values
        > 1 reduce N by ~fit_stride² and speed up fitting proportionally with
        negligible effect on fit quality (S2 is smooth).
    """
    s2_arr      = sf['s2']
    n_counts    = sf['n_counts']
    lag_du      = sf['lag_du']
    lag_dv      = sf['lag_dv']
    lag_dw      = sf['lag_dw']
    epoch_pairs = sf['epoch_pairs']

    n_pairs, n_lag_v, n_lag_u = s2_arr.shape
    p    = inner_uv_pixels
    cv   = n_lag_v // 2
    cu   = n_lag_u // 2
    v_sl = slice(cv - p, cv + p)
    u_sl = slice(cu - p, cu + p)

    DV, DU = np.meshgrid(lag_dv[v_sl], lag_du[u_sl], indexing='ij')
    row_off = np.abs(np.arange(2 * p) - p)
    col_off = np.abs(np.arange(2 * p) - p)
    ROW_OFF, COL_OFF = np.meshgrid(row_off, col_off, indexing='ij')
    near_zero = ((ROW_OFF <= min_same_epoch_lag_pix) &
                 (COL_OFF <= min_same_epoch_lag_pix))

    # Stride mask: every fit_stride-th pixel in both directions
    if fit_stride > 1:
        stride_mask = np.zeros((2 * p, 2 * p), dtype=bool)
        stride_mask[::fit_stride, ::fit_stride] = True
    else:
        stride_mask = None

    dU_list, dV_list, dW_list, log_s2_list = [], [], [], []
    fit_mask     = np.zeros((n_pairs, n_lag_v, n_lag_u), dtype=bool)
    display_mask = np.zeros((n_pairs, n_lag_v, n_lag_u), dtype=bool)

    for k, (i, j) in enumerate(epoch_pairs):
        plane = s2_arr[k, v_sl, u_sl]
        n_win = n_counts[k, v_sl, u_sl]
        n_max = n_counts[k].max()
        mask  = np.isfinite(plane) & (n_win >= min_n_fraction * n_max)
        if i == j:
            mask &= ~near_zero
        display_mask[k, v_sl, u_sl] = mask
        if stride_mask is not None:
            mask &= stride_mask
        fit_mask[k, v_sl, u_sl] = mask
        log_s2 = np.log10(np.maximum(plane, s2_floor))
        sel = mask.ravel()
        dU_list.append(DU.ravel()[sel])
        dV_list.append(DV.ravel()[sel])
        dW_list.append(np.full(sel.sum(), lag_dw[k]))
        log_s2_list.append(log_s2.ravel()[sel])

    lags_flat  = np.column_stack([np.concatenate(dU_list),
                                  np.concatenate(dV_list),
                                  np.concatenate(dW_list)])
    log_s2_obs = np.concatenate(log_s2_list)
    return fit_mask, display_mask, lags_flat, log_s2_obs


def _point_weights(lags_flat, sf, weighting):
    """Per-point weights w_i; objective = sum(w_i*(obs-pred)^2/sigma^2).

    '1/r'  — constant weight per radial bin (2π r dr × 1/r = const)
    '1/r2' — constant weight per log radial bin (2π r dr × 1/r² = d ln r)
    None   — uniform
    """
    if weighting is None:
        return np.ones(len(lags_flat))
    if weighting in ('1/r', '1/r2'):
        r_uv = np.hypot(lags_flat[:, 0], lags_flat[:, 1])
        lag_step = float(abs(sf['lag_du'][1] - sf['lag_du'][0]))
        r_clamp = np.maximum(r_uv, lag_step)
        if weighting == '1/r':
            return 1.0 / r_clamp
        return 1.0 / r_clamp ** 2
    raise ValueError(f"Unknown weighting: {weighting!r}")


def extract_params(record, profile=None):
    """Extract fit-parameter dict from a summarize_chunks() structured array row."""
    if profile is None:
        profile = weibull_log_s2
    keys = list(_GEOM_KEYS) + list(profile.param_names)
    return {k: float(record[k]) for k in keys}


def build_fit_result(sf, params, profile=None,
                     inner_uv_pixels: int = 200,
                     min_same_epoch_lag_pix: int = 4,
                     s2_floor: float = 10**-3.75,
                     noise_scale_dex: float = 0.1,
                     min_n_fraction: float = 0.1,
                     fit_stride: int = 1,
                     weighting='1/r') -> dict:
    """
    Build the same dict as fit_s2 returns, but from a given params array or
    dict rather than from optimisation.  Useful for manually exploring
    parameter space and generating plots.

    Parameters
    ----------
    sf      : dict returned by compute_s2()
    params  : array-like [s11,s22,s33,l12,l13,l23,*profile_params]
              or a dict with those keys
    profile : 1D profile callable (default: weibull_log_s2)
    inner_uv_pixels, min_same_epoch_lag_pix, s2_floor, noise_scale_dex :
              same meaning as in fit_s2; must match if comparing to an
              existing fit_s2 result
    """
    from types import SimpleNamespace
    if profile is None:
        profile = weibull_log_s2

    if isinstance(params, dict):
        params_arr = np.array([params[k] for k in _GEOM_KEYS]
                              + [params[k] for k in profile.param_names])
    else:
        params_arr = np.asarray(params, dtype=float)

    params_dict = {**dict(zip(_GEOM_KEYS, params_arr[:_N_GEOM])),
                   **dict(zip(profile.param_names, params_arr[_N_GEOM:]))}

    fit_mask, display_mask, lags_flat, log_s2_obs = _make_fit_data(
        sf, inner_uv_pixels, min_same_epoch_lag_pix, s2_floor, min_n_fraction,
        fit_stride)

    weights_flat = _point_weights(lags_flat, sf, weighting)
    residuals    = (np.sqrt(weights_flat) * (log_s2_obs - log_s2_model(params_arr, lags_flat, profile=profile))
                    / noise_scale_dex)
    mock_fit     = SimpleNamespace(x=params_arr, fun=residuals,
                                   nfev=0, success=True)

    # Build 3D fit_weight: 0 where excluded, w_i where included.
    lag_du, lag_dv = sf['lag_du'], sf['lag_dv']
    DV_full, DU_full = np.meshgrid(lag_dv, lag_du, indexing='ij')
    if weighting in ('1/r', '1/r2'):
        lag_step = float(abs(lag_du[1] - lag_du[0]))
        r_clamp = np.maximum(np.hypot(DU_full, DV_full), lag_step)
        w_2d = 1.0 / r_clamp if weighting == '1/r' else 1.0 / r_clamp ** 2
    else:
        w_2d = np.ones_like(DU_full)
    fit_weight = (fit_mask * w_2d).astype(np.float32)

    return {
        'fit':          mock_fit,
        'params':       params_dict,
        'profile':      profile,
        'weighting':    weighting,
        's2_pred':      predict_s2(params_arr, sf['lag_du'], sf['lag_dv'],
                                   sf['lag_dw'], profile=profile),
        'fit_weight':   fit_weight,
        'display_mask': display_mask,
        'log_s2_obs':   log_s2_obs,
    }


def fit_s2(result: dict,
           profile=None,
           guess: dict | None = None,
           inner_uv_pixels: int = 200,
           min_same_epoch_lag_pix: int = 4,
           s2_floor: float = 10**-3.75,
           noise_scale_dex: float = 0.1,
           min_n_fraction: float = 0.1,
           fit_stride: int = 1,
           max_nfev: int | None = None,
           weighting='1/r') -> dict:
    """
    Fit S2(dU, dV, dW) to the structure function using a pluggable 1D profile.

    The full parameter vector is [s11, s22, s33, l12, l13, l23, *profile_params].
    Fitting is done in log10 space (L2 loss) with residuals normalised by
    noise_scale_dex.

    Parameters
    ----------
    result              : dict returned by compute_s2()
    profile             : 1D profile callable (default: weibull_log_s2).
                          Must have .n_params, .param_names, .default_guess.
    guess               : optional dict overriding any geometric or profile
                          parameter defaults
    inner_uv_pixels     : half-width in pixels of the UV lag region to include
    min_same_epoch_lag_pix : exclude central ±N pixels for dW=0 planes
    s2_floor            : S2 values below this are clipped before taking log
    noise_scale_dex     : residuals divided by this (inlier/outlier boundary)
    max_nfev            : maximum function evaluations (None = unlimited)
    """
    if profile is None:
        profile = weibull_log_s2

    fit_mask, display_mask, lags_flat, log_s2_obs = _make_fit_data(
        result, inner_uv_pixels, min_same_epoch_lag_pix, s2_floor,
        min_n_fraction, fit_stride)

    inner_s2  = result['s2'][:, result['lag_dv'].size//2 - inner_uv_pixels
                                :result['lag_dv'].size//2 + inner_uv_pixels,
                               result['lag_du'].size//2 - inner_uv_pixels
                                :result['lag_du'].size//2 + inner_uv_pixels]
    amp_guess = float(np.nanmedian(inner_s2[np.isfinite(inner_s2)]))

    g = guess or {}
    try:
        du, dv = estimate_uvshift(result)
    except Exception:
        du, dv = 0.0, 0.0
    # Try several scales and pick the one with the lowest chi² at the initial
    # guess (single evaluation, no optimisation) to avoid basin-of-attraction
    # sensitivity to the scale parameter.
    _scales = (0.05, 0.1, 0.2, 0.3)
    _prof_defaults = [amp_guess if d is None else d
                      for d in profile.default_guess]
    best_scale, best_chi2 = _scales[0], np.inf
    for _s in _scales:
        _geom = params_from_uvshift(du, dv, scale=_s, elongation=5.0)
        _p0 = np.array([_geom[k] for k in _GEOM_KEYS] + _prof_defaults)
        _res = (log_s2_obs - log_s2_model(_p0, lags_flat, profile=profile))
        _chi2 = float(np.mean(_res ** 2))
        if _chi2 < best_chi2:
            best_chi2, best_scale = _chi2, _s
    auto_geom = params_from_uvshift(du, dv, scale=best_scale, elongation=5.0)
    geom_p0 = [g.get(k, auto_geom[k]) for k in _GEOM_KEYS]
    prof_bounds = getattr(profile, 'param_bounds',
                          [(-np.inf, np.inf)] * profile.n_params)
    prof_p0  = [np.clip(g.get(name, (amp_guess if default is None else default)),
                        b[0], b[1])
                for name, default, b in zip(profile.param_names,
                                            profile.default_guess,
                                            prof_bounds)]
    p0 = np.array(geom_p0 + prof_p0)

    sqrt_w = np.sqrt(_point_weights(lags_flat, result, weighting))

    # Optimise with log(s11), log(s22), log(s33) to enforce positivity and
    # prevent collapse to the degenerate s→0 minimum.  Profile params with
    # positive bounds are also log-transformed.
    _N_LOG = 3   # first 3 geom params (s11, s22, s33)
    _prof_log = [b[0] > 0 for b in prof_bounds]   # which prof params to log

    def _to_opt(p):
        q = p.copy()
        q[:_N_LOG] = np.log(np.maximum(p[:_N_LOG], 1e-30))
        for i, do_log in enumerate(_prof_log):
            if do_log:
                q[_N_GEOM + i] = np.log(max(p[_N_GEOM + i], 1e-30))
        return q

    def _from_opt(q):
        p = q.copy()
        p[:_N_LOG] = np.exp(q[:_N_LOG])
        for i, do_log in enumerate(_prof_log):
            if do_log:
                p[_N_GEOM + i] = np.exp(q[_N_GEOM + i])
        return p

    # Build bounds in the transformed (q) space.
    # s11/s22/s33 are log-transformed; bound them to [0, 10] ly.
    # l12/l13/l23 are unconstrained.
    _S_MAX = 10.0
    geom_lo = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    geom_hi = [np.log(_S_MAX), np.log(_S_MAX), np.log(_S_MAX),
               np.inf, np.inf, np.inf]
    prof_lo = [np.log(b[0]) if do_log else b[0]
               for b, do_log in zip(prof_bounds, _prof_log)]
    prof_hi = [np.log(b[1]) if (do_log and np.isfinite(b[1])) else b[1]
               for b, do_log in zip(prof_bounds, _prof_log)]
    q_lo = np.array(geom_lo + prof_lo)
    q_hi = np.array(geom_hi + prof_hi)

    q0 = _to_opt(p0)

    def _residuals(q):
        return sqrt_w * (log_s2_obs - log_s2_model(_from_opt(q), lags_flat,
                                                    profile=profile)) / noise_scale_dex

    # Stage 1: pre-fit profile params with geometry fixed so the profile is
    # already reasonable before geometry is allowed to move.
    geom_q0 = q0[:_N_GEOM]
    prof_q0  = q0[_N_GEOM:]

    def _residuals_profile(prof_q):
        return _residuals(np.concatenate([geom_q0, prof_q]))

    pre = least_squares(_residuals_profile, prof_q0, loss='linear',
                        max_nfev=200,
                        bounds=(q_lo[_N_GEOM:], q_hi[_N_GEOM:]))
    q0 = np.concatenate([geom_q0, pre.x])

    # Stage 2: full optimisation from the pre-warmed starting point.
    fit = least_squares(_residuals, q0, loss='linear',
                        max_nfev=max_nfev, bounds=(q_lo, q_hi))

    fit_params = _from_opt(fit.x)

    out = build_fit_result(result, fit_params, profile=profile,
                           inner_uv_pixels=inner_uv_pixels,
                           min_same_epoch_lag_pix=min_same_epoch_lag_pix,
                           s2_floor=s2_floor,
                           noise_scale_dex=noise_scale_dex,
                           min_n_fraction=min_n_fraction,
                           fit_stride=fit_stride,
                           weighting=weighting)
    # Drop the least_squares Jacobian (~half the per-window memory: m residuals
    # x n params).  It is never read or saved — params come from build_fit_result
    # and uncertainties from jackknife, not the Jacobian covariance.
    fit.jac = None
    out['fit'] = fit
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_s2_1d(sf, fit=None, ellipsoidal=False, ax=None, **scatter_kwargs):
    """
    Plot S2 vs lag magnitude (or ellipsoidal radius) as a conditional density
    plot via util_efs.scatterplot.

    Data are shown in black/gray; model prediction (if fit given) overlaid in
    blue with nograyscale=True.

    Parameters
    ----------
    sf            : dict from compute_s2()
    fit           : optional dict from fit_s2(); if given, overlays prediction
                    and uses fit_mask to select points
    ellipsoidal   : if True, x-axis is |L⁻¹ lag| (ellipsoidal radius) rather
                    than Euclidean |lag| [ly]
    ax            : axes to draw into (default: current axes)
    **scatter_kwargs : forwarded to util_efs.scatterplot (e.g. xrange, yrange,
                       xnpix, ynpix, nograyscale, linecolor)
    """
    s2_arr = sf['s2']
    lag_du = sf['lag_du']
    lag_dv = sf['lag_dv']
    lag_dw = sf['lag_dw']

    DV, DU = np.meshgrid(lag_dv, lag_du, indexing='ij')
    mask = (fit['fit_weight'] > 0) if fit is not None else np.isfinite(s2_arr)

    if ellipsoidal and fit is not None:
        p = fit['params']
        L = np.array([[p['s11'], p['l12'], p['l13']],
                      [0.0,     p['s22'], p['l23']],
                      [0.0,     0.0,      p['s33']]])

    x_data, y_data, x_pred, y_pred = [], [], [], []
    for k, dw in enumerate(lag_dw):
        m      = mask[k]
        du_m   = DU[m]
        dv_m   = DV[m]

        if ellipsoidal and fit is not None:
            lags_m = np.column_stack(
                [du_m, dv_m, np.full(du_m.size, dw)])
            r  = scipy.linalg.solve_triangular(L, lags_m.T)
            xk = np.log10(np.maximum(np.sqrt((r * r).sum(axis=0)), 1e-3))
        else:
            xk = np.log10(
                np.maximum(np.sqrt(du_m**2 + dv_m**2 + dw**2), 1e-3))

        x_data.append(xk)
        y_data.append(np.log10(np.maximum(s2_arr[k][m], 1e-4)))
        if fit is not None:
            x_pred.append(xk)
            y_pred.append(np.log10(np.maximum(fit['s2_pred'][k][m], 1e-4)))

    x_data = np.concatenate(x_data)
    y_data = np.concatenate(y_data)

    if ax is not None:
        plt.sca(ax)

    kw = scatter_kwargs.copy()

    util_efs.scatterplot(x_data, y_data, nograyscale=True, **kw)
    xrange = plt.xlim()
    yrange = plt.ylim()
    kw.pop('xrange', None)
    kw.pop('yrange', None)

    if fit is not None:
        util_efs.scatterplot(np.concatenate(x_pred), np.concatenate(y_pred),
                       xrange=xrange, yrange=yrange,
                       nograyscale=True, linecolor='blue', **kw)

    cur = plt.gca()
    cur.set_xlabel(
        'log$_{10}$ ellipsoidal radius' if ellipsoidal
        else 'log$_{10}$ |lag| [ly]', fontsize=8)
    cur.set_ylabel('log$_{10}$ S$_2$', fontsize=8)
    return xrange, yrange


def _make_thumbnail_axes(sf, fit, subplot_spec,
                          uv_range=0.2, vmin_sf=1e-4, vmax_sf=None,
                          vdiff=0.3, weighted_diff=False):
    """
    Populate a region (given by subplot_spec or None for a new figure) with
    the 5x9 thumbnail grid.  Returns the figure and the axes array.
    """
    s2_arr   = sf['s2']
    lag_du   = sf['lag_du']
    lag_dv   = sf['lag_dv']
    pairs    = sf['epoch_pairs']
    s2_pred      = fit['s2_pred']
    fit_weight   = fit['fit_weight']
    fit_mask     = fit_weight > 0
    display_mask = fit.get('display_mask', fit_mask)
    n_pairs    = len(pairs)

    pairs_per_row = 3
    n_rows = (n_pairs + pairs_per_row - 1) // pairs_per_row
    n_cols = pairs_per_row * 3   # 9

    if vmax_sf is None:
        p = fit['params']
        vmax_sf = p.get('var_inf', p.get('A', 5e-3))

    fig = plt.gcf()
    if subplot_spec is None:
        inner_gs = gridspec.GridSpec(n_rows, n_cols, hspace=0, wspace=0)
    else:
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols, subplot_spec=subplot_spec, hspace=0, wspace=0)

    axes = np.array([[fig.add_subplot(inner_gs[r, c])
                      for c in range(n_cols)]
                     for r in range(n_rows)])

    log_diff = np.full_like(s2_arr, np.nan)
    for k in range(n_pairs):
        m = display_mask[k]
        raw = np.where(
            m,
            np.log10(np.maximum(s2_arr[k],  vmin_sf))
            - np.log10(np.maximum(s2_pred[k], vmin_sf)),
            np.nan)
        if weighted_diff:
            log_diff[k] = np.where(m, np.sqrt(fit_weight[k]) * raw, np.nan)
        else:
            log_diff[k] = raw
    type_data = [('S2', s2_arr, False), ('pred', s2_pred, False),
                 ('diff', log_diff, True)]

    du_step = lag_du[1] - lag_du[0]
    dv_step = lag_dv[1] - lag_dv[0]
    extent = [lag_du[0] - du_step/2, lag_du[-1] + du_step/2,
              lag_dv[0] - dv_step/2, lag_dv[-1] + dv_step/2]

    cmap = plt.get_cmap('binary')
    norm_log  = matplotlib.colors.LogNorm(vmin=vmin_sf, vmax=vmax_sf)
    norm_lin  = matplotlib.colors.Normalize(vmin=-vdiff, vmax=vdiff)
    _OV_ALPHA = 0.35  # red overlay alpha

    for k in range(n_pairs):
        row   = k // pairs_per_row
        group = k % pairs_per_row
        excl  = ~display_mask[k]   # (n_lag_v, n_lag_u)

        for ti, (_, images, is_diff) in enumerate(type_data):
            col = group * 3 + ti
            ax  = axes[row, col]

            # Apply colormap in numpy, then alpha-blend red exclusion mask —
            # single imshow call per panel instead of two.
            im = images[k]                             # (n_lag_v, n_lag_u)
            norm = norm_lin if is_diff else norm_log
            rgba = cmap(norm(np.where(np.isfinite(im), im, np.nan)))
            rgba = rgba.copy()
            rgba[excl, 0] = _OV_ALPHA + (1-_OV_ALPHA) * rgba[excl, 0]
            rgba[excl, 1] = (1-_OV_ALPHA) * rgba[excl, 1]
            rgba[excl, 2] = (1-_OV_ALPHA) * rgba[excl, 2]
            rgba[excl, 3] = 1.0

            ax.imshow(rgba, extent=extent, origin='lower',
                      aspect='auto', interpolation='nearest')
            ax.set_xlim(-uv_range, uv_range)
            ax.set_ylim(-uv_range, uv_range)
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused cells
    for k in range(n_pairs, n_rows * pairs_per_row):
        for ti in range(3):
            axes[k // pairs_per_row,
                 (k % pairs_per_row) * 3 + ti].set_visible(False)

    # Column headers and row labels
    col_labels = ['S2', 'pred', 'diff'] * pairs_per_row
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=6, pad=2)
    for row in range(n_rows):
        row_pairs = [str(pairs[row * pairs_per_row + g])
                     for g in range(pairs_per_row)
                     if row * pairs_per_row + g < n_pairs]
        axes[row, 0].set_ylabel(', '.join(row_pairs), fontsize=5, labelpad=3)

    return fig, axes


def _chunk_suptitle(fit, chunk_id):
    """Return the standard suptitle string for a chunk."""
    fitres     = fit['fit']
    resid      = fitres.fun
    sum_w      = float(fit['fit_weight'][fit['fit_weight'] > 0].sum())
    chi2dof    = float(np.sum(resid**2)) / sum_w
    n_pts      = len(fit['log_s2_obs'])
    ax = principal_axes_from_params(fit['params'])
    a1, a2, a3 = ax['a1'], ax['a2'], ax['a3']
    theta, phi = ax['theta'], ax['phi']
    return (
        f"chunk {chunk_id}   "
        f"$\\chi^2$/dof={chi2dof:.2f}   "
        f"N={n_pts/1e6:.2f}M   "
        f"$a_1$={a1:.3f} ly   "
        f"$a_2/a_1$={a2/a1:.2f}   "
        f"$a_3/a_1$={a3/a1:.2f}   "
        f"$\\theta$={np.degrees(theta):.1f}$^\\circ$   "
        f"$\\phi$={np.degrees(phi):.1f}$^\\circ$"
    )


def plot_s2_thumbnails(sf, fit, chunk_id=None, figsize=(18, 10),
                       uv_range=0.2, vmin_sf=1e-4, vmax_sf=None,
                       vdiff=0.3, weighted_diff=False):
    """
    Standalone thumbnail grid for one chunk: 5 rows x 9 cols, no spacing.

    Each row contains 3 epoch pairs; each pair occupies 3 consecutive columns
    (S2, pred, diff).  Column headers repeat "S2 / pred / diff" across the
    top; row labels on the left list the epoch pairs in each row.

    Parameters
    ----------
    sf       : dict from compute_s2()
    fit      : dict from fit_s2()
    chunk_id : label for the suptitle
    figsize  : matplotlib figsize
    uv_range : +/- half-width of each thumbnail [ly]
    vmin_sf, vmax_sf : color limits for S2 / prediction thumbnails (log scale)
    vdiff    : +/- color limit for residual thumbnails (linear scale)
    """
    fig = plt.figure(figsize=figsize)
    _make_thumbnail_axes(sf, fit, subplot_spec=None,
                         uv_range=uv_range, vmin_sf=vmin_sf,
                         vmax_sf=vmax_sf, vdiff=vdiff,
                         weighted_diff=weighted_diff)
    fig.suptitle(_chunk_suptitle(fit, chunk_id), fontsize=7)
    fig.subplots_adjust(top=0.93, left=0.08, right=0.99, bottom=0.01)
    return fig


def plot_full_page(sf, fit, data, chunk_id=None, figsize=(8.5, 11),
                   uv_range=0.35, vmin_sf=1e-4, vmax_sf=None, vdiff=0.3,
                   weighted_diff=False, newfig=False):
    """
    Full-page summary for one chunk.

    Layout (top to bottom):
      - Thumbnail grid (thumbnails take ~60% of page height)
      - Row of two 1-D SF plots (vs |lag| and vs ellipsoidal radius)
      - Row with the RGB epoch composite

    Parameters
    ----------
    sf, fit, data : outputs of compute_s2(), fit_s2(), read_chunk()
    chunk_id      : label for the suptitle
    figsize       : matplotlib figsize
    uv_range, vmin_sf, vmax_sf, vdiff : passed to thumbnail grid
    newfig        : make a new figure?
    """
    if newfig:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.gcf()

    # Outer: 3 rows (thumbnails / 1-D plots / RGB), 3 cols so RGB can be
    # centred by occupying the middle column only.
    outer = gridspec.GridSpec(3, 3, figure=fig,
                              height_ratios=[3, 1.2, 1.4],
                              width_ratios=[1, 2, 1],
                              hspace=0.3, wspace=0.3)

    # Thumbnails span the full width of the top row
    _make_thumbnail_axes(sf, fit, subplot_spec=outer[0, :],
                         uv_range=uv_range, vmin_sf=vmin_sf,
                         vmax_sf=vmax_sf, vdiff=vdiff,
                         weighted_diff=weighted_diff)

    # 1-D SF plots: split the full-width row into two equal halves
    inner_1d = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1, :], wspace=0.3)
    ax_lag = fig.add_subplot(inner_1d[0, 0])
    ax_ell = fig.add_subplot(inner_1d[0, 1])
    plot_s2_1d(sf, fit, ellipsoidal=False, ax=ax_lag)
    ax_lag.set_title('$S_2$ vs $|$lag$|$', fontsize=8)
    plot_s2_1d(sf, fit, ellipsoidal=True, ax=ax_ell)
    ax_ell.set_title('$S_2$ vs ellipsoidal radius', fontsize=8)

    # Bottom row: individual epoch images (smushed) + RGB composite
    flux    = data['flux_epochs']   # (n_epochs, n_v, n_u)
    W_vals  = data['W_values']
    n_ep    = flux.shape[0]
    U_grid  = data['U_grid']
    V_grid  = data['V_grid']
    du_im   = float(np.median(np.diff(U_grid[0])))
    dv_im   = float(np.median(np.diff(V_grid[:, 0])))
    ext_im  = [U_grid[0, 0] - du_im/2, U_grid[0, -1] + du_im/2,
               V_grid[0, 0] - dv_im/2, V_grid[-1, 0] + dv_im/2]

    # Common percentile normalisation across all epochs
    finite_all = flux[np.isfinite(flux)]
    lo_ep, hi_ep = (np.percentile(finite_all, [2, 99])
                    if len(finite_all) else (0, 1))

    # Width ratio: each epoch image ~ same width as RGB, RGB gets 2× weight
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_ep + 1, subplot_spec=outer[2, :],
        width_ratios=[1] * n_ep + [2],
        wspace=0.04)

    for ep in range(n_ep):
        ax_ep = fig.add_subplot(bot_gs[0, ep])
        ch = np.clip(flux[ep].astype(float), lo_ep, hi_ep)
        ch = (ch - lo_ep) / max(hi_ep - lo_ep, 1e-30)
        ch[~np.isfinite(flux[ep])] = 0.0
        ax_ep.imshow(ch, extent=ext_im, origin='lower', aspect='equal',
                     cmap='gray', vmin=0, vmax=1)
        ax_ep.set_title(f'ep {ep}\nW={W_vals[ep]-W_vals[0]:.2f}', fontsize=6)
        ax_ep.xaxis.set_ticklabels([])
        ax_ep.yaxis.set_ticklabels([])

    ax_rgb = fig.add_subplot(bot_gs[0, n_ep])
    plot_rgb_epochs(data, fit, ax=ax_rgb)

    fig.suptitle(_chunk_suptitle(fit, chunk_id), fontsize=8)
    fig.subplots_adjust(top=0.95, bottom=0.02)
    return fig


def plot_rgb_epochs(data, fit, ax=None, percentile_clip=(5, 99)):
    """
    RGB composite of the first three flux epochs with expected motion vectors.

    R = epoch 0, G = epoch 1, B = epoch 2.  Each channel is independently
    percentile-clipped and scaled to [0, 1].

    Two arrows are drawn at the image centre showing the expected apparent
    UV-plane shift between consecutive epochs derived from the fitted
    principal-axis direction:

        (dU, dV) = dW * tan(theta) * (cos(phi), sin(phi))

    where dW = W_values[i+1] - W_values[i] and (theta, phi) are the polar
    and azimuthal angles of the longest principal axis.

    Parameters
    ----------
    data            : dict from read_chunk()
    fit             : dict from fit_s2()
    ax              : matplotlib Axes (default: current axes)
    percentile_clip : (lo_pct, hi_pct) for per-channel normalisation
    """
    flux   = data['flux_epochs']   # assumed (n_epochs, n_v, n_u)
    U_grid = data['U_grid']        # (n_u,)
    V_grid = data['V_grid']        # (n_v,)
    W_vals = data['W_values']      # (n_epochs,)

    # Normalised RGB, shape (n_v, n_u, 3)
    lo_pct, hi_pct = percentile_clip
    rgb = np.zeros((*flux.shape[1:], 3), dtype=float)
    for c in range(3):
        ch = flux[c].astype(float)
        finite = ch[np.isfinite(ch)]
        if len(finite) > 0:
            lo, hi = np.percentile(finite, [lo_pct, hi_pct])
        else:
            lo = hi = 0
        ch = np.clip(ch, lo, hi)
        ch = (ch - lo) / max(hi - lo, 1e-30)
        ch[~np.isfinite(flux[c])] = 0.0
        rgb[..., c] = ch

    # Expected motion vectors: UV location of S2 minimum at given dW slice,
    # i.e. (l13/s33, l23/s33) * dW from minimizing |L^-1 lag| over (dU, dV).
    p = fit['params']
    du_per_dw = p['l13'] / p['s33']
    dv_per_dw = p['l23'] / p['s33']

    def motion_uv(dw):
        return np.array([du_per_dw * dw, dv_per_dw * dw])

    dW_01 = float(W_vals[1] - W_vals[0])
    dW_12 = float(W_vals[2] - W_vals[1])
    v_01  = motion_uv(dW_01)
    v_12  = motion_uv(dW_12)

    if ax is None:
        ax = plt.gca()

    du  = float(np.median(np.diff(U_grid)))
    dv  = float(np.median(np.diff(V_grid)))
    ext = [U_grid[0, 0] - du/2, U_grid[0, -1] + du/2,
           V_grid[0, 0] - dv/2, V_grid[-1, 0] + dv/2]
    ax.imshow(rgb, extent=ext, origin='lower', aspect='equal')

    span     = U_grid[0, -1] - U_grid[0, 0]
    arrow_kw = dict(length_includes_head=True,
                    head_width=span * 0.012, head_length=span * 0.006,
                    linewidth=1.5)
    u_cen = float(np.mean(U_grid))
    v_cen = float(np.mean(V_grid))
    ax.arrow(u_cen, v_cen, v_01[0], v_01[1], color='yellow', **arrow_kw)
    ax.arrow(u_cen + v_01[0], v_cen + v_01[1],
             v_12[0], v_12[1], color='cyan',   **arrow_kw)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    # ax.legend(handles=[
    #     mpatches.Patch(color='yellow',
    #                    label=f'0->1 (dW={dW_01:.3f} ly)'),
    #     mpatches.Patch(color='cyan',
    #                    label=f'1->2 (dW={dW_12:.3f} ly)'),
    # ], fontsize=7, loc='lower right')

    # ax.set_xlabel('U [ly]', fontsize=8)
    # ax.set_ylabel('V [ly]', fontsize=8)
    # ax.set_title('RGB epochs 0/1/2 + expected motion', fontsize=9)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    return ax


# ---------------------------------------------------------------------------
# HDF5 save / load
# ---------------------------------------------------------------------------

def save_chunk_result(sf, fit, out_fn=None, data_dir='data'):
    """
    Save compute_s2 / fit_s2 outputs to a geometry-named HDF5 file.

    Parameters
    ----------
    sf       : dict returned by compute_s2() (carries the window geometry)
    fit      : dict returned by fit_s2(); if it has a 'jackknife' entry
               ({quantity: stderr}) those are written under fit/jackknife.
    out_fn   : str or None
        Output path; defaults to window_result_path(row, col, size) — i.e.
        sf_fit_r<row>_c<col>_s<size>.h5.
    """
    row  = int(sf.get('row', -1))
    col  = int(sf.get('col', -1))
    size = int(sf.get('size', -1))
    if out_fn is None:
        out_fn = window_result_path(row, col, size, data_dir)
    os.makedirs(os.path.dirname(out_fn) or '.', exist_ok=True)

    fitres = fit['fit']                   # scipy OptimizeResult
    w      = fit['fit_weight']
    resid  = fitres.fun
    sum_w  = float(np.sum(w[w > 0]))
    chi2_dof = float(np.dot(resid, resid) / sum_w) if sum_w > 0 else np.nan

    with h5py.File(out_fn, 'w') as f:
        f.attrs['profile_name'] = getattr(fit.get('profile'), '__name__', 'unknown')
        f.attrs['weighting']    = fit.get('weighting', '')
        f.attrs['row']          = row
        f.attrs['col']          = col
        f.attrs['size']         = size
        f.attrs['chunk_id']     = window_chunk_id(row, col, size, data_dir)

        g = f.create_group('sf')
        g.create_dataset('s2',          data=sf['s2'],          compression='gzip')
        g.create_dataset('n_counts',    data=sf['n_counts'],    compression='gzip')
        g.create_dataset('lag_du',      data=sf['lag_du'])
        g.create_dataset('lag_dv',      data=sf['lag_dv'])
        g.create_dataset('lag_dw',      data=sf['lag_dw'])
        g.create_dataset('epoch_pairs', data=np.array(sf['epoch_pairs'], dtype=np.int32))
        g.create_dataset('w_values',    data=sf['w_values'])
        g.attrs['u_mean']       = float(sf['u_mean'])
        g.attrs['v_mean']       = float(sf['v_mean'])
        g.attrs['w_mean']       = float(sf['w_mean'])
        g.attrs['size_ly']      = float(sf.get('size_ly', np.nan))
        g.attrs['flux_mad_std'] = float(sf['flux_mad_std'])

        g = f.create_group('fit')
        g.create_dataset('s2_pred',      data=fit['s2_pred'],      compression='gzip')
        g.create_dataset('fit_weight',   data=fit['fit_weight'],   compression='gzip')
        g.create_dataset('display_mask', data=fit['display_mask'].astype(np.uint8),
                         compression='gzip')
        g.create_dataset('log_s2_obs',   data=fit['log_s2_obs'])
        g.create_dataset('residuals',    data=resid)
        g.attrs['nfev']     = int(fitres.nfev)
        g.attrs['success']  = bool(fitres.success)
        g.attrs['chi2_dof'] = chi2_dof

        pg = g.create_group('params')
        for k, v in fit['params'].items():
            pg.attrs[k] = float(v)

        jk = fit.get('jackknife')
        if jk:
            jg = g.create_group('jackknife')     # {quantity: stderr}
            for k, v in jk.items():
                jg.attrs[k] = float(v)

    return out_fn


def load_chunk_result(fn):
    """
    Load a file written by save_chunk_result().

    Returns (sf, fit) dicts with the same structure as compute_s2() /
    fit_s2(), minus the scipy OptimizeResult (fit['fit'] is a simple
    namespace with .nfev, .success, .fun).
    """
    import types

    with h5py.File(fn, 'r') as f:
        sf = {
            's2':          f['sf/s2'][:],
            'n_counts':    f['sf/n_counts'][:],
            'lag_du':      f['sf/lag_du'][:],
            'lag_dv':      f['sf/lag_dv'][:],
            'lag_dw':      f['sf/lag_dw'][:],
            'epoch_pairs': [tuple(row) for row in f['sf/epoch_pairs'][:]],
            'w_values':     f['sf/w_values'][:],
            'u_mean':       float(f['sf'].attrs['u_mean']),
            'v_mean':       float(f['sf'].attrs['v_mean']),
            'w_mean':       float(f['sf'].attrs['w_mean']),
            'size_ly':      float(f['sf'].attrs.get('size_ly', np.nan)),
            'flux_mad_std': float(f['sf'].attrs.get('flux_mad_std', np.nan)),
            'row':          int(f.attrs.get('row', -1)),
            'col':          int(f.attrs.get('col', -1)),
            'size':         int(f.attrs.get('size', -1)),
        }

        fg   = f['fit']
        resid = fg['residuals'][:]
        mock  = types.SimpleNamespace(
            fun     = resid,
            nfev    = int(fg.attrs['nfev']),
            success = bool(fg.attrs['success']),
        )
        params = dict(fg['params'].attrs)
        fit = {
            'fit':          mock,
            'params':       params,
            'profile':      None,
            'weighting':    f.attrs.get('weighting', ''),
            's2_pred':      fg['s2_pred'][:],
            'fit_weight':   fg['fit_weight'][:],
            'display_mask': fg['display_mask'][:].astype(bool),
            'log_s2_obs':   fg['log_s2_obs'][:],
        }
        if 'jackknife' in fg:
            fit['jackknife'] = dict(fg['jackknife'].attrs)

    return sf, fit


# ---------------------------------------------------------------------------
# PDF output
# ---------------------------------------------------------------------------

def make_chunk_plots_pdf(res, pdf_path, vmin_sf=0.1, vmax_sf=3.0,
                         vdiff=0.2, uv_range=0.32, data_dir='data', **kwargs):
    """
    Render one plot_full_page per window and write to a PDF, ordered by (row,
    col).  `res` may be a summary array (from process_chunks), a results dict, a
    directory of sf_fit_*.h5, or an iterable of result paths; each window's
    saved fit is loaded from disk one at a time and its image re-read from the
    fullsky map, so this is memory-bounded regardless of grid size.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        for path in _result_paths(res, data_dir):
            sf, fit = load_chunk_result(path)
            row, col, size = sf['row'], sf['col'], sf['size']
            data = read_window(row, col, size, size, data_dir=data_dir)
            cid = window_chunk_id(row, col, size, data_dir)
            label = f'r{row} c{col} s{size}' + (f' (chunk {cid})' if cid >= 0 else '')
            plot_full_page(sf, fit, data, chunk_id=label,
                           vmin_sf=vmin_sf, vmax_sf=vmax_sf,
                           vdiff=vdiff, uv_range=uv_range,
                           newfig=True, **kwargs)
            pdf.savefig()
            plt.close()


# ---------------------------------------------------------------------------
# Parallel batch processing
# ---------------------------------------------------------------------------

def _fit_scalars(params):
    """Flatten a fit's params into scalar quantities to track across jackknife
    samples: the raw L / profile params plus derived axes, angles [deg], and
    drift rates."""
    d = {k: float(v) for k, v in params.items()}
    ax = principal_axes_from_params(params)
    d['a1'], d['a2'], d['a3'] = ax['a1'], ax['a2'], ax['a3']
    d['theta'] = float(np.degrees(ax['theta']))
    d['phi']   = float(np.degrees(ax['phi']))
    d['psi']   = float(np.degrees(ax['psi']))
    d['du_per_dw'] = params['l13'] / params['s33']
    d['dv_per_dw'] = params['l23'] / params['s33']
    return d


def _jackknife_stderr(samples):
    """Block-jackknife standard error per quantity from N delete-one-block fits:
    var = (N-1)/N * sum_i (x_i - mean)^2."""
    out = {}
    for k in samples[0]:
        x = np.array([s[k] for s in samples], float)
        x = x[np.isfinite(x)]
        n = x.size
        out[k] = (float(np.sqrt((n - 1) / n * np.sum((x - x.mean()) ** 2)))
                  if n >= 2 else np.nan)
    return out


def _jackknife_fit(data, k, compute_kw, fit_kw):
    """Delete each of k x k spatial blocks of the window in turn, refit, and
    return {quantity: jackknife stderr}.  Blocks partition the window, so this is
    a grouped (block) jackknife with N = k*k samples; k=2 is the quadrant scheme.

    The spread across deletions estimates fit uncertainty, but is a LOWER bound —
    the blocks share the field's large-scale correlated modes.  Principal-axis
    angles can be noisy under axis flips.
    """
    flux = data['flux_epochs']
    _, ny, nx = flux.shape
    r_edges = np.linspace(0, ny, k + 1).astype(int)
    c_edges = np.linspace(0, nx, k + 1).astype(int)
    samples = []
    for i in range(k):
        for j in range(k):
            d = dict(data)
            f = flux.copy()
            f[:, r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]] = np.nan
            d['flux_epochs'] = f
            try:
                sf = compute_s2(d, **compute_kw)
                samples.append(_fit_scalars(fit_s2(sf, **fit_kw)['params']))
            except Exception:
                continue
    return _jackknife_stderr(samples) if len(samples) >= 2 else {}


def _process_window(spec, data_dir='data', save_dir='data', skip_existing=False,
                    max_nfev=None, background=0.03, arcsinh_scale=0.03,
                    profile=None, weighting='1/r', min_n_fraction=0.1,
                    fit_stride=1, assume_stationary=True, edge_mask_radius=50,
                    min_coverage=0.25, jackknife_k=1):
    """Fit one window, STREAM its full arrays to disk, and return only its
    summary record (or the Exception).  The big arrays never go back to the
    parent, so peak memory is ~n_workers windows regardless of grid size."""
    row, col, size = _window_spec(spec, data_dir)
    out = window_result_path(row, col, size, save_dir)
    sp = profile if profile is not None else weibull_log_s2
    try:
        if skip_existing and os.path.exists(out):
            sf, fit = load_chunk_result(out)          # resume: no recompute
            return out, _summary_record(sf, fit, sp, data_dir, save_dir)
        data = read_window(row, col, size, size, data_dir=data_dir,
                           edge_mask_radius=edge_mask_radius,
                           min_coverage=min_coverage)
        compute_kw = dict(background=background, arcsinh_scale=arcsinh_scale,
                          assume_stationary=assume_stationary)
        fit_kw = dict(profile=profile, max_nfev=max_nfev, weighting=weighting,
                      min_n_fraction=min_n_fraction, fit_stride=fit_stride)
        sf  = compute_s2(data, **compute_kw)
        fit = fit_s2(sf, **fit_kw)
        if jackknife_k > 1:
            fit['jackknife'] = _jackknife_fit(data, jackknife_k,
                                              compute_kw, fit_kw)
        save_chunk_result(sf, fit, out_fn=out, data_dir=save_dir)
        return out, _summary_record(sf, fit, sp, data_dir, save_dir)
    except Exception as exc:
        return out, exc


def process_chunks(specs=None, n_workers=None, max_nfev=None,
                   background=0.03, arcsinh_scale=0.03, profile=None,
                   weighting='1/r', min_n_fraction=0.1, fit_stride=1,
                   assume_stationary=True, edge_mask_radius=50,
                   min_coverage=0.25, jackknife_k=1, data_dir='data',
                   *, save_dir, skip_existing=False):
    """
    Run read_window -> compute_s2 -> fit_s2 on each window in parallel, STREAMING
    each window's full result to disk as it completes.

    Each window's arrays are written to save_dir/sf_fit_r<row>_c<col>_s<size>.h5
    (via save_chunk_result); only its summary row comes back to the parent, so
    peak memory is ~n_workers windows (~181 MB each) instead of growing with the
    grid.  Reload the full arrays later with load_chunk_result / load_results, or
    plot from disk with make_chunk_plots_pdf.

    specs : iterable of window specs — a (row, col, size) triple, or a chunk
            id / key (chunk_windows.csv).  None (default) runs all official
            chunks (official_windows); for the overlap grid pass window_grid(...).
    save_dir : required output directory for the sf_fit_*.h5 files; created if
            missing, and existing files in it are overwritten.
    skip_existing : default False recomputes and overwrites every window.  Set
            True only to resume a crashed run — it skips windows whose output
            file already exists, trusting them as-is (so don't resume across a
            code change, or you'll mix stale results).
    jackknife_k : >1 also runs a k x k block jackknife per window; 1 skips it.

    Returns the summary structured array (identical in construction to
    summarize_chunks) — one row per completed window, each with its `path`.
    Failed windows are printed and omitted.
    """
    import multiprocessing
    from functools import partial
    try:
        import tqdm
        wrap = lambda it, **kw: tqdm.tqdm(it, **kw)
    except ImportError:
        wrap = lambda it, **kw: it

    os.makedirs(save_dir, exist_ok=True)
    if profile is None:
        profile = weibull_log_s2
    if specs is None:
        specs = official_windows(data_dir)
    specs = [_window_spec(s, data_dir) for s in specs]
    # Build the fullmap edge mask once here so the spawned workers mmap it
    # rather than each rebuilding it (which would load the whole noclip map).
    if len(specs):
        _fullsky_edge_mask(data_dir, edge_mask_radius)

    worker = partial(_process_window, data_dir=data_dir, save_dir=save_dir,
                     skip_existing=skip_existing, max_nfev=max_nfev,
                     background=background, arcsinh_scale=arcsinh_scale,
                     profile=profile, weighting=weighting,
                     min_n_fraction=min_n_fraction, fit_stride=fit_stride,
                     assume_stationary=assume_stationary,
                     edge_mask_radius=edge_mask_radius, min_coverage=min_coverage,
                     jackknife_k=jackknife_k)
    with multiprocessing.Pool(n_workers) as pool:
        items = list(wrap(
            pool.imap_unordered(worker, specs),
            total=len(specs)))

    records = []
    for out, val in items:
        if isinstance(val, Exception):
            print(f"FAILED {out}: {val}")
        else:
            records.append(val)
    return _assemble_summary(records, profile)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

# Jackknife quantities that get a *_err column in the summary.
_SUMMARY_JK_KEYS = ('a1', 'a2', 'a3', 'theta', 'phi', 'psi',
                    'du_per_dw', 'dv_per_dw')


def _summary_dtype(profile):
    """Structured dtype for one summary row (shared by process_chunks streaming
    output and summarize_chunks so there is a single summary definition)."""
    return np.dtype([
        ('row',          'i4'),
        ('col',          'i4'),
        ('size',         'i4'),
        ('size_ly',      'f8'),
        ('chunk_id',     'i4'),
        ('path',         'U160'),
        ('u_mean',       'f8'),
        ('v_mean',       'f8'),
        ('w_mean',       'f8'),
        ('du_per_dw',    'f8'),
        ('dv_per_dw',    'f8'),
        ('s11',          'f8'),
        ('s22',          'f8'),
        ('s33',          'f8'),
        ('l12',          'f8'),
        ('l13',          'f8'),
        ('l23',          'f8'),
    ] + [(name, 'f8') for name in profile.param_names] + [
        ('a1',           'f8'),
        ('a2',           'f8'),
        ('a3',           'f8'),
        ('theta',        'f8'),
        ('phi',          'f8'),
        ('psi',          'f8'),
    ] + [(f'{k}_err', 'f8') for k in _SUMMARY_JK_KEYS] + [
        ('chi2_dof',     'f8'),
        ('n_pts',        'i4'),
        ('fit_success',  '?'),
        ('n_epochs',     'i4'),
        ('w_span',       'f8'),
        ('flux_mad_std', 'f8'),
    ])


def _summary_record(sf, fit, profile, data_dir='data', save_dir=None):
    """One summary-row tuple (matching _summary_dtype) for a fitted window — the
    single record definition used by both the streaming worker and
    summarize_chunks, so the two never diverge.  chunk_id is looked up in
    data_dir (chunk_windows.csv); the `path` points into save_dir (where the
    result file lives, = data_dir unless streaming elsewhere)."""
    p = fit['params']
    row, col, size = int(sf['row']), int(sf['col']), int(sf['size'])
    chunk_id = window_chunk_id(row, col, size, data_dir)
    path = window_result_path(row, col, size,
                              data_dir if save_dir is None else save_dir)

    W_values = sf['w_values']
    w_span   = float(W_values.max() - W_values.min())
    n_epochs = sum(1 for i, j in sf['epoch_pairs'] if i == j)

    ax = principal_axes_from_params(p)
    fitres   = fit['fit']
    resid    = fitres.fun
    n_pts    = len(fit['log_s2_obs'])
    fw       = fit['fit_weight']
    sum_w    = float(fw[fw > 0].sum())
    chi2_dof = float(np.sum(resid**2)) / sum_w

    jk = fit.get('jackknife', {})
    jk_vals = tuple(float(jk.get(k, np.nan)) for k in _SUMMARY_JK_KEYS)
    prof_vals = tuple(p[name] for name in profile.param_names)
    return (
        row, col, size, sf.get('size_ly', np.nan), chunk_id, path,
        sf['u_mean'], sf['v_mean'], sf['w_mean'],
        p['l13'] / p['s33'], p['l23'] / p['s33'],
        p['s11'], p['s22'], p['s33'],
        p['l12'], p['l13'], p['l23'],
        *prof_vals,
        ax['a1'], ax['a2'], ax['a3'],
        np.degrees(ax['theta']), np.degrees(ax['phi']), np.degrees(ax['psi']),
        *jk_vals,
        chi2_dof, n_pts, bool(fitres.success),
        n_epochs, w_span,
        sf.get('flux_mad_std', np.nan),
    )


def _assemble_summary(records, profile):
    """Stack summary-row tuples into the structured array, ordered by (row, col).
    Both process_chunks (streaming) and summarize_chunks end here, so their
    outputs are identical in construction."""
    out = np.array(records, dtype=_summary_dtype(profile))
    return out[np.lexsort((out['col'], out['row']))] if out.size else out


def _iter_results(res):
    """Yield (sf, fit) from either an in-memory {key: {sf, fit}} dict or an
    iterable of saved result paths (loaded one at a time, bounding memory)."""
    if isinstance(res, dict):
        for v in res.values():
            yield v['sf'], v['fit']
    else:
        for path in res:
            yield load_chunk_result(path)


def summarize_chunks(res, profile=None, data_dir='data'):
    """
    Build a numpy structured array summarising one row per window.

    res : an in-memory {key: {sf, fit}} dict, OR an iterable of saved result
          paths (loaded one at a time).  The rows are the same _summary_record
          tuples that process_chunks streams back, so the two agree exactly.

    Identity is the window geometry (row, col, size), with a `path` to the saved
    HDF5 and a convenience `chunk_id` (matching official chunk, or -1).  Fields:
    row, col, size, size_ly, chunk_id, path, u_mean, v_mean, w_mean,
    du_per_dw, dv_per_dw, s11..<profile params>, a1, a2, a3, theta, phi, psi,
    *_err (jackknife stderrs; NaN if jackknife_k was 1),
    chi2_dof, n_pts, fit_success, n_epochs, w_span, flux_mad_std.
    """
    if profile is None:
        profile = weibull_log_s2
    records = [_summary_record(sf, fit, profile, data_dir)
               for sf, fit in _iter_results(res)]
    return _assemble_summary(records, profile)


def load_results(paths):
    """Rebuild the full in-memory {path: {sf, fit}} dict from saved result paths
    (only when it fits — ~181 MB per window; fine for the official 115, not the
    468-window overlap grid).  Accepts a summary array (uses its `path` column),
    a directory, or an iterable of paths."""
    return {p: dict(zip(('sf', 'fit'), load_chunk_result(p)))
            for p in _result_paths(paths)}


def _parse_result_path(path):
    """(row, col, size) parsed from an sf_fit_r*_c*_s*.h5 path (for ordering)."""
    m = re.search(r'sf_fit_r(-?\d+)_c(-?\d+)_s(\d+)\.h5$', os.path.basename(path))
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def _result_paths(res, save_dir='data'):
    """Sorted result paths from a summary structured array (its `path` column),
    a results dict, a directory, or an iterable of paths."""
    if isinstance(res, np.ndarray) and res.dtype.names and 'path' in res.dtype.names:
        paths = [str(p) for p in res['path']]
    elif isinstance(res, dict):
        paths = list(res)
    elif isinstance(res, str):
        import glob
        paths = glob.glob(f'{res}/sf_fit_r*_c*_s*.h5')
    else:
        paths = list(res)
    return sorted(paths, key=_parse_result_path)


def result_paths(save_dir='data'):
    """Sorted list of saved sf_fit_*.h5 result files in `save_dir` (e.g. to
    reload / re-summarise a finished run: summarize_chunks(result_paths()))."""
    return _result_paths(save_dir)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    fname = sys.argv[1] if len(sys.argv) > 1 else "data/uvw_chunk_0_products.h5"
    print(f"Reading {fname} ...")
    data = read_chunk(fname)
    print(f"  flux_epochs shape : {data['flux_epochs'].shape}")
    print(f"  W_values          : {data['W_values']}")

    print("Computing S2 ...")
    t0 = time.time()
    result = compute_s2(data)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")

    s2       = result["s2"]
    n_counts = result["n_counts"]
    lag_du   = result["lag_du"]
    lag_dv   = result["lag_dv"]

    print(f"  s2 shape                       : {s2.shape}")
    print(f"  epoch_pairs                    : {result['epoch_pairs']}")
    print(f"  lag_dw [ly]                    : {result['lag_dw']}")
    print(f"  lag_du[:4] and lag_du[-4:] [ly]: {lag_du[:4]}  ...  {lag_du[-4:]}")

    n_epochs = data["flux_epochs"].shape[0]
    n_rows   = data["flux_epochs"].shape[1]
    n_cols   = data["flux_epochs"].shape[2]
    cr, cc   = n_rows - 1, n_cols - 1   # center index (zero-lag)

    for e in range(n_epochs):
        v0 = s2[e, cr, cc]
        n0 = n_counts[e, cr, cc]
        print(f"  same-epoch {e}: s2[{cr},{cc}]={v0:.2e} (should be ~0), N={n0:.0f}")

    # Symmetry check: s2(lag) should equal s2(-lag) for same-epoch pairs.
    # In centered layout, lag +k is at center+k and lag -k at center-k.
    dk = 5
    for e in range(n_epochs):
        a = s2[e, cr + dk, cc + dk]
        b = s2[e, cr - dk, cc - dk]
        print(f"  same-epoch {e}: s2[+{dk},+{dk}]={a:.6f}, s2[-{dk},-{dk}]={b:.6f}  "
              f"(symmetry error {abs(a - b):.2e})")
