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
sys.path.insert(0, os.path.expanduser('~/projects/util_efs/python'))

import numpy as np
import h5py
from numpy.fft import rfft2, irfft2
from itertools import combinations
import scipy.linalg
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import util_efs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_chunk(filename: str) -> dict:
    """
    Read a uvw_chunk_NNN_products.h5 file.

    Returns a dict with keys:
      flux_epochs : (n_epochs, n_rows, n_cols) float64  — may contain NaNs
      U_grid      : (n_rows, n_cols) float64            — U coords [ly]
      V_grid      : (n_rows, n_cols) float64            — V coords [ly]
      W_values    : (n_epochs,) float64                 — W coord per epoch [ly]
    """
    with h5py.File(filename, "r") as f:
        return {
            "flux_epochs": f["raw_data/flux_epochs"][:],
            "U_grid":      f["raw_data/U_grid"][:],
            "V_grid":      f["raw_data/V_grid"][:],
            "W_values":    f["raw_data/W_values"][:],
        }


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
               assume_stationary: bool = True) -> dict:
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
    assume_stationary: if True, replace the per-lag ⟨f²⟩ estimate with the
                       global per-epoch variance, matching the
                       S2 = 2*(var − AC) formula used in get_sf_2d.
                       Default False.

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

    for e in range(flux.shape[0]):
        flux[e] -= np.nanmean(flux[e])

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
        _next_pow2(2 * n_rows - 1),
        _next_pow2(2 * n_cols - 1),
    )

    # Pre-compute per-epoch FFTs
    masks = np.isfinite(flux).astype(np.float64)
    f_    = np.where(masks.astype(bool), flux, 0.0)

    M_fft = [rfft2(masks[e], s=fft_shape) for e in range(n_epochs)]

    if assume_stationary:
        # Implement S2 = 2*(var - AC); images are already mean-subtracted above
        # so var = mean(f²) and AC(0) = var, ensuring S2(0) = 0.
        valid = [masks[e].astype(bool) for e in range(n_epochs)]
        evars = [float(f_[e][valid[e]].var()) if valid[e].any() else 0.0
                 for e in range(n_epochs)]
        MF_fft  = [rfft2(masks[e] * f_[e],   s=fft_shape) for e in range(n_epochs)]
        MF2_fft = [evars[e] * M_fft[e]                    for e in range(n_epochs)]
    else:
        MF_fft  = [rfft2(masks[e] * f_[e],    s=fft_shape) for e in range(n_epochs)]
        MF2_fft = [rfft2(masks[e] * f_[e]**2, s=fft_shape) for e in range(n_epochs)]

    out_shape = (n_pairs, 2 * n_rows - 1, 2 * n_cols - 1)
    s2       = np.full(out_shape, np.nan)
    n_counts = np.zeros(out_shape, dtype=np.float64)

    for k, (i, j) in enumerate(epoch_pairs):
        dsq, N = _sum_diff_sq_full(
            M_fft[i], MF_fft[i], MF2_fft[i],
            M_fft[j], MF_fft[j], MF2_fft[j],
            fft_shape, n_rows, n_cols,
        )
        valid = N > 0
        s2[k][valid]  = dsq[valid] / N[valid]
        n_counts[k]   = N

    s2       = np.fft.fftshift(s2,       axes=(1, 2))
    n_counts = np.fft.fftshift(n_counts, axes=(1, 2))
    lag_du   = np.fft.fftshift(lag_du)
    lag_dv   = np.fft.fftshift(lag_dv)

    return {
        "s2":          s2,
        "n_counts":    n_counts,
        "lag_du":      lag_du,
        "lag_dv":      lag_dv,
        "lag_dw":      lag_dw,
        "epoch_pairs": epoch_pairs,
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
# Structure function fitting
# ---------------------------------------------------------------------------

def log_s2_model(params, lags):
    """
    Evaluate log10 S2 for the Weibull structure function model:

        S2(lag) = var_inf * (1 - exp(-r^beta))^(alpha/beta)

    where r = |L⁻¹ lag|.  At small lag S2 ≈ var_inf * r^alpha;
    at large lag S2 → var_inf.

    L is upper-triangular with the diagonal given directly as scale values
    (in ly), not exponentiated:

        L = [[s11,  l12,  l13],
             [0,    s22,  l23],
             [0,    0,    s33]]

    Parameters
    ----------
    params : array-like [s11, s22, s33, l12, l13, l23, alpha, beta, var_inf]
             or a params dict as returned by fit_s2(...)['params']
    lags   : (N, 3) array of [dU, dV, dW] lag vectors [ly]

    Returns
    -------
    (N,) array of log10 S2 values
    """
    if isinstance(params, dict):
        params = [params['s11'], params['s22'], params['s33'],
                  params['l12'], params['l13'], params['l23'],
                  params['alpha'], params['beta'], params['var_inf']]
    s11, s22, s33, l12, l13, l23, alpha, beta, var_inf = params
    L = np.array([[s11,  l12,  l13],
                  [0.0,  s22,  l23],
                  [0.0,  0.0,  s33]])
    r = scipy.linalg.solve_triangular(L, np.asarray(lags).T)   # (3, N)
    r_norm = np.maximum(np.sqrt((r * r).sum(axis=0)), 1e-100)
    weibull = np.maximum(1.0 - np.exp(-r_norm ** beta), 1e-300)
    return np.log10(var_inf) + (alpha / beta) * np.log10(weibull)


def predict_s2(params, lag_du, lag_dv, lag_dw):
    """
    Evaluate the S2 model on a full (n_pairs, n_lag_v, n_lag_u) grid.

    Parameters
    ----------
    params  : array-like or dict of model parameters (see log_s2_model)
    lag_du  : (n_lag_u,) U lag coordinates [ly]
    lag_dv  : (n_lag_v,) V lag coordinates [ly]
    lag_dw  : (n_pairs,) W lag per plane [ly]

    Returns
    -------
    (n_pairs, n_lag_v, n_lag_u) array of predicted S2 values
    """
    DV, DU = np.meshgrid(lag_dv, lag_du, indexing='ij')
    du_flat = DU.ravel()
    dv_flat = DV.ravel()
    n_lag_v, n_lag_u = DV.shape
    n_pairs = len(lag_dw)
    s2_pred = np.empty((n_pairs, n_lag_v, n_lag_u))
    for k, dw in enumerate(lag_dw):
        lags = np.column_stack([du_flat, dv_flat, np.full(du_flat.size, dw)])
        s2_pred[k] = (10 ** log_s2_model(params, lags)).reshape(n_lag_v, n_lag_u)
    return s2_pred


def params_from_principal_axes(a1, a2, a3, theta, phi, psi, alpha,
                               beta=2.0, var_inf=0.001):
    """
    Build model params dict from principal-axis ellipsoid description.

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
    alpha : float
        Power-law slope of S2.
    beta : float
        Weibull shape parameter controlling transition sharpness (default 2).
    var_inf : float
        Asymptotic S2 value at large lags (default 0.001).

    Returns
    -------
    dict with keys s11, s22, s33, l12, l13, l23, alpha, beta, var_inf
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

    return dict(s11=s11, s22=s22, s33=s33, l12=l12, l13=l13, l23=l23,
                alpha=alpha, beta=beta, var_inf=var_inf)


def principal_axes_from_params(params):
    """
    Decompose model params into principal-axis ellipsoid description.

    Inverse of params_from_principal_axes.

    Parameters
    ----------
    params : dict with keys s11, s22, s33, l12, l13, l23, alpha[, beta, var_inf]
             or array-like [s11, s22, s33, l12, l13, l23, alpha, ...]

    Returns
    -------
    a1, a2, a3 : float  -- semi-axis lengths [ly], sorted descending
    theta      : float  -- polar angle of a1 from W axis [rad]
    phi        : float  -- azimuthal angle of a1 from U in UV plane [rad]
    psi        : float  -- roll of a2/a3 around a1 [rad]
    alpha      : float  -- power-law slope
    """
    if not isinstance(params, dict):
        s11, s22, s33, l12, l13, l23, alpha = params[:7]
    else:
        s11 = params['s11']; s22 = params['s22']; s33 = params['s33']
        l12 = params['l12']; l13 = params['l13']; l23 = params['l23']
        alpha = params['alpha']

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

    return a1, a2, a3, theta, phi, psi, alpha


def fit_s2(result: dict,
           guess: dict | None = None,
           inner_uv_pixels: int = 200,
           min_same_epoch_lag_pix: int = 4,
           s2_floor: float = 10**-3.75,
           noise_scale_dex: float = 0.1) -> dict:
    """
    Fit S2(dU, dV, dW) = var_inf*(1-exp(-r^beta))^(alpha/beta) to the
    structure function, where r = |L⁻¹ lag|.
    See log_s2_model() for the full parameterization.

    Fitting is done in log10 space with a pseudo-Huber (soft_l1) loss, with
    residuals normalized by noise_scale_dex.

    Parameters
    ----------
    result              : dict returned by compute_s2()
    guess               : optional dict with any of the keys 's11','s22','s33',
                          'l12','l13','l23','alpha','beta','var_inf'
                          to override defaults
    inner_uv_pixels     : half-width in pixels of the UV lag region to include;
                          keeps lags in [-inner, +inner) in each UV direction
    min_same_epoch_lag_pix : for dW=0 planes, exclude the central square of
                          ±this many pixels in each UV direction (avoids zero
                          lag and PSF-correlated region)
    s2_floor            : S2 values below this are clipped before taking log
    noise_scale_dex     : residuals are divided by this before applying the
                          pseudo-Huber loss (sets the inlier/outlier boundary)

    Returns
    -------
    dict with keys:
      fit          : scipy.optimize.least_squares result object
      params       : dict of named fitted parameters
      s2_pred      : (n_pairs, 2*p, 2*p) predicted S2 on the inner UV grid,
                     where p = inner_uv_pixels
      lag_du_inner : (2*p,) U lags used for s2_pred
      lag_dv_inner : (2*p,) V lags used for s2_pred
      lags_used    : (N, 3) [dU, dV, dW] of data points included in fit
      log_s2_obs   : (N,) observed log10 S2 values
      log_s2_pred  : (N,) predicted log10 S2 at lags_used
    """
    s2_arr      = result['s2']
    lag_du      = result['lag_du']
    lag_dv      = result['lag_dv']
    lag_dw      = result['lag_dw']
    epoch_pairs = result['epoch_pairs']

    n_pairs, n_lag_v, n_lag_u = s2_arr.shape
    cv = n_lag_v // 2
    cu = n_lag_u // 2
    p  = inner_uv_pixels

    v_sl = slice(cv - p, cv + p)
    u_sl = slice(cu - p, cu + p)
    dv_inner = lag_dv[v_sl]     # (2p,)
    du_inner = lag_du[u_sl]     # (2p,)
    DV, DU = np.meshgrid(dv_inner, du_inner, indexing='ij')   # (2p, 2p)

    # Row/col offsets from zero-lag center within the inner slice
    row_off = np.abs(np.arange(2 * p) - p)
    col_off = np.abs(np.arange(2 * p) - p)
    ROW_OFF, COL_OFF = np.meshgrid(row_off, col_off, indexing='ij')
    near_zero = (ROW_OFF <= min_same_epoch_lag_pix) & (COL_OFF <= min_same_epoch_lag_pix)

    # Build flat data arrays
    dU_list, dV_list, dW_list, log_s2_list = [], [], [], []

    fit_mask = np.zeros((n_pairs, n_lag_v, n_lag_u), dtype=bool)

    for k, (i, j) in enumerate(epoch_pairs):
        plane = s2_arr[k, v_sl, u_sl]
        mask  = np.isfinite(plane)
        if i == j:
            mask &= ~near_zero
        fit_mask[k, v_sl, u_sl] = mask
        log_s2 = np.log10(np.maximum(plane, s2_floor))
        sel = mask.ravel()
        dU_list.append(DU.ravel()[sel])
        dV_list.append(DV.ravel()[sel])
        dW_list.append(np.full(sel.sum(), lag_dw[k]))
        log_s2_list.append(log_s2.ravel()[sel])

    lags_flat  = np.column_stack([np.concatenate(dU_list),
                                   np.concatenate(dV_list),
                                   np.concatenate(dW_list)])  # (N, 3)
    log_s2_obs = np.concatenate(log_s2_list)                  # (N,)

    # Default var_inf guess: median of finite S2 in the fit region
    inner_s2 = s2_arr[:, v_sl, u_sl]
    var_inf_guess = float(np.nanmedian(inner_s2[np.isfinite(inner_s2)]))

    # Free params: [s11, s22, s33, l12, l13, l23, alpha, beta, var_inf]
    g = guess or {}
    p0 = np.array([g.get('s11',     1.0),
                   g.get('s22',     1.0),
                   g.get('s33',     1.0),
                   g.get('l12',     0.0),
                   g.get('l13',     0.0),
                   g.get('l23',     0.0),
                   g.get('alpha',   0.4),
                   g.get('beta',    2.0),
                   g.get('var_inf', var_inf_guess)])

    def _unpack(params):
        s11, s22, s33, l12, l13, l23, alpha, beta, var_inf = params
        return dict(s11=s11, s22=s22, s33=s33,
                    l12=l12, l13=l13, l23=l23,
                    alpha=alpha, beta=beta, var_inf=var_inf)

    def _residuals(params):
        return (log_s2_obs - log_s2_model(params, lags_flat)) / noise_scale_dex

    fit = least_squares(_residuals, p0, loss='soft_l1', f_scale=1.0)

    return {
        'fit':        fit,
        'params':     _unpack(fit.x),
        's2_pred':    predict_s2(fit.x, lag_du, lag_dv, lag_dw),
        'fit_mask':   fit_mask,
        'log_s2_obs': log_s2_obs,
    }


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
    mask = fit['fit_mask'] if fit is not None else np.isfinite(s2_arr)

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
                          uv_range=0.2, vmin_sf=1e-4, vmax_sf=5e-3,
                          vdiff=1e-4):
    """
    Populate a region (given by subplot_spec or None for a new figure) with
    the 5x9 thumbnail grid.  Returns the figure and the axes array.
    """
    s2_arr   = sf['s2']
    lag_du   = sf['lag_du']
    lag_dv   = sf['lag_dv']
    pairs    = sf['epoch_pairs']
    s2_pred  = fit['s2_pred']
    fit_mask = fit['fit_mask']
    n_pairs  = len(pairs)

    pairs_per_row = 3
    n_rows = (n_pairs + pairs_per_row - 1) // pairs_per_row
    n_cols = pairs_per_row * 3   # 9

    fig = plt.gcf()
    if subplot_spec is None:
        inner_gs = gridspec.GridSpec(n_rows, n_cols, hspace=0, wspace=0)
    else:
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols, subplot_spec=subplot_spec, hspace=0, wspace=0)

    axes = np.array([[fig.add_subplot(inner_gs[r, c])
                      for c in range(n_cols)]
                     for r in range(n_rows)])

    diff_arr = np.array(
        [(s2_arr[k] - s2_pred[k]) * fit_mask[k] for k in range(n_pairs)])
    type_data = [('S2', s2_arr, False), ('pred', s2_pred, False),
                 ('diff', diff_arr, True)]

    for k in range(n_pairs):
        row   = k // pairs_per_row
        group = k % pairs_per_row
        for ti, (_, images, is_diff) in enumerate(type_data):
            col = group * 3 + ti
            ax  = axes[row, col]
            plt.sca(ax)
            if is_diff:
                util_efs.imshow(images[k].T, lag_du, lag_dv,
                                vmin=-vdiff, vmax=vdiff, cmap='binary')
            else:
                util_efs.imshow(images[k].T, lag_du, lag_dv, log=True,
                                vmin=vmin_sf, vmax=vmax_sf, cmap='binary')
            ax.set_xlim(-uv_range, uv_range)
            ax.set_ylim(-uv_range, uv_range)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

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
    fitres  = fit['fit']
    resid   = fitres.fun
    chi2dof = float(np.sum(resid**2)) / max(len(resid) - len(fitres.x), 1)
    n_pts   = len(fit['log_s2_obs'])
    a1, a2, a3, theta, phi, psi, alpha = principal_axes_from_params(
        fit['params'])
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
                       uv_range=0.2, vmin_sf=1e-4, vmax_sf=5e-3,
                       vdiff=1e-4):
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
                         vmax_sf=vmax_sf, vdiff=vdiff)
    fig.suptitle(_chunk_suptitle(fit, chunk_id), fontsize=7)
    fig.subplots_adjust(top=0.93, left=0.08, right=0.99, bottom=0.01)
    return fig


def plot_full_page(sf, fit, data, chunk_id=None, figsize=(8.5, 11),
                   uv_range=0.2, vmin_sf=1e-4, vmax_sf=5e-3, vdiff=1e-4):
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
    """
    fig = plt.figure(figsize=figsize)

    # Outer: 3 rows (thumbnails / 1-D plots / RGB), 3 cols so RGB can be
    # centred by occupying the middle column only.
    outer = gridspec.GridSpec(3, 3, figure=fig,
                              height_ratios=[3, 1.2, 1.4],
                              width_ratios=[1, 2, 1],
                              hspace=0.3, wspace=0.3)

    # Thumbnails span the full width of the top row
    _make_thumbnail_axes(sf, fit, subplot_spec=outer[0, :],
                         uv_range=uv_range, vmin_sf=vmin_sf,
                         vmax_sf=vmax_sf, vdiff=vdiff)

    # 1-D SF plots: split the full-width row into two equal halves
    inner_1d = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1, :], wspace=0.3)
    ax_lag = fig.add_subplot(inner_1d[0, 0])
    ax_ell = fig.add_subplot(inner_1d[0, 1])
    plot_s2_1d(sf, fit, ellipsoidal=False, ax=ax_lag)
    ax_lag.set_title('$S_2$ vs $|$lag$|$', fontsize=8)
    plot_s2_1d(sf, fit, ellipsoidal=True, ax=ax_ell)
    ax_ell.set_title('$S_2$ vs ellipsoidal radius', fontsize=8)

    # RGB composite: centred, using only the middle column
    ax_rgb = fig.add_subplot(outer[2, 1])
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

    # Expected motion vectors
    a1, a2, a3, theta, phi, psi, alpha = principal_axes_from_params(
        fit['params'])

    def motion_uv(dw):
        return np.array([dw * np.sin(theta) / np.cos(theta) * np.cos(phi),
                         dw * np.sin(theta) / np.cos(theta) * np.sin(phi)])

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
