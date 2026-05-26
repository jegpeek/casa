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
import scipy.ndimage
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import util_efs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_chunk(filename: str, edge_mask_radius: int = 20) -> dict:
    """
    Read a uvw_chunk_NNN_products.h5 file.

    Returns a dict with keys:
      flux_epochs : (n_epochs, n_rows, n_cols) float64  — may contain NaNs
      U_grid      : (n_rows, n_cols) float64            — U coords [ly]
      V_grid      : (n_rows, n_cols) float64            — V coords [ly]
      W_values    : (n_epochs,) float64                 — W coord per epoch [ly]

    Pixels within edge_mask_radius of any NaN/zero pixel are set to NaN.
    Bad pixels are always contiguous with the image boundary (verified across
    all chunks), so this safely erodes image edges without masking interiors.
    """
    with h5py.File(filename, "r") as f:
        flux = f["raw_data/flux_epochs"][:]
        U_grid   = f["raw_data/U_grid"][:]
        V_grid   = f["raw_data/V_grid"][:]
        W_values = f["raw_data/W_values"][:]

    if edge_mask_radius > 0:
        for ep in range(flux.shape[0]):
            bad = ~np.isfinite(flux[ep]) | (flux[ep] == 0)
            if bad.any():
                dist = scipy.ndimage.distance_transform_edt(~bad)
                flux[ep][dist <= edge_mask_radius] = np.nan

    return {
        "flux_epochs": flux,
        "U_grid":      U_grid,
        "V_grid":      V_grid,
        "W_values":    W_values,
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
               assume_stationary: bool = True,
               background: float | None = 0.03,
               arcsinh_scale: float | None = 0.03) -> dict:
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
    background       : if not None, subtract this constant from all epochs
                       before any other processing.  Use to remove a common
                       instrumental background shared across epochs.
    arcsinh_scale    : if not None, apply arcsinh(flux / arcsinh_scale) after
                       background subtraction and clipping but before per-epoch
                       mean subtraction.  Sets the transition scale between
                       linear (noise-dominated) and logarithmic (signal-dominated)
                       behaviour; typically set to the per-pixel noise level.

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
        flux -= background

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
        "u_mean":      float(np.mean(U_grid)),
        "v_mean":      float(np.mean(V_grid)),
        "w_mean":      float(np.mean(W)),
        "w_values":    W.copy(),
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
    weibull = np.maximum(1.0 - np.exp(-r ** beta), 1e-300)
    return np.log10(var_inf) + (alpha / beta) * np.log10(weibull)

weibull_log_s2.n_params      = 3
weibull_log_s2.param_names   = ['alpha', 'beta', 'var_inf']
weibull_log_s2.default_guess = [0.4, 2.0, None]   # None → data-driven
weibull_log_s2.param_bounds  = [(1e-3, np.inf), (1e-3, np.inf), (1e-6, np.inf)]


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
    s2_pred = np.empty((n_pairs, n_lag_v, n_lag_u))
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
    fit_mask = np.zeros((n_pairs, n_lag_v, n_lag_u), dtype=bool)

    for k, (i, j) in enumerate(epoch_pairs):
        plane = s2_arr[k, v_sl, u_sl]
        n_win = n_counts[k, v_sl, u_sl]
        n_max = n_counts[k].max()
        mask  = np.isfinite(plane) & (n_win >= min_n_fraction * n_max)
        if i == j:
            mask &= ~near_zero
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
    return fit_mask, lags_flat, log_s2_obs


def _point_weights(lags_flat, sf, weighting):
    """Per-point weights w_i; objective = sum(w_i*(obs-pred)^2/sigma^2)."""
    if weighting is None:
        return np.ones(len(lags_flat))
    if weighting == '1/r':
        r_uv = np.hypot(lags_flat[:, 0], lags_flat[:, 1])
        lag_step = float(abs(sf['lag_du'][1] - sf['lag_du'][0]))
        return 1.0 / np.maximum(r_uv, lag_step)
    raise ValueError(f"Unknown weighting: {weighting!r}")


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

    fit_mask, lags_flat, log_s2_obs = _make_fit_data(
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
    if weighting == '1/r':
        lag_step = float(abs(lag_du[1] - lag_du[0]))
        w_2d = 1.0 / np.maximum(np.hypot(DU_full, DV_full), lag_step)
    else:
        w_2d = np.ones_like(DU_full)
    fit_weight = fit_mask * w_2d   # (n_pairs, n_lag_v, n_lag_u)

    return {
        'fit':        mock_fit,
        'params':     params_dict,
        'profile':    profile,
        'weighting':  weighting,
        's2_pred':    predict_s2(params_arr, sf['lag_du'], sf['lag_dv'],
                                 sf['lag_dw'], profile=profile),
        'fit_weight': fit_weight,
        'log_s2_obs': log_s2_obs,
    }


def fit_s2(result: dict,
           profile=None,
           guess: dict | None = None,
           inner_uv_pixels: int = 200,
           min_same_epoch_lag_pix: int = 4,
           s2_floor: float = 10**-3.75,
           noise_scale_dex: float = 0.1,
           min_n_fraction: float = 0.1,
           fit_stride: int = 2,
           max_nfev: int | None = None,
           weighting='1/r') -> dict:
    """
    Fit S2(dU, dV, dW) to the structure function using a pluggable 1D profile.

    The full parameter vector is [s11, s22, s33, l12, l13, l23, *profile_params].
    Fitting is done in log10 space with a pseudo-Huber (soft_l1) loss, with
    residuals normalised by noise_scale_dex.

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

    fit_mask, lags_flat, log_s2_obs = _make_fit_data(
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
    # scale=0.3 ly places saturation at the inner-window edge, keeping the
    # power-law regime well-sampled; elongation=1 forces l13=l23=0 (no shift),
    # so use 5 instead.
    auto_geom = params_from_uvshift(du, dv, scale=0.3, elongation=5.0)
    geom_p0 = [g.get(k, auto_geom[k]) for k in _GEOM_KEYS]
    prof_p0  = [g.get(name, (amp_guess if default is None else default))
                for name, default in zip(profile.param_names,
                                         profile.default_guess)]
    p0 = np.array(geom_p0 + prof_p0)

    sqrt_w = np.sqrt(_point_weights(lags_flat, result, weighting))

    # Optimise with log(s11), log(s22), log(s33) to enforce positivity and
    # prevent collapse to the degenerate s→0 minimum.  Profile params with
    # positive bounds are also log-transformed.
    _N_LOG = 3   # first 3 geom params (s11, s22, s33)
    prof_bounds = getattr(profile, 'param_bounds',
                          [(-np.inf, np.inf)] * profile.n_params)
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

    pre = least_squares(_residuals_profile, prof_q0, loss='soft_l1',
                        f_scale=1.0, max_nfev=200)
    q0 = np.concatenate([geom_q0, pre.x])

    # Stage 2: full optimisation from the pre-warmed starting point.
    fit = least_squares(_residuals, q0, loss='soft_l1', f_scale=1.0,
                        max_nfev=max_nfev)

    fit_params = _from_opt(fit.x)

    # Detect degenerate collapse (Cholesky diagonals → 0): any s < ~3 pixels.
    # The degenerate minimum has lower chi² but is physically meaningless
    # (ellipsoid collapses to a point, model = var_inf everywhere).  Fall back
    # to the stage-1 result: initial geometry with profile pre-fitted.
    lag_step = float(abs(result['lag_du'][1] - result['lag_du'][0]))
    collapsed = min(fit_params[:3]) < 3 * lag_step
    if collapsed:
        # Degenerate minimum: fall back to stage-1 result (initial geometry +
        # pre-fitted profile).  The mock fit in build_fit_result will carry the
        # correct residuals for these params; don't override with fit.fun.
        fit_params = _from_opt(np.concatenate([geom_q0, pre.x]))

    out = build_fit_result(result, fit_params, profile=profile,
                           inner_uv_pixels=inner_uv_pixels,
                           min_same_epoch_lag_pix=min_same_epoch_lag_pix,
                           s2_floor=s2_floor,
                           noise_scale_dex=noise_scale_dex,
                           min_n_fraction=min_n_fraction,
                           fit_stride=fit_stride,
                           weighting=weighting)
    out['collapsed'] = collapsed
    if not collapsed:
        out['fit'] = fit   # replace mock with real optimizer result
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
    s2_pred    = fit['s2_pred']
    fit_weight = fit['fit_weight']
    fit_mask   = fit_weight > 0
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
        m = fit_mask[k]
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

    # Pre-compute mask overlay (white tint on excluded pixels)
    du_step = lag_du[1] - lag_du[0]
    dv_step = lag_dv[1] - lag_dv[0]
    mask_extent = [lag_du[0] - du_step / 2, lag_du[-1] + du_step / 2,
                   lag_dv[0] - dv_step / 2, lag_dv[-1] + dv_step / 2]

    for k in range(n_pairs):
        row   = k // pairs_per_row
        group = k % pairs_per_row

        # RGBA overlay: white at 35% alpha where excluded, transparent elsewhere
        excl = ~fit_mask[k]                          # (n_lag_v, n_lag_u)
        overlay = np.zeros((*excl.shape, 4))
        overlay[excl] = [1.0, 0.0, 0.0, 0.35]

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
            ax.imshow(overlay, extent=mask_extent, origin='lower',
                      aspect='auto', interpolation='nearest')
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
    fitres     = fit['fit']
    resid      = fitres.fun
    sum_w      = float(fit['fit_weight'][fit['fit_weight'] > 0].sum())
    chi2dof    = float(np.sum(resid**2)) / sum_w
    n_pts      = len(fit['log_s2_obs'])
    ax = principal_axes_from_params(fit['params'])
    a1, a2, a3 = ax['a1'], ax['a2'], ax['a3']
    theta, phi = ax['theta'], ax['phi']
    collapsed_str = "  [FALLBACK]" if fit.get('collapsed') else ""
    return (
        f"chunk {chunk_id}   "
        f"$\\chi^2$/dof={chi2dof:.2f}   "
        f"N={n_pts/1e6:.2f}M   "
        f"$a_1$={a1:.3f} ly   "
        f"$a_2/a_1$={a2/a1:.2f}   "
        f"$a_3/a_1$={a3/a1:.2f}   "
        f"$\\theta$={np.degrees(theta):.1f}$^\\circ$   "
        f"$\\phi$={np.degrees(phi):.1f}$^\\circ$"
        f"{collapsed_str}"
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
# Parallel batch processing
# ---------------------------------------------------------------------------

def _process_chunk(fn, max_nfev=None, background=0.03, arcsinh_scale=0.03,
                   profile=None, weighting='1/r', min_n_fraction=0.1,
                   fit_stride=1):
    try:
        data = read_chunk(fn)
        sf   = compute_s2(data, background=background, arcsinh_scale=arcsinh_scale)
        fit  = fit_s2(sf, profile=profile, max_nfev=max_nfev, weighting=weighting,
                      min_n_fraction=min_n_fraction, fit_stride=fit_stride)
        return fn, dict(sf=sf, fit=fit)
    except Exception as exc:
        return fn, exc


def process_chunks(fns, n_workers=None, max_nfev=None,
                   background=0.03, arcsinh_scale=0.03, profile=None,
                   weighting='1/r', min_n_fraction=0.1, fit_stride=1):
    """
    Run read_chunk -> compute_s2 -> fit_s2 on each file in parallel.

    Returns a dict mapping filename -> {'sf': ..., 'fit': ...}.
    Failed chunks are printed and omitted from the result.
    """
    import multiprocessing
    from functools import partial
    try:
        import tqdm
        wrap = lambda it, **kw: tqdm.tqdm(it, **kw)
    except ImportError:
        wrap = lambda it, **kw: it

    worker = partial(_process_chunk, max_nfev=max_nfev, background=background,
                     arcsinh_scale=arcsinh_scale, profile=profile,
                     weighting=weighting, min_n_fraction=min_n_fraction,
                     fit_stride=fit_stride)
    with multiprocessing.Pool(n_workers) as pool:
        items = list(wrap(
            pool.imap_unordered(worker, fns),
            total=len(fns)))

    res = {}
    for fn, val in items:
        if isinstance(val, Exception):
            print(f"FAILED {fn}: {val}")
        else:
            res[fn] = val
    return res


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summarize_chunks(res, profile=None):
    """
    Build a numpy structured array summarising one entry per chunk from the
    dict returned by process_chunks().

    Fields
    ------
    chunk_id   : integer parsed from filename
    chunk_name : filename string (up to 128 chars)
    u_mean, v_mean, w_mean : mean spatial coordinates of the chunk [ly]
    du_per_dw, dv_per_dw   : apparent UV drift rate [ly / ly] = l13/s33, l23/s33;
                             use for quiver(u_mean, v_mean, du_per_dw, dv_per_dw)
    s11..var_inf           : raw fitted L-matrix parameters and Weibull params
    a1, a2, a3             : principal semi-axes [ly], descending
    theta, phi, psi        : orientation of a1 [deg]; theta from W, phi in UV plane
    a2_over_a1, a3_over_a1 : axis ratios (elongation diagnostics)
    chi2_dof               : reduced chi-squared of the fit
    n_pts                  : number of lag points used in fit
    fit_success            : scipy optimizer success flag
    n_epochs               : number of flux epochs in the chunk
    w_span                 : max(W) - min(W) [ly]
    """
    import re

    if profile is None:
        profile = weibull_log_s2

    dtype = np.dtype([
        ('chunk_id',     'i4'),
        ('chunk_name',   'U128'),
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
        ('chi2_dof',     'f8'),
        ('n_pts',        'i4'),
        ('fit_success',  '?'),
        ('collapsed',    '?'),
        ('n_epochs',     'i4'),
        ('w_span',       'f8'),
    ])

    rows = []
    for fn, val in res.items():
        sf  = val['sf']
        fit = val['fit']
        p   = fit['params']

        m = re.search(r'uvw_chunk_(\d+)_products', fn)
        chunk_id = int(m.group(1)) if m else -1

        u_mean   = sf['u_mean']
        v_mean   = sf['v_mean']
        w_mean   = sf['w_mean']
        W_values = sf['w_values']
        w_span   = float(W_values.max() - W_values.min())
        n_epochs = sum(1 for i, j in sf['epoch_pairs'] if i == j)

        ax = principal_axes_from_params(p)
        a1, a2, a3 = ax['a1'], ax['a2'], ax['a3']
        theta, phi, psi = ax['theta'], ax['phi'], ax['psi']

        fitres   = fit['fit']
        resid    = fitres.fun
        n_pts    = len(fit['log_s2_obs'])
        fw       = fit['fit_weight']
        sum_w    = float(fw[fw > 0].sum())
        chi2_dof = float(np.sum(resid**2)) / sum_w

        prof_vals = tuple(p[name] for name in profile.param_names)
        rows.append((
            chunk_id, fn,
            u_mean, v_mean, w_mean,
            p['l13'] / p['s33'], p['l23'] / p['s33'],
            p['s11'], p['s22'], p['s33'],
            p['l12'], p['l13'], p['l23'],
            *prof_vals,
            a1, a2, a3,
            np.degrees(theta), np.degrees(phi), np.degrees(psi),
            chi2_dof, n_pts, bool(fitres.success),
            bool(fit.get('collapsed')),
            n_epochs, w_span,
        ))

    out = np.array(rows, dtype=dtype)
    return out[np.argsort(out['chunk_id'])]


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
