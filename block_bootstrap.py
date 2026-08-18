"""
Block bootstrap for the 3D structure function.

Motivation
----------
The delete-one-block jackknife (k=2) gives only 4 resamples per window, which
is far too few to characterise a distribution and forces a normal-theory error
bar.  A block bootstrap resamples the k x k sub-blocks of a window WITH
replacement, giving as many replicates as we care to draw.

Resampling rule (as specified by the user)
------------------------------------------
Draw multiplicities m_b for each of the K = k*k blocks from Multinomial(K, 1/K),
so sum_b m_b = K.  A pixel pair (x, y) contributing to lag r then enters the
S2 estimator with weight m_{b(x)} * m_{b(y)}, where b(.) is the block
containing the pixel.  Concretely:

  * both endpoints in blocks drawn once  -> weight 1 (the ordinary case)
  * either endpoint in a block drawn 0x  -> weight 0 (pair drops out entirely)
  * both endpoints in the same block drawn twice -> weight 4

This is the natural pair-level analogue of resampling blocks with replacement:
if a block appears twice it contributes two copies of every pixel it holds, so
a within-block pair appears 2 x 2 = 4 times.

Why this is cheap
-----------------
Both the numerator and the denominator of S2 are *bilinear* in the block
multiplicities.  Writing D_bc(r) and N_bc(r) for the sum-of-squared-differences
and pair count restricted to pixel pairs with one endpoint in block b and the
other in block c,

    D*(r) = sum_{b,c} m_b m_c D_bc(r),     N*(r) = sum_{b,c} m_b m_c N_bc(r),
    S2*(r) = D*(r) / N*(r).

So we pay the FFT cost ONCE per window to build the K^2 block-pair
correlations, and every subsequent replicate is a weighted sum of precomputed
arrays -- no new FFTs, no new pixel passes.  The precompute is ~K^2/2 times the
work of a single compute_s2, but each replicate afterwards costs milliseconds
instead of ~0.4 s, and (far more importantly) the expensive part, fit_s2,
is unchanged.

Under the stationary approximation used by compute_s2 (assume_stationary=True),
D_bc is not stored directly.  Instead

    D*(r) = (Ef2_i* + Ef2_j*) * N*(r) - 2 * X*(r)

where X_bc(r) is the cross-correlation of the masked fields and Ef2_e* is the
block-weighted mean square of epoch e.  We store N_bc and X_bc and the
per-block per-epoch sums needed for Ef2_e*.

Fixed vs resampled preprocessing
--------------------------------
Background subtraction, percentile clipping, the arcsinh transform and the
global mean subtraction are treated as FIXED preprocessing, computed once on
the full window and held constant across replicates.  Rationale: S2 is
invariant to a constant offset of the field (the offset cancels exactly
between the E[f^2] and E[f_i f_j] terms), so the global mean is a numerical
device to avoid catastrophic cancellation rather than an estimated parameter.
Re-deriving the clip percentiles per replicate would additionally make the
valid-pixel mask itself random, which is not what the resampling rule
describes.  What IS resampled: the pair counts, the cross-correlations, and
the per-epoch mean squares.

Memory
------
Block-pair correlations are stored clipped to the lag window actually used by
the fit (|dU|, |dV| <= inner_uv_pixels + 2) in float32.  For k=3 on a 400 px
window with 15 epoch pairs this is roughly 300-400 MB per window, which is why
workers must be limited; see bootstrap_windows.py.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from scipy.fft import rfft2, irfft2, next_fast_len

import structure_function as sfmod


# ---------------------------------------------------------------------------
# Block geometry
# ---------------------------------------------------------------------------

def block_edges(n: int, k: int) -> np.ndarray:
    """k+1 integer edges splitting range(n) into k near-equal blocks."""
    return np.linspace(0, n, k + 1).round().astype(int)


def block_bounds(n_rows: int, n_cols: int, k: int):
    """List of K = k*k blocks as (r0, r1, c0, c1), row-major."""
    re = block_edges(n_rows, k)
    ce = block_edges(n_cols, k)
    return [(re[i], re[i + 1], ce[j], ce[j + 1])
            for i in range(k) for j in range(k)]


# ---------------------------------------------------------------------------
# Preprocessing (mirrors compute_s2 exactly, up to the point of correlation)
# ---------------------------------------------------------------------------

def _preprocess(data, clip_percentiles, fill_nans, background,
                arcsinh_scale, subtract_mean):
    """Return (flux, masks, flux_mad_std) using compute_s2's exact recipe."""
    flux = data["flux_epochs"].copy()

    if background is not None:
        bg = np.broadcast_to(np.asarray(background, dtype=float), (flux.shape[0],))
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

    all_valid = np.concatenate([flux[e][np.isfinite(flux[e])]
                                for e in range(flux.shape[0])])
    if all_valid.size > 0:
        med = float(np.median(all_valid))
        flux_mad_std = float(np.median(np.abs(all_valid - med)) * 1.4826)
    else:
        flux_mad_std = np.nan

    if subtract_mean == 'global':
        if all_valid.size > 0:
            flux -= float(all_valid.mean())
    elif subtract_mean == 'epoch':
        for e in range(flux.shape[0]):
            flux[e] -= np.nanmean(flux[e])
    elif subtract_mean != 'none':
        raise ValueError(f"subtract_mean must be 'global', 'epoch', or 'none'; "
                         f"got {subtract_mean!r}")

    if fill_nans:
        for e in range(flux.shape[0]):
            plane = flux[e]
            flux[e] = np.where(np.isfinite(plane), plane, np.nanmean(plane))

    masks = np.isfinite(flux).astype(np.float64)
    return flux, masks, flux_mad_std


# ---------------------------------------------------------------------------
# Block-pair correlation
# ---------------------------------------------------------------------------

def _xcorr_blocks(a_fft, b_fft, fft_shape, h_a, w_a, h_b, w_b):
    """
    Cross-correlation of two sub-blocks in natural lag order.

    Returns C with shape (h_a + h_b - 1, w_a + w_b - 1) where

        C[p + h_a - 1, q + w_a - 1] = sum_{r,c} A[r, c] * B[r + p, c + q]

    for p in [-(h_a - 1), h_b - 1] and q likewise.  Inputs are rfft2 of A and
    B zero-padded to fft_shape, which must be >= (h_a + h_b - 1, w_a + w_b - 1).
    """
    c = irfft2(np.conj(a_fft) * b_fft, s=fft_shape)
    fr, fc = fft_shape
    out = np.empty((h_a + h_b - 1, w_a + w_b - 1))
    # non-negative lags p = 0 .. h_b-1 live at rows 0 .. h_b-1
    # negative lags     p = -(h_a-1) .. -1 live at rows fr-(h_a-1) .. fr-1
    out[h_a - 1:, w_a - 1:] = c[:h_b,               :w_b]
    out[:h_a - 1, w_a - 1:] = c[fr - h_a + 1:fr,    :w_b]
    out[h_a - 1:, :w_a - 1] = c[:h_b,               fc - w_a + 1:fc]
    out[:h_a - 1, :w_a - 1] = c[fr - h_a + 1:fr,    fc - w_a + 1:fc]
    return out


def precompute_blocks(data: dict,
                      k: int = 3,
                      inner_uv_pixels: int = 200,
                      clip_percentiles=(0.002, 0.998),
                      fill_nans: bool = False,
                      background=0.03,
                      arcsinh_scale=0.03,
                      subtract_mean: str = 'global') -> dict:
    """
    Precompute everything needed to rebuild S2 for arbitrary block weights.

    Returns a dict holding, for every ordered block pair (b, c) and every
    epoch pair, the pair count N_bc and cross term X_bc restricted to the lag
    window used by the fit.  Pass the result to s2_from_weights().

    Only assume_stationary=True is supported (the mode all PROFILES use); the
    non-stationary path would require storing f^2 correlations per block pair,
    tripling the memory for no benefit to the current fits.
    """
    flux, masks, flux_mad_std = _preprocess(
        data, clip_percentiles, fill_nans, background, arcsinh_scale,
        subtract_mean)

    U_grid, V_grid, W = data["U_grid"], data["V_grid"], data["W_values"]
    n_epochs, n_rows, n_cols = flux.shape

    f_ = np.where(masks.astype(bool), flux, 0.0)

    blocks = block_bounds(n_rows, n_cols, k)
    K = len(blocks)

    # Per-block, per-epoch sums for the block-weighted mean square.
    # Ef2_e*(m) = sum_b m_b * sf2[e, b] / sum_b m_b * nval[e, b]
    sf2 = np.zeros((n_epochs, K))
    nval = np.zeros((n_epochs, K))
    for bi, (r0, r1, c0, c1) in enumerate(blocks):
        for e in range(n_epochs):
            mm = masks[e, r0:r1, c0:c1].astype(bool)
            sf2[e, bi] = float((f_[e, r0:r1, c0:c1][mm] ** 2).sum())
            nval[e, bi] = float(mm.sum())

    # One padded FFT shape big enough for every block pair
    hmax = max(r1 - r0 for r0, r1, _, _ in blocks)
    wmax = max(c1 - c0 for _, _, c0, c1 in blocks)
    fft_shape = (next_fast_len(2 * hmax - 1), next_fast_len(2 * wmax - 1))

    M_fft = np.empty((n_epochs, K), dtype=object)
    F_fft = np.empty((n_epochs, K), dtype=object)
    for bi, (r0, r1, c0, c1) in enumerate(blocks):
        for e in range(n_epochs):
            M_fft[e, bi] = rfft2(masks[e, r0:r1, c0:c1], s=fft_shape)
            F_fft[e, bi] = rfft2(masks[e, r0:r1, c0:c1] * f_[e, r0:r1, c0:c1],
                                 s=fft_shape)

    same_pairs = [(e, e) for e in range(n_epochs)]
    cross_pairs = list(combinations(range(n_epochs), 2))
    epoch_pairs = same_pairs + cross_pairs
    n_pairs = len(epoch_pairs)
    lag_dw = np.array([float(W[j] - W[i]) for i, j in epoch_pairs])

    # Lag window retained.  fit_s2 uses |lag| <= inner_uv_pixels and
    # estimate_uvshift reaches one index further, so keep a small margin.
    P = int(min(inner_uv_pixels + 2, n_rows - 1, n_cols - 1))
    win = 2 * P + 1                      # global dv index 0 <-> dv = -P

    Nb, Xb, slices = {}, {}, {}
    for bi, (r0b, r1b, c0b, c1b) in enumerate(blocks):
        hb, wb = r1b - r0b, c1b - c0b
        for ci, (r0c, r1c, c0c, c1c) in enumerate(blocks):
            hc, wc = r1c - r0c, c1c - c0c
            # global lag range spanned by this block pair
            dv_lo, dv_hi = r0c - r1b + 1, r1c - r0b - 1
            du_lo, du_hi = c0c - c1b + 1, c1c - c0b - 1
            v0, v1 = max(dv_lo, -P), min(dv_hi, P)
            u0, u1 = max(du_lo, -P), min(du_hi, P)
            if v0 > v1 or u0 > u1:
                continue                  # no overlap with the fit window
            rs = slice(v0 - dv_lo, v1 - dv_lo + 1)
            cs = slice(u0 - du_lo, u1 - du_lo + 1)
            nb = np.empty((n_pairs, v1 - v0 + 1, u1 - u0 + 1), dtype=np.float32)
            xb = np.empty_like(nb)
            for pi, (i, j) in enumerate(epoch_pairs):
                nb[pi] = _xcorr_blocks(M_fft[i, bi], M_fft[j, ci],
                                       fft_shape, hb, wb, hc, wc)[rs, cs]
                xb[pi] = _xcorr_blocks(F_fft[i, bi], F_fft[j, ci],
                                       fft_shape, hb, wb, hc, wc)[rs, cs]
            Nb[(bi, ci)] = nb
            Xb[(bi, ci)] = xb
            slices[(bi, ci)] = (slice(v0 + P, v1 + P + 1),
                                slice(u0 + P, u1 + P + 1))

    du_step = float(U_grid[0, 1] - U_grid[0, 0])
    dv_step = float(V_grid[1, 0] - V_grid[0, 0])
    size_px = int(data.get('size', -1))
    du_pix = (float(np.median(np.abs(np.diff(U_grid[0]))))
              if U_grid.shape[1] > 1 else np.nan)

    return dict(
        Nb=Nb, Xb=Xb, slices=slices, blocks=blocks, K=K, k=k,
        sf2=sf2, nval=nval, epoch_pairs=epoch_pairs, lag_dw=lag_dw,
        n_epochs=n_epochs, n_rows=n_rows, n_cols=n_cols, P=P, win=win,
        du_step=du_step, dv_step=dv_step,
        u_mean=float(np.mean(U_grid)), v_mean=float(np.mean(V_grid)),
        w_mean=float(np.mean(W)), w_values=W.copy(),
        flux_mad_std=flux_mad_std,
        row=int(data.get('row', -1)), col=int(data.get('col', -1)),
        size=size_px, size_ly=size_px * du_pix if size_px > 0 else np.nan,
    )


def s2_from_weights(pre: dict, m: np.ndarray) -> dict:
    """
    Rebuild an S2 result dict for block multiplicities m (length K).

    The returned dict matches compute_s2's contract (same keys, same full lag
    shape and fftshifted ordering) so it can be handed straight to fit_s2.
    Lags outside the retained window are NaN with zero counts; fit_s2 never
    looks there.
    """
    m = np.asarray(m, dtype=np.float64)
    K, P, win = pre['K'], pre['P'], pre['win']
    n_pairs = len(pre['epoch_pairs'])
    n_rows, n_cols = pre['n_rows'], pre['n_cols']

    Nacc = np.zeros((n_pairs, win, win), dtype=np.float64)
    Xacc = np.zeros((n_pairs, win, win), dtype=np.float64)
    for (bi, ci), nb in pre['Nb'].items():
        w = m[bi] * m[ci]
        if w == 0.0:
            continue
        vs, us = pre['slices'][(bi, ci)]
        Nacc[:, vs, us] += w * nb
        Xacc[:, vs, us] += w * pre['Xb'][(bi, ci)]

    # Block-weighted mean square per epoch
    num = pre['sf2'] @ m
    den = pre['nval'] @ m
    with np.errstate(invalid='ignore', divide='ignore'):
        ef2 = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)

    out_shape = (n_pairs, 2 * n_rows - 1, 2 * n_cols - 1)
    s2 = np.full(out_shape, np.nan, dtype=np.float32)
    n_counts = np.zeros(out_shape, dtype=np.int32)

    cv, cu = n_rows - 1, n_cols - 1          # index of lag 0 after fftshift
    vsl = slice(cv - P, cv + P + 1)
    usl = slice(cu - P, cu + P + 1)

    for pi, (i, j) in enumerate(pre['epoch_pairs']):
        N = Nacc[pi]
        valid = N >= 0.5
        dsq = (ef2[i] + ef2[j]) * N - 2.0 * Xacc[pi]
        plane = np.full((win, win), np.nan, dtype=np.float32)
        plane[valid] = (dsq[valid] / N[valid]).astype(np.float32)
        s2[pi, vsl, usl] = plane
        n_counts[pi, vsl, usl] = np.rint(N).astype(np.int32)

    lag_du = np.empty(2 * n_cols - 1)
    lag_du[:n_cols] = np.arange(n_cols) * pre['du_step']
    lag_du[n_cols:] = np.arange(-(n_cols - 1), 0) * pre['du_step']
    lag_dv = np.empty(2 * n_rows - 1)
    lag_dv[:n_rows] = np.arange(n_rows) * pre['dv_step']
    lag_dv[n_rows:] = np.arange(-(n_rows - 1), 0) * pre['dv_step']

    return {
        "s2": s2, "n_counts": n_counts,
        "lag_du": np.fft.fftshift(lag_du), "lag_dv": np.fft.fftshift(lag_dv),
        "lag_dw": pre['lag_dw'], "epoch_pairs": pre['epoch_pairs'],
        "u_mean": pre['u_mean'], "v_mean": pre['v_mean'],
        "w_mean": pre['w_mean'], "w_values": pre['w_values'],
        "flux_mad_std": pre['flux_mad_std'],
        "row": pre['row'], "col": pre['col'],
        "size": pre['size'], "size_ly": pre['size_ly'],
    }


def draw_multiplicities(K: int, rng) -> np.ndarray:
    """Multinomial(K, 1/K) block multiplicities -- blocks resampled with
    replacement, total count preserved."""
    return rng.multinomial(K, np.full(K, 1.0 / K)).astype(np.float64)
