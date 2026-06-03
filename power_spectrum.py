#!/usr/bin/env python3
"""
Structure function / power spectrum comparison with Leike+2020.

Uses the full-image resampled epochs to compute azimuthally-averaged S2(r)
in physical units (pc), then overlays Leike+2020 slope references.

S2 is now computed via structure_function.compute_s2 (linear FFT,
zero-padded), consistent with the chunk-level analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_cdt, label, binary_fill_holes
from matplotlib.backends.backend_pdf import PdfPages
from structure_function import (LY_PER_PC, ARCSINH_SCALE, compute_s2,
                                read_fullmap)

NOISE = ARCSINH_SCALE

# Dilation parameters for the log panel
DILATION_SEED_THRESH = 0.10  # seed: (flux - per-epoch median) > this value
DILATION_RADIUS_PIX  = 10    # expand seeds by this many pixels
LOG_CLIP             = NOISE  # clip excess at this floor before log; try NOISE/2


def dilation_mask(flux, dilation_seed_thresh=DILATION_SEED_THRESH,
                  dilation_radius_pix=DILATION_RADIUS_PIX, min_island_size=1000,
                  fill_holes=True):
    """
    Build mask around bright pixels for structure-function analysis.

    Steps:
      1. Dilate: include all finite pixels within dilation_radius_pix Manhattan
         distance of a seed (excess > dilation_seed_thresh), restricted to
         positive excess.
      2. Fill holes: enclosed False regions (not reachable from image border)
         are filled; re-intersect with finite_mask so NaN pixels stay excluded.
         Holes abutting NaN regions are not filled (correct: not truly enclosed).
      3. Remove small islands: connected components < min_island_size pixels
         are dropped.

    Returns (mask, background) where background is the per-epoch median.
    Log users should clip excess at LOG_CLIP before taking log, since filled
    holes may contain pixels with low or negative excess.
    """
    finite_mask = np.isfinite(flux)
    background = np.nanmedian(flux[finite_mask])
    excess = flux - background
    seed = finite_mask & (excess > dilation_seed_thresh)
    dist = distance_transform_cdt(~seed)
    mask = finite_mask & (dist <= dilation_radius_pix)
    if fill_holes:
        mask = binary_fill_holes(mask) & finite_mask
    labeled, _ = label(mask)
    sizes = np.bincount(labeled.ravel())   # sizes[0] = background pixels
    large = np.where(sizes >= min_island_size)[0]
    large = large[large != 0]              # exclude background label
    mask = np.isin(labeled, large)
    return mask, background


def azimuthal_average_sf(sf, pixel_ly=None, n_bins=100):
    """
    Azimuthally average S2 from compute_s2 output (same-epoch pair, index 0).

    Expects fftshifted centered layout from structure_function.compute_s2.
    Returns (r_pc, s2, npairs_per_bin).
    Lags limited to < min(n_rows, n_cols) / 3 pixels.
    """
    s2_grid = sf['s2'][0].astype(float)
    nc_grid = sf['n_counts'][0].astype(float)
    lag_du  = sf['lag_du']
    lag_dv  = sf['lag_dv']
    if pixel_ly is None:
        pos = lag_du[lag_du > 0]
        pixel_ly = float(pos.min()) if pos.size else float(lag_dv[lag_dv > 0].min())
    DU, DV  = np.meshgrid(lag_du, lag_dv)   # 'xy': shape (n_dv, n_du) matches s2[0]
    R_ly    = np.sqrt(DU**2 + DV**2)
    R_pc    = R_ly / LY_PER_PC

    n_rows = (s2_grid.shape[0] + 1) // 2
    n_cols = (s2_grid.shape[1] + 1) // 2
    max_lag_ly = min(n_rows, n_cols) / 3 * pixel_ly

    valid = (nc_grid > 0) & np.isfinite(s2_grid) & (R_ly > 0.5 * pixel_ly) & (R_ly <= max_lag_ly)

    r_min_pc = pixel_ly / LY_PER_PC * 0.9
    r_max_pc = max_lag_ly / LY_PER_PC * 1.1
    bins = np.logspace(np.log10(r_min_pc), np.log10(r_max_pc), n_bins + 1)
    r_centers = np.sqrt(bins[:-1] * bins[1:])

    r_flat  = R_pc[valid].ravel()
    s2_flat = s2_grid[valid].ravel()
    nc_flat = nc_grid[valid].ravel()

    idx = np.searchsorted(bins, r_flat) - 1
    ok  = (idx >= 0) & (idx < n_bins)
    idx, s2_flat, nc_flat = idx[ok], s2_flat[ok], nc_flat[ok]

    s2_sum = np.bincount(idx, weights=s2_flat * nc_flat, minlength=n_bins)
    nc_sum = np.bincount(idx, weights=nc_flat,           minlength=n_bins)

    good   = nc_sum > 0
    s2_out = np.where(good, s2_sum / nc_sum, np.nan)
    return r_centers, s2_out, nc_sum


def reference_line(r_pc, alpha, r0, s2_0):
    """Power-law reference S2 = s2_0 * (r/r0)^alpha."""
    return s2_0 * (r_pc / r0) ** alpha


def plot_panel(ax, epoch_curves, pixel_ly, ny, nx, title):
    """Plot per-epoch S2 curves plus reference lines on ax."""
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(epoch_curves)))
    r_ref = np.logspace(-4, 1, 300)

    anchor_r, anchor_s2 = None, None

    for i, (r_pc, s2) in enumerate(epoch_curves):
        good = np.isfinite(s2) & (s2 > 0)
        if not good.any():
            continue
        ax.loglog(r_pc[good], s2[good], color=colors[i], lw=1.2, label=f'epoch {i}')

        # Fit alpha over 3 px – N/6 px range
        fit_lo = 3 * pixel_ly / LY_PER_PC
        fit_hi = min(ny, nx) / 6 * pixel_ly / LY_PER_PC
        fm = good & (r_pc >= fit_lo) & (r_pc <= fit_hi)
        if fm.sum() >= 5:
            c = np.polyfit(np.log10(r_pc[fm]), np.log10(s2[fm]), 1)
            print(f"    epoch {i}: α = {c[0]:.3f}")

        # Use last valid epoch as anchor for reference lines
        i_a = np.searchsorted(r_pc[good], 0.01)
        i_a = min(i_a, good.sum() - 1)
        anchor_r, anchor_s2 = r_pc[good][i_a], s2[good][i_a]

    if anchor_r is not None:
        ax.loglog(r_ref, reference_line(r_ref, 0.5,  anchor_r, anchor_s2),
                  'k--', lw=0.8, label='r^0.5 (chunks)')
        ax.loglog(r_ref, reference_line(r_ref, 2/3,  anchor_r, anchor_s2),
                  color='0.4', ls='--', lw=0.8, label='r^0.67 (Kolmogorov)')
        ax.loglog(r_ref, reference_line(r_ref, 1.82, anchor_r, anchor_s2),
                  color='0.7', ls='--', lw=0.8, label='r^1.82 (Leike β=2.82)')

    ax.axvspan(2, 100, alpha=0.08, color='red', label='Leike range')
    ax.axvline(pixel_ly / LY_PER_PC, color='gray', ls=':', lw=0.8, label='1 pixel')
    ax.set_xlabel('r [pc]')
    ax.set_ylabel('S₂(r)  [transform units²]')
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.2)
    ax.set_xlim(pixel_ly / LY_PER_PC * 0.5, 5)

    # Auto y-limits from the data, ignoring reference lines
    all_s2 = np.concatenate([s2[np.isfinite(s2) & (s2 > 0)]
                              for _, s2 in epoch_curves
                              if np.any(np.isfinite(s2) & (s2 > 0))])
    if all_s2.size:
        ax.set_ylim(all_s2.min() * 0.3, all_s2.max() * 3)


def compute_s2_log(data, clip_snr=3, noise_scale=NOISE):
    """
    Compute S2 using a log+clip transform in place of arcsinh.

    Applies log(clip((flux - noise_scale) / noise_scale, clip_snr, inf)) to
    each epoch, then calls compute_s2 with arcsinh_scale=None, background=0.
    Off-cloud pixels below the clip floor all map to log(clip_snr), so
    within-off-cloud pairs contribute exactly zero to S2.
    """
    d = dict(data)
    flux = data['flux_epochs'].copy()
    d['flux_epochs'] = np.log(np.clip((flux - noise_scale) / noise_scale,
                                      clip_snr, np.inf))
    return compute_s2(d, arcsinh_scale=None, background=0,
                      subtract_mean='global', assume_stationary=True)


def plot_s2_curves(ax, curves, pixel_ly=None, title='', slope_range_px=(4, 20)):
    """
    Plot a list of (r_pc, s2, label) tuples on ax with Kolmogorov/Leike refs.

    Annotates each curve with its fitted slope over slope_range_px pixels.
    Reference lines are anchored to the first valid curve at r ~ 1 px.
    """
    if pixel_ly is None:
        for r_pc, s2, _ in curves:
            good = np.isfinite(s2) & (s2 > 0)
            if good.any():
                pixel_ly = float(r_pc[good].min()) * LY_PER_PC
                break
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(curves), 1)))
    r_ref  = np.logspace(-5, 1, 400)
    anchor_r = anchor_s2 = None

    for i, (r_pc, s2, label) in enumerate(curves):
        good = np.isfinite(s2) & (s2 > 0)
        if not good.any():
            continue

        fit_lo = slope_range_px[0] * pixel_ly / LY_PER_PC
        fit_hi = slope_range_px[1] * pixel_ly / LY_PER_PC
        fm = good & (r_pc >= fit_lo) & (r_pc <= fit_hi)
        slope_str = ''
        if fm.sum() >= 5:
            slope = np.polyfit(np.log10(r_pc[fm]), np.log10(s2[fm]), 1)[0]
            slope_str = f'  α={slope:.2f}'

        ax.loglog(r_pc[good], s2[good], color=colors[i], lw=1.5,
                  label=label + slope_str)

        if anchor_r is None:
            i_a = min(np.searchsorted(r_pc[good], pixel_ly / LY_PER_PC),
                      good.sum() - 1)
            anchor_r, anchor_s2 = r_pc[good][i_a], s2[good][i_a]

    if anchor_r is not None:
        ax.loglog(r_ref, reference_line(r_ref, 2/3,  anchor_r, anchor_s2),
                  'k--', lw=0.8, alpha=0.5, label='r^0.67 Kolmogorov')
        ax.loglog(r_ref, reference_line(r_ref, 1.82, anchor_r, anchor_s2),
                  color='0.5', ls='--', lw=0.8, alpha=0.5, label='r^1.82 Leike')

    ax.axvline(pixel_ly / LY_PER_PC, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('r [pc]')
    ax.set_ylabel('S₂(r)  [log units²]')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.2)


def make_log_sensitivity_pdf(pdf_path, data_dir='data',
                              clip_snrs=(2, 3, 4, 5), fixed_clip=3,
                              fixed_epoch=0, split_col=2600):
    """
    Three-page PDF exploring log+clip S2 sensitivity.

    Page 1 — clip variation  : fixed_epoch, clip_snr sweeps clip_snrs.
    Page 2 — epoch variation : fixed_clip, epochs 0–4.
    Page 3 — spatial cut     : fixed_clip, fixed_epoch, cols < split_col vs
                                cols >= split_col (split in U pixel space).
    """
    def _sfm_to_curve(sfm, label):
        r_pc, s2, _ = azimuthal_average_sf(sfm)
        return r_pc, s2, label

    def _masked_data(data, col_lo, col_hi):
        d = dict(data)
        flux = data['flux_epochs'].copy()
        if col_lo > 0:
            flux[:, :, :col_lo] = np.nan
        if col_hi < flux.shape[2]:
            flux[:, :, col_hi:] = np.nan
        d['flux_epochs'] = flux
        return d

    with PdfPages(pdf_path) as pdf:

        # --- Page 1: clip variation ---
        print(f'Page 1: clip variation (epoch {fixed_epoch})')
        data = read_fullmap(epochs=fixed_epoch, data_dir=data_dir)
        curves = []
        for clip in clip_snrs:
            print(f'  clip={clip}')
            sfm = compute_s2_log(data, clip_snr=clip)
            curves.append(_sfm_to_curve(sfm, f'clip={clip}'))
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_s2_curves(ax, curves,
                       title=f'Clip sensitivity  (epoch {fixed_epoch})')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: epoch variation ---
        print(f'Page 2: epoch variation (clip={fixed_clip})')
        curves = []
        for ep in range(5):
            print(f'  epoch={ep}')
            data = read_fullmap(epochs=ep, data_dir=data_dir)
            sfm = compute_s2_log(data, clip_snr=fixed_clip)
            curves.append(_sfm_to_curve(sfm, f'epoch {ep}'))
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_s2_curves(ax, curves,
                       title=f'Epoch sensitivity  (clip={fixed_clip})')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: spatial cut ---
        print(f'Page 3: spatial cut at col {split_col} (clip={fixed_clip}, epoch {fixed_epoch})')
        data = read_fullmap(epochs=fixed_epoch, data_dir=data_dir)
        nx = data['flux_epochs'].shape[2]
        U_grid = data['U_grid']
        u_split = float(U_grid[U_grid.shape[0]//2, split_col])
        curves = []
        for label, c0, c1 in [(f'U<{u_split:.2f} ly (cols <{split_col})',  0, split_col),
                               (f'U≥{u_split:.2f} ly (cols ≥{split_col})', split_col, nx)]:
            print(f'  {label}')
            sfm = compute_s2_log(_masked_data(data, c0, c1), clip_snr=fixed_clip)
            curves.append(_sfm_to_curve(sfm, label))
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_s2_curves(ax, curves,
                       title=f'Spatial cut at col {split_col}  (clip={fixed_clip}, epoch {fixed_epoch})')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f'Saved {pdf_path}')


def main():
    print("Loading data...")
    epochs = np.load('data/resampled_epochs.npy')   # (5, ny, nx)
    U = np.load('data/U_grid.npy')
    V = np.load('data/V_grid.npy')
    pixel_ly = abs(float(U[0, 1]) - float(U[0, 0]))
    print(f"Pixel scale: {pixel_ly:.5f} ly = {pixel_ly/LY_PER_PC:.5f} pc")

    n_epochs, ny, nx = epochs.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Structure function S₂(r) — full cloud image, per epoch', fontsize=12)

    print(f"\nComputing per-epoch (seed excess>{DILATION_SEED_THRESH}, "
          f"dilate {DILATION_RADIUS_PIX} px Manhattan):")
    arcsinh_curves, log_curves = [], []
    for i in range(n_epochs):
        flux = epochs[i]
        mask, bg = dilation_mask(flux)
        excess = flux - bg
        print(f"  epoch {i}: bg={bg:.4f}, coverage={mask.mean():.4f}")

        flux_masked = np.where(mask, flux, np.nan)
        single_data = {
            'flux_epochs': flux_masked[np.newaxis],
            'U_grid':      U,
            'V_grid':      V,
            'W_values':    np.array([0.0]),
        }
        sf_e = compute_s2(single_data, background=bg,
                           arcsinh_scale=NOISE, clip_percentiles=None,
                           assume_stationary=False)
        r_pc, s2, _ = azimuthal_average_sf(sf_e, pixel_ly)
        arcsinh_curves.append((r_pc, s2))

        # log panel: pre-compute log field, bypass compute_s2 transforms
        log_field = np.where(mask, np.log(np.maximum(excess, LOG_CLIP) / NOISE), np.nan)
        single_log = {
            'flux_epochs': log_field[np.newaxis],
            'U_grid':      U,
            'V_grid':      V,
            'W_values':    np.array([0.0]),
        }
        sf_log = compute_s2(single_log, background=None,
                             arcsinh_scale=None, clip_percentiles=None,
                             assume_stationary=False)
        r_pc, s2, _ = azimuthal_average_sf(sf_log, pixel_ly)
        log_curves.append((r_pc, s2))

    print("\narcsinh alphas:")
    plot_panel(axes[0], arcsinh_curves, pixel_ly, ny, nx,
               f'arcsinh((flux−bg) / {NOISE}), same footprint')
    print("log alphas:")
    plot_panel(axes[1], log_curves, pixel_ly, ny, nx,
               f'log((flux−bg) / {NOISE}), within {DILATION_RADIUS_PIX} px of excess>{DILATION_SEED_THRESH}')

    plt.tight_layout()
    outfile = 'sf_leike_comparison.pdf'
    plt.savefig(outfile)
    print(f"\nSaved {outfile}")
    plt.close()


if __name__ == '__main__':
    main()
