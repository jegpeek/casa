#!/usr/bin/env python3
"""
Compare the Cas A light-echo structure function with Leike+2020.

This module provides three capabilities:

  1. reproduce_leike_fig4 / power_3d — reproduce the Leike+2020 3D power
     spectrum of the dust extinction cube (optionally per-octant, with a
     full-volume slope sanity check against the published 2.52/2.82 values).
  2. plot_echo_leike_s2 — overlay the echo and Leike S2(r) curves on common
     axes (the key comparison plot).
  3. make_log_sensitivity_pdf — explore how robust the echo S2 is to choices
     of clip threshold, epoch, and spatial sub-region.

All echo S2 curves are computed via structure_function.compute_s2 (linear
FFT, zero-padded), consistent with the chunk-level analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from matplotlib.backends.backend_pdf import PdfPages
from structure_function import LY_PER_PC, compute_s2, read_fullmap

# Unit conversions for the Leike+2020 extinction density.  Leike, Glatzle &
# Enßlin 2020 (A&A 639, A138) reconstruct the differential dust extinction as a
# Gaia G-band optical depth ("e-folds") per parsec — see their Fig. 2 caption
# ("differential extinction in e-folds per parsec") and Fig. 14 ("G-band dust
# extinction density in e-folds per parsec").
MAG_PER_EFOLD = 2.5 / np.log(10)   # e-folds (natural-log tau) -> extinction mag
A_G_OVER_A_V  = 0.789              # Gaia G / Johnson V extinction ratio (R_V=3.1);
                                   # Wang & Chen 2019, ApJ 877, 116


def av_per_pc(extinction_density):
    """Convert Leike+2020 G-band extinction density to A_V [mag / pc].

    Input is the native Leike cube unit: differential dust extinction as a
    Gaia G-band optical depth ("e-folds") per parsec.  Two factors take it to
    Johnson-V magnitudes per pc:
        A_V = density * MAG_PER_EFOLD / A_G_OVER_A_V
    i.e. e-folds -> magnitudes (2.5/ln10 ≈ 1.086), then Gaia G -> V
    (A_G = 0.789 A_V).  A_G/A_V really depends on stellar SED and total A_V
    because G is broad; we adopt one representative value and leave that
    modelling to StarHorse/Leike.
    """
    return extinction_density * MAG_PER_EFOLD / A_G_OVER_A_V


def azimuthal_average_sf(sf, pixel_ly=None, n_bins=100, pair=0):
    """
    Azimuthally average S2 from compute_s2 output.

    pair : index along the pair axis of sf['s2'].  Same-epoch (dW=0) pairs come
           first, so pair=e selects epoch e's structure function (default 0).
    Expects fftshifted centered layout from structure_function.compute_s2.
    Returns (r_pc, s2, npairs_per_bin).
    Lags limited to < min(n_rows, n_cols) / 3 pixels.
    """
    s2_grid = sf['s2'][pair].astype(float)
    nc_grid = sf['n_counts'][pair].astype(float)
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


def compute_s2_echo(data, clip_threshold=0.09, norm=1.0, transform='log'):
    """
    Compute S2 of the clipped echo field.

    Forms field = norm * clip(flux, clip_threshold, inf), applies the requested
    transform, and calls compute_s2 with arcsinh_scale=None (this path does its
    own transform, so compute_s2's arcsinh/background steps are skipped).

    clip_threshold : lower clip on the flux [flux units].  read_fullmap already
                     removes the per-epoch instrumental background
                     (NOCLIP_BACKGROUNDS ~ 0.3) so off-cloud flux is ~0; the
                     default 0.09 is ~3x the ~0.03 per-pixel noise.
    transform      : 'log'    -> S2 of log(field)  [fill-factor-sensitive form]
                     'linear' -> S2 of field itself [intensity/A(V)-sensitive]
    norm           : multiplies the field after clipping (default 1).  For
                     'linear' this scales S2 by norm² — the hook for matching the
                     echo intensity/A(V) amplitude to Leike.  Under 'log' it is
                     an additive constant that cancels in S2 (no effect).

    Off-cloud pixels below clip_threshold all map to the same value, so
    within-off-cloud pairs contribute exactly zero to S2.
    """
    d = dict(data)
    flux = data['flux_epochs'].copy()
    field = norm * np.clip(flux, clip_threshold, np.inf)
    d['flux_epochs'] = np.log(field) if transform == 'log' else field
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
                             clip_thresholds=(0.06, 0.09, 0.12, 0.15),
                             fixed_clip=0.09, fixed_epoch=0, split_col=2600):
    """
    Three-page PDF exploring log+clip S2 sensitivity.

    Page 1 — clip variation  : fixed_epoch, clip_threshold sweeps clip_thresholds.
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
        for clip in clip_thresholds:
            print(f'  clip={clip:g}')
            sfm = compute_s2_echo(data, clip_threshold=clip)
            curves.append(_sfm_to_curve(sfm, f'clip={clip:g}'))
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_s2_curves(ax, curves,
                       title=f'Clip sensitivity  (epoch {fixed_epoch})')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: epoch variation ---
        print(f'Page 2: epoch variation (clip={fixed_clip:g})')
        curves = []
        for ep in range(5):
            print(f'  epoch={ep}')
            data = read_fullmap(epochs=ep, data_dir=data_dir)
            sfm = compute_s2_echo(data, clip_threshold=fixed_clip)
            curves.append(_sfm_to_curve(sfm, f'epoch {ep}'))
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_s2_curves(ax, curves,
                       title=f'Epoch sensitivity  (clip={fixed_clip:g})')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: spatial cut ---
        print(f'Page 3: spatial cut at col {split_col} (clip={fixed_clip:g}, epoch {fixed_epoch})')
        data = read_fullmap(epochs=fixed_epoch, data_dir=data_dir)
        nx = data['flux_epochs'].shape[2]
        U_grid = data['U_grid']
        u_split = float(U_grid[U_grid.shape[0]//2, split_col])
        curves = []
        for label, c0, c1 in [(f'U<{u_split:.2f} ly (cols <{split_col})',  0, split_col),
                               (f'U≥{u_split:.2f} ly (cols ≥{split_col})', split_col, nx)]:
            print(f'  {label}')
            sfm = compute_s2_echo(_masked_data(data, c0, c1),
                                  clip_threshold=fixed_clip)
            curves.append(_sfm_to_curve(sfm, label))
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_s2_curves(ax, curves,
                       title=f'Spatial cut at col {split_col}  (clip={fixed_clip:g}, epoch {fixed_epoch})')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f'Saved {pdf_path}')


def load_leike_cube(h5_path, field_kind='linear', rho_floor=1e-12):
    """Load and preprocess the Leike mean_std.h5 extinction cube.

    Parameters
    ----------
    h5_path   : path to mean_std.h5
    field_kind: 'linear' (extinction density rho, e-folds/pc)
                'log'    (log rho, dimensionless)
    rho_floor : lower clip before log (field_kind='log' only)

    Returns
    -------
    cube : (740, 740, 540) float64 array, mean-subtracted
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        cube = f['mean'][:]
    s = cube.astype(np.float64)
    if field_kind == 'log':
        s = np.log(np.maximum(s, rho_floor))
    s -= s.mean()
    return s


def power_3d(field, n_bins=30, dx_pc=1.0, subtract_mean=True):
    """3D shell-averaged power spectrum of a field array.

    Normalization matches Leike+2020: P = |F|^2 / N_voxels.  k is in cycles/pc
    (numpy fftfreq convention; scale = 1/k).  Slope of log P vs log k = -beta
    directly, so a fit recovers the published beta (2.52 linear / 2.82 log).

    Parameters
    ----------
    field         : 3D float array (the whole Leike cube or any subregion)
    n_bins        : number of log-spaced k bins
    dx_pc         : voxel size in pc
    subtract_mean : if True (default) drop the k=0 (total-dust) mode by
                    subtracting the field mean before the FFT.

    Returns
    -------
    k_pc : (n,) wavenumber in cycles/pc;  scale = 1/k_pc  [pc]
    P    : (n,) shell-averaged |F|^2/N
    """
    s = np.asarray(field, dtype=np.float64)
    if subtract_mean:
        s = s - s.mean()
    F = np.fft.fftn(s)
    power = (F.real**2 + F.imag**2) / s.size
    del F

    kx = np.fft.fftfreq(s.shape[0], d=dx_pc)
    ky = np.fft.fftfreq(s.shape[1], d=dx_pc)
    kz = np.fft.fftfreq(s.shape[2], d=dx_pc)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
    power = power.ravel()
    del KX, KY, KZ

    ok = kmag > 0
    kmag, power = kmag[ok], power[ok]

    bins = np.logspace(np.log10(kmag.min()), np.log10(kmag.max()), n_bins + 1)
    which = np.digitize(kmag, bins)

    k_cen, P_k = [], []
    for b in range(1, len(bins)):
        m = which == b
        if m.any():
            k_cen.append(kmag[m].mean())
            P_k.append(power[m].mean())
    return np.array(k_cen), np.array(P_k)


def leike_s2_avg(cube, slices=0, axis=2, dx_pc=1.0, log_floor=1e-3,
                 transform='log'):
    """Average 2D S2 over one or more slices of the Leike extinction cube.

    cube     : raw (nx, ny, nz) extinction density array (e-folds/pc) as
               loaded from mean_std.h5 — not logged or mean-subtracted
    slices   : integer offset(s) from the centre slice of `cube` along `axis`.
               0 (default) is the centre — the midplane for a full cube; pass an
               array such as np.arange(-10, 11) to average a range.  None
               averages every slice.  Offsets outside the cube are dropped.
    axis     : 0, 1, or 2 — axis to slice along
    dx_pc    : voxel size in pc
    log_floor: lower clip applied to the density floor (before log if any)
    transform: 'log' -> S2 of log(field); 'linear' -> S2 of field itself.

    Returns averaged S2 dict compatible with azimuthal_average_sf.
    """
    n_total = cube.shape[axis]
    if slices is None:
        indices = np.arange(n_total)
    else:
        indices = n_total // 2 + np.atleast_1d(np.asarray(slices, dtype=int))
        indices = indices[(indices >= 0) & (indices < n_total)]
    if indices.size == 0:
        raise ValueError('slices select no planes within the cube')

    s2_num = None
    nc_den = None
    sf_ref = None

    for idx in indices:
        slc = [slice(None), slice(None), slice(None)]
        slc[axis] = int(idx)
        slab = np.asarray(cube[tuple(slc)], dtype=np.float64)  # 2D
        # ensure (n_rows, n_cols) orientation; axis=0 gives (ny,nz), others (nx,n*)
        if axis != 0:
            slab = slab.T

        ny, nx = slab.shape
        pix_ly = dx_pc * LY_PER_PC
        x_ly = (np.arange(nx) - nx // 2) * pix_ly
        y_ly = (np.arange(ny) - ny // 2) * pix_ly
        U_grid, V_grid = np.meshgrid(x_ly, y_ly)
        field = np.maximum(slab[np.newaxis], log_floor)
        d = {
            'flux_epochs': np.log(field) if transform == 'log' else field,
            'U_grid':      U_grid,
            'V_grid':      V_grid,
            'W_values':    np.array([0.0]),
        }

        sf = compute_s2(d, arcsinh_scale=None, background=0,
                        subtract_mean='global', assume_stationary=True)

        valid = np.isfinite(sf['s2'])
        w    = np.where(valid, sf['n_counts'].astype(float), 0.0)
        s2_w = np.where(valid, sf['n_counts'].astype(float) * sf['s2'], 0.0)

        if s2_num is None:
            s2_num, nc_den, sf_ref = s2_w, w, sf
        else:
            s2_num += s2_w
            nc_den += w

    result = dict(sf_ref)
    result['s2']       = np.where(nc_den > 0, s2_num / nc_den, np.nan).astype(np.float32)
    result['n_counts'] = nc_den.astype(np.int32)
    return result


def _load_cube(leike_h5):
    """Load the raw Leike extinction mean cube (nx, ny, nz), e-folds/pc."""
    import h5py
    with h5py.File(leike_h5, 'r') as f:
        return f['mean'][:]


def echo_s2_curve(fullsky=None, sf=None, clip_threshold=0.09, epoch=0, stride=1,
                  data_dir='data', transform='log', norm=1.0):
    """Azimuthally-averaged S2(r) of the light echo.  Returns (r_pc, s2).

    epoch         : same-epoch structure function to use (0 = first epoch).
    clip_threshold: lower clip on the flux (see compute_s2_echo).
    transform     : 'log' -> S2 of log(field); 'linear' -> S2 of field itself.
    norm          : echo field normalization; scales linear S2 by norm², no-op
                    under 'log' (see compute_s2_echo).
    sf            : precomputed compute_s2_echo output to reuse (skips loading
                    and the FFT); when given, fullsky/clip_threshold/stride/
                    transform/norm are ignored.  The full 5-epoch SF must be
                    loaded for `epoch` to index epochs directly.
    """
    if sf is None:
        if fullsky is None:
            fullsky = read_fullmap(data_dir=data_dir, stride=stride)
        sf = compute_s2_echo(fullsky, clip_threshold=clip_threshold,
                             transform=transform, norm=norm)
    r_pc, s2, _ = azimuthal_average_sf(sf, pair=epoch)
    return r_pc, s2


def leike_s2_curve(cube=None, leike_h5='leike2020/mean_std.h5',
                   footprint='full', leike_axis=2, slices=0,
                   log_floor=1e-3, stride=1, transform='log'):
    """Azimuthally-averaged S2(r) of the Leike+2020 cube.  Returns (r_pc, s2).

    footprint : 'full', 'above' (high half of leike_axis, +z above the plane),
                or 'below' (low half).
    slices    : integer offset(s) from the centre of the footprint along
                leike_axis (0 = centre slice, the midplane for 'full'; pass
                np.arange(-10, 11) for a range, None for all).  See leike_s2_avg.
    stride    : in-plane downsampling factor (voxel size becomes stride pc),
                speeding up the FFT ~stride² at the cost of sub-pixel lags.
    transform : 'log' -> S2 of log(field); 'linear' -> S2 of field itself.
    """
    if cube is None:
        cube = _load_cube(leike_h5)

    # Restrict to the requested footprint along the slice axis.
    n = cube.shape[leike_axis]
    if footprint != 'full':
        mid = n // 2
        rng = slice(mid, n) if footprint == 'above' else slice(0, mid)
        slc = [slice(None)] * 3
        slc[leike_axis] = rng
        cube = cube[tuple(slc)]

    # In-plane downsampling (leave the slice axis intact).
    if stride != 1:
        slc = [slice(None, None, stride)] * 3
        slc[leike_axis] = slice(None)
        cube = cube[tuple(slc)]

    sf = leike_s2_avg(cube, slices=slices, axis=leike_axis,
                      dx_pc=stride, log_floor=log_floor, transform=transform)
    r_pc, s2, _ = azimuthal_average_sf(sf)
    return r_pc, s2


def _s2_ylabel(transform):
    return 'S₂  [log-field²]' if transform == 'log' else 'S₂  [field²]'


def s2_to_ps(r_pc, s2, sigma2=None, n_k=64, taper_frac=0.25, n_lin=4000):
    """2D isotropic power spectrum from an azimuthally-averaged S2(r).

    Wiener-Khinchin, done in configuration space so the mask is handled by the
    S2 estimator (pair counting) rather than by an FFT window:

        C(r) = sigma2 - S2(r)/2        (covariance; sigma2 = plateau of S2 / 2)
        P(k) = 2π ∫ C(r) J0(2π k r) r dr

    sigma2     : field variance.  Default: half the mean of S2 over its outer
                 fifth (the saturation plateau) — reliable only if S2 has
                 levelled off within the measured range.  If it has not, the
                 recovered slope is biased and low-k is unreliable.
    taper_frac : cosine-taper the outer fraction of C(r) to zero to suppress
                 truncation ringing.
    n_k        : number of log-spaced k samples.
    n_lin      : C(r) is resampled onto this many points on a linear grid before
                 integrating, so the oscillatory J0 kernel is well sampled at
                 large r (a coarse log grid steepens the recovered slope ~0.4).

    Returns (k, P) with r in pc -> k in cycles/pc.  Band-limited: trust roughly
    1/r_max < k < 1/(2 r_min); low-k is sensitive to sigma2 and the taper.
    """
    good = np.isfinite(s2) & np.isfinite(r_pc)
    r = np.asarray(r_pc, float)[good]
    s2 = np.asarray(s2, float)[good]
    order = np.argsort(r)
    r, s2 = r[order], s2[order]
    if r.size < 4:
        return np.array([]), np.array([])

    if sigma2 is None:
        n_tail = max(3, r.size // 5)
        sigma2 = 0.5 * np.mean(s2[-n_tail:])
    C = sigma2 - 0.5 * s2

    if taper_frac > 0:
        r0 = r[-1] * (1 - taper_frac)
        m = r > r0
        C[m] *= 0.5 * (1 + np.cos(np.pi * (r[m] - r0) / (r[-1] - r0)))

    # resample onto a fine linear grid (incl. r=0, C(0)=sigma2) so J0 is well
    # sampled at large r; the S2 grid itself is log-spaced and too coarse there
    rl = np.linspace(0.0, r[-1], n_lin)
    Cl = np.interp(rl, np.concatenate([[0.0], r]), np.concatenate([[sigma2], C]))

    k = np.logspace(np.log10(1.0 / r[-1]), np.log10(0.5 / r[0]), n_k)
    P = np.array([2 * np.pi * np.trapz(Cl * j0(2 * np.pi * kk * rl) * rl, rl)
                  for kk in k])
    return k, P


def plot_echo_leike_s2(fullsky=None, cube=None,
                       leike_h5='leike2020/mean_std.h5',
                       clip_threshold=0.09, epoch=0, stride=1, norm=1.0,
                       footprint='full', slices=0, leike_axis=2,
                       log_floor=1e-3, transform='log', ax=None,
                       echo_label='echo', leike_label='Leike'):
    """Overlay one light-echo and one Leike+2020 S2(r) curve.

    fullsky  : data dict from read_fullmap(); loaded (per `stride`) if None.
    cube     : raw Leike extinction cube (nx,ny,nz); loaded from leike_h5 if
               None.  Must NOT be mean-subtracted.
    transform: 'log' -> S2 of log(field); 'linear' -> S2 of field itself.
    norm     : echo field normalization (scales linear echo S2 by norm²).
    See echo_s2_curve / leike_s2_curve for the meaning of the remaining
    parameters.  plot_echo_leike_sensitivity sweeps any one of them.
    """
    if ax is None:
        _, ax = plt.subplots()

    r_echo, s2_echo = echo_s2_curve(fullsky=fullsky, clip_threshold=clip_threshold,
                                    epoch=epoch, stride=stride,
                                    transform=transform, norm=norm)
    r_leike, s2_leike = leike_s2_curve(cube=cube, leike_h5=leike_h5,
                                       footprint=footprint, leike_axis=leike_axis,
                                       slices=slices, log_floor=log_floor,
                                       stride=stride, transform=transform)
    ax.loglog(r_echo  * LY_PER_PC, s2_echo,  label=echo_label)
    ax.loglog(r_leike * LY_PER_PC, s2_leike, label=leike_label)
    ax.set_xlabel('r  [ly]')
    ax.set_ylabel(_s2_ylabel(transform))
    ax.legend()
    return ax


# Choices varied by plot_echo_leike_sensitivity, one at a time (the rest held
# at nominal).  ECHO_PARAMS reshape the echo curve; LEIKE_PARAMS the Leike one.
DEFAULT_SWEEPS = {
    'clip_threshold': [0.06, 0.09, 0.12, 0.15],   # ~2-5x the ~0.03 noise
    'epoch':          [0, 2, 4],
    'log_floor':      [0.3e-3, 1e-3, 3e-3],
    'footprint':      ['full', 'above', 'below'],
}
ECHO_PARAMS  = ('clip_threshold', 'epoch')
LEIKE_PARAMS = ('log_floor', 'footprint')


def plot_echo_leike_sensitivity(ax=None, stride=2,
                                leike_h5='leike2020/mean_std.h5',
                                clip_threshold=0.09, epoch=0, norm=1.0,
                                footprint='full', slices=0, leike_axis=2,
                                log_floor=1e-3, transform='log', domain='sf'):
    """Overlay echo vs Leike, each as a spread of sensitivity lines.

    Every choice in DEFAULT_SWEEPS is varied one at a time (the others held at
    the nominal values passed here), so the single echo and Leike curves of
    plot_echo_leike_s2 each become a cloud showing how robust they are.  Each
    nominal curve is drawn bold; its variations thin and faded, in the same
    colour.  Data are loaded once and reused across all variations.

    transform : 'log' -> S2 of log(field) [fill-factor sensitive]; 'linear' ->
                S2 of the clipped field itself [intensity/A(V) sensitive].  The
                clip / floor are applied first in both cases.
    norm      : echo field normalization applied after clipping.  Meaningful for
                'linear' (scales echo S2 by norm² — the intensity/A(V) match to
                Leike); a no-op under 'log'.
    domain    : 'sf' plots S2(r) vs r [ly]; 'ps' plots the power spectrum P(k)
                derived from each S2 via s2_to_ps (Wiener-Khinchin), vs k [1/pc].
                A clean power-law Leike cloud in 'ps' validates the transform (it
                recovers the 2D-slice slope β-1 ≈ 1.82, not the 3D β ≈ 2.82).
    stride    : in-plane downsampling for both sides (default 2 for speed).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    nominal = dict(clip_threshold=clip_threshold, epoch=epoch,
                   log_floor=log_floor, footprint=footprint)
    faint = dict(lw=1, alpha=0.7)

    def _draw(r_pc, s2, **kw):
        if domain == 'ps':
            x, y = s2_to_ps(r_pc, s2)
        else:
            x, y = r_pc * LY_PER_PC, s2
        ax.loglog(x, y, **kw)

    # ---- echo cloud ----  (nominal C0, variations cycle C1, C2, ...)
    fullsky = read_fullmap(stride=stride)
    sf_nom = compute_s2_echo(fullsky, clip_threshold=clip_threshold,  # nom + epoch
                             transform=transform, norm=norm)
    r, s2, _ = azimuthal_average_sf(sf_nom, pair=epoch)
    _draw(r, s2, color='C0', lw=2.5, label='echo (nominal)')
    ci = 1
    for param in ECHO_PARAMS:
        for v in DEFAULT_SWEEPS[param]:
            if v == nominal[param]:
                continue
            if param == 'epoch':
                r, s2 = echo_s2_curve(sf=sf_nom, epoch=v)
            else:                                           # clip_threshold
                r, s2 = echo_s2_curve(fullsky=fullsky, clip_threshold=v,
                                      epoch=epoch, transform=transform, norm=norm)
            _draw(r, s2, color=f'C{ci}', label=f'{param}={v}', **faint)
            ci += 1

    # ---- Leike cloud ----  (nominal C0, variations cycle C1, C2, ...)
    cube = _load_cube(leike_h5)

    def _leike(**over):
        opts = dict(footprint=footprint, leike_axis=leike_axis, slices=slices,
                    log_floor=log_floor, stride=stride, transform=transform)
        opts.update(over)
        return leike_s2_curve(cube=cube, **opts)

    r, s2 = _leike()
    _draw(r, s2, color='C0', lw=2.5, label='Leike (nominal)')
    ci = 1
    for param in LEIKE_PARAMS:
        for v in DEFAULT_SWEEPS[param]:
            if v == nominal[param]:
                continue
            r, s2 = _leike(**{param: v})
            _draw(r, s2, color=f'C{ci}', label=f'{param}={v}', **faint)
            ci += 1

    if domain == 'ps':
        ax.set_xlabel('k  [1/pc]')
        ax.set_ylabel('P(k)  (from S₂)')
    else:
        ax.set_xlabel('r  [ly]')
        ax.set_ylabel(_s2_ylabel(transform))
    ax.set_title(f'Echo vs Leike — sensitivity ({transform}, {domain}, stride={stride})')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.2)
    return ax


# ---------------------------------------------------------------------------
# Leike+2020 3D power-spectrum reproduction (formerly leike.py)
# ---------------------------------------------------------------------------

def to_field(cube, field_kind='linear', rho_floor=1e-12):
    """Map the raw extinction cube to the field whose spectrum we measure.

    'linear' -> rho itself (Leike Fig. 4); 'log' -> log rho (Fig. 13).
    """
    if field_kind == 'log':
        return np.log(np.clip(cube, rho_floor, None))
    return np.asarray(cube, dtype=np.float64)


def octant_slices(shape):
    """Index tuples for the 8 octants, in Leike's bit convention.

    octant i = 4*b2 + 2*b1 + b0, b_j in {0,1}; b_j = 0 selects the positive
    half of axis j.  Octant 3 = (-x, -y, +z) is the dust-poor one that sits
    well below the rest in Fig. 4.
    """
    nx, ny, nz = shape
    hx, hy, hz = nx // 2, ny // 2, nz // 2
    pos = [slice(hx, nx), slice(hy, ny), slice(hz, nz)]
    neg = [slice(0, hx),  slice(0, hy),  slice(0, hz)]
    out = {}
    for i in range(8):
        b0, b1, b2 = i & 1, (i >> 1) & 1, (i >> 2) & 1
        out[i] = (neg[0] if b0 else pos[0],
                  neg[1] if b1 else pos[1],
                  neg[2] if b2 else pos[2])
    return out


def reproduce_leike_fig4(leike_h5='leike2020/mean_std.h5', field_kind='linear',
                         n_bins=30, octants=True, fit_pc=(2.3, 125),
                         out_png='leike2020_fig4.png', ax=None):
    """Reproduce Leike+2020 Fig. 4: the 3D power spectrum of the dust cube.

    Splits the released (740, 740, 540) mean cube at its centre into eight
    octants and plots each octant's spectrum (octant 3 highlighted), then
    prints the full-volume slope as an end-to-end sanity check against the
    published beta (2.52 linear / 2.82 log).

    field_kind : 'linear' (Fig. 4) or 'log' (Fig. 13)
    octants    : if False, only the full-volume curve is computed/plotted
    fit_pc     : (lo, hi) scale range [pc] for the slope fit
    """
    import h5py
    with h5py.File(leike_h5, 'r') as f:
        cube = f['mean'][:]
    print(f"Loaded cube {cube.shape}, "
          f"range [{np.nanmin(cube):.3e}, {np.nanmax(cube):.3e}] e-folds/pc")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))

    if octants:
        sl = octant_slices(cube.shape)
        for i in range(8):
            k, P = power_3d(to_field(cube[sl[i]], field_kind), n_bins=n_bins)
            kw = (dict(lw=2.4, color='crimson', zorder=5) if i == 3
                  else dict(lw=1.3, alpha=0.85))
            ax.loglog(k, P, label=f'octant {i}', **kw)

    kf, Pf = power_3d(to_field(cube, field_kind), n_bins=n_bins)
    lam = 1.0 / kf
    fit = (lam > fit_pc[0]) & (lam < fit_pc[1]) & (Pf > 0)
    slope = -np.polyfit(np.log(kf[fit]), np.log(Pf[fit]), 1)[0]
    paper = '2.82' if field_kind == 'log' else '2.52'
    print(f"Full-volume {field_kind} slope ({fit_pc[0]}-{fit_pc[1]} pc): "
          f"{slope:.2f}  (paper: {paper})")
    ax.loglog(kf, Pf, 'k--', lw=1.0, label=f'full volume (β={slope:.2f})')

    ax.set_xlabel(r'$k\ \mathrm{[1/pc]}$')
    ylab = 'logarithmic ' if field_kind == 'log' else ''
    ax.set_ylabel(rf'$P(k)$ of {ylab}extinction density')
    ax.set_title('Leike+2020 Fig. 4' +
                 ('  (log -> Fig. 13)' if field_kind == 'log' else ''))
    ax.legend(ncol=2, fontsize=9, frameon=False)
    sec = ax.secondary_xaxis('top',
        functions=(lambda x: 1.0 / np.where(x > 0, x, np.nan),
                   lambda L: 1.0 / np.where(L > 0, L, np.nan)))
    sec.set_xlabel('scale [pc]')
    if out_png is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(out_png, dpi=150)
        print(f'wrote {out_png}')
    return ax


if __name__ == '__main__':
    reproduce_leike_fig4()
