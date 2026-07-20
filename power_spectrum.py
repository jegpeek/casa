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

import os
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
N_H_PER_AV    = 1.9e21             # N(H)/A_V [cm^-2 mag^-1]; Bohlin, Savage &
                                   # Drake 1978 (N_H/E(B-V)=5.8e21, R_V=3.1)
PC_CM         = 3.0857e18          # cm per parsec

# Shared dust unit: the Leike and Edenhofer readers both return A_V mag/pc so
# downstream code needn't juggle e-folds vs ZGR23 E.  Leike's native Gaia-G
# e-fold is LEIKE_EFOLD_TO_AV mag of A_V (= av_per_pc(1)); Edenhofer's native
# ZGR23 E is ZGR23_AV_PER_E mag (defined by its map, near EDENHOFER_HEALPIX).
LEIKE_EFOLD_TO_AV = MAG_PER_EFOLD / A_G_OVER_A_V   # e-folds/pc -> A_V mag/pc ≈ 1.376


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


def nh_per_cm3(extinction_density):
    """Hydrogen number density n(H) [cm^-3] from a Leike G-band extinction
    density (e-folds/pc).

    Chains av_per_pc with the gas-to-dust column ratio:
        n(H) = av_per_pc(density) * N_H_PER_AV / PC_CM
    The A_V[mag/pc] -> n(H)[cm^-3] factor is N_H_PER_AV / PC_CM ≈ 6.2e2
    (i.e. n(H)=1 cm^-3 corresponds to ~1.6e-3 mag/pc).
    """
    return av_per_pc(extinction_density) * N_H_PER_AV / PC_CM


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
    d['flux_epochs'] = _echo_field(data['flux_epochs'], clip_threshold, norm,
                                   transform)
    return compute_s2(d, arcsinh_scale=None, background=0,
                      subtract_mean='global', assume_stationary=True)


def _echo_field(flux, clip_threshold, norm, transform):
    """Echo flux -> measured field: clip at clip_threshold, scale by norm, then
    log or linear.  Shared by compute_s2_echo (SF route) and echo_ps_curve
    (direct PS).  NaNs pass through (clip/log preserve them), so isfinite marks
    the footprint mask; the clip therefore never influences the mask."""
    field = norm * np.clip(flux, clip_threshold, np.inf)
    return np.log(field) if transform == 'log' else field


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


def _leike_slice_field(slab, log_floor, transform):
    """Density slice -> measured field: floor at log_floor (in the cube's units,
    A_V mag/pc from the readers by default), then log or linear.  Shared by
    leike_s2_avg (SF route) and leike_ps_curve (direct PS).  NaNs (e.g.
    Edenhofer's central hole) pass through as the footprint mask; the floor never
    influences the mask."""
    field = np.maximum(slab, log_floor)
    return np.log(field) if transform == 'log' else field


def leike_s2_avg(cube, slices=0, axis=2, dx_pc=1.0, log_floor=3e-3,
                 transform='log'):
    """Average 2D S2 over one or more slices of the Leike extinction cube.

    cube     : raw (nx, ny, nz) extinction-density array (A_V mag/pc from the
               readers by default; unit-agnostic) — not logged or mean-subtracted
    slices   : integer offset(s) from the centre slice of `cube` along `axis`.
               0 (default) is the centre — the midplane for a full cube; pass an
               array such as np.arange(-10, 11) to average a range.  None
               averages every slice.  Offsets outside the cube are dropped.
    axis     : 0, 1, or 2 — axis to slice along
    dx_pc    : voxel size in pc
    log_floor: lower clip on the density (cube units, A_V mag/pc by default;
               applied before the log if any)
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
        d = {
            'flux_epochs': _leike_slice_field(slab[np.newaxis], log_floor,
                                              transform),
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


def _load_cube(leike_h5, units='av'):
    """Load the Leike+2020 mean extinction cube (nx, ny, nz) — the Leike reader
    for the echo-vs-dust comparison, in the shared A_V/pc unit by default.

    units : 'av' (default) -> A_V mag/pc (native e-folds * LEIKE_EFOLD_TO_AV
            ≈ 1.376); 'raw' -> native Gaia-G e-folds/pc.
    """
    import h5py
    with h5py.File(leike_h5, 'r') as f:
        cube = f['mean'][:]
    if units == 'av':
        cube = cube * np.float32(LEIKE_EFOLD_TO_AV)
    return cube


def _existing_path(path):
    """Return `path`, or its `?download=1` download-artifact sibling if that is
    what is actually on disk (Zenodo wget leaves the query string in the name).
    Falls back to `path` unchanged; callers guard on existence."""
    if os.path.exists(path):
        return path
    alt = path + '?download=1'
    return alt if os.path.exists(alt) else path


def _load_leike_sample(sample_h5, index=0, units='av'):
    """Load one Leike+2020 posterior sample cube (nx, ny, nz) from samples.h5
    (dataset 'dust_samples').  units as _load_cube: 'av' (default) -> A_V mag/pc,
    'raw' -> native e-folds/pc."""
    import h5py
    with h5py.File(_existing_path(sample_h5), 'r') as f:
        cube = f['dust_samples'][index]
    if units == 'av':
        cube = cube * np.float32(LEIKE_EFOLD_TO_AV)
    return cube


def _load_leike(leike_h5='leike2020/mean_std.h5', use_sample=0,
                samples_h5='leike2020/samples.h5', units='av'):
    """Leike cube selector: the posterior mean (use_sample='mean') or posterior
    sample `use_sample` (int index, default 0).  A sample carries the inferred
    small-scale power that the Wiener-smoothed mean suppresses, so its power-
    spectrum slope is close to Leike's published beta while the mean's is biased
    steep (~0.3-0.6) — prefer a sample for slope work."""
    if use_sample == 'mean':
        return _load_cube(leike_h5, units=units)
    return _load_leike_sample(samples_h5, int(use_sample), units=units)


# ---------------------------------------------------------------------------
# Edenhofer+2023 HEALPix map -> Cartesian cube (Leike-compatible)
# ---------------------------------------------------------------------------

EDENHOFER_HEALPIX = 'edenhofer23/mean_and_std_healpix.fits'
# Posterior samples of the same reconstruction: 'SAMPLES'[i] is one draw with
# the exact shell/pixel layout of the mean file's 'MEAN' HDU, so it is a drop-in
# field for the regrid.  Analogue of Leike's leike2020/samples.h5.
EDENHOFER_SAMPLES = 'edenhofer23/samples_healpix.fits'

# Edenhofer+2023 report differential extinction in the ZGR23 (Zhang, Green &
# Rix 2023) unit E, a *reddening-like* quantity (a close relative of E(B-V))
# with A_V = 2.8 E (ZGR23's published extinction curve; dustmaps.edenhofer2023
# docs).  This is the Edenhofer reader's native-E -> A_V/pc factor, the analogue
# of LEIKE_EFOLD_TO_AV.  Only linear-transform amplitude depends on it; the log
# transform / all slopes are invariant.
ZGR23_AV_PER_E = 2.8      # A_V [mag] per unit ZGR23 extinction E


def _edenhofer_cache_path(healpix_path, method='interp', sample=None):
    """Default .npy cache path for the regridded cube, alongside the source
    HEALPix file (so a prebuilt cube next to the real data is picked up).  The
    'mass' regrid and each posterior sample cache to their own file; the mean
    map (sample=None) keeps the bare `eden_leikebox` name."""
    suffix = '_mass' if method == 'mass' else ''
    tag = '' if sample is None else f'_sample{sample}'
    return os.path.join(os.path.dirname(healpix_path),
                        f'eden_leikebox{suffix}{tag}.npy')


# Box aliases mirroring the Zenodo interp2box.py exporter.  The 'leike' box has
# the exact extent and (740, 740, 540) shape of the Leike+2020 cube (dx ~ 1 pc),
# so its output is a drop-in for _load_cube / leike_s2_curve.
_EDENHOFER_BOXES = {
    'leike': (((-369.5, 369.5), (-369.5, 369.5), (-269.5, 269.5)),
              (740, 740, 540)),
}


def _cart2sph(x, y, z):
    """Cartesian -> (r, lon, lat) in radians.  Vendored from interp2box.py."""
    lon = np.arctan2(y, x)
    rho2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(rho2))
    r = np.sqrt(rho2 + z**2)
    return r, lon, lat


def _hp_bilinear(m, lon_deg, lat_deg, nside, nest):
    """Bilinear (4-neighbour) HEALPix interpolation of a single-shell map `m`
    at the given lon/lat (degrees).  Vendored from interp2box.get_interp_val,
    which reaches into healpy's internal _get_interpol for the neighbour
    pixels + weights.
    """
    from healpy.pixelfunc import lonlat2thetaphi
    from healpy import _healpy_pixel_lib as pixlib
    theta, phi = lonlat2thetaphi(lon_deg, lat_deg)
    fn = pixlib._get_interpol_nest if nest else pixlib._get_interpol_ring
    res = fn(nside, theta, phi)
    p = np.array(res[0:4])          # (4, npts) neighbour pixel indices
    w = np.array(res[4:8])          # (4, npts) neighbour weights
    return np.sum(m[p] * w, axis=0)


def _edenhofer_field(hdul, sample=None):
    """(nshell, npix) differential-extinction field from an open Edenhofer
    HEALPix file, plus its NSIDE and NEST flag.  sample=None reads the posterior
    mean ('MEAN', mean_and_std_healpix.fits); an int reads that posterior sample
    ('SAMPLES'[sample], samples_healpix.fits).  Samples share the mean's shell/
    pixel layout, so either is a drop-in field for the regrid."""
    hdu = hdul['MEAN'] if sample is None else hdul['SAMPLES']
    nside = int(hdu.header['NSIDE'])
    nest = hdu.header['ORDERING'].lower().startswith('nest')
    data = hdu.data if sample is None else hdu.data[sample]
    return np.asarray(data, dtype=np.float64), nside, nest


def _edenhofer_interp_build(healpix_path, extent, shp, z_chunk, verbose,
                            sample=None):
    """interp2box-style build: sample the log-interpolated density field at each
    Cartesian voxel centre (bilinear angular + linear radial, in log space).
    Returns raw ZGR23 density with NaN in the central hole.  Point-sampling —
    not mass conserving, and the log step biases peaks/mass down (Jensen)."""
    from astropy.io import fits
    nx, ny, nz = shp
    (x0, x1), (y0, y1), (z0, z1) = extent
    with fits.open(healpix_path, 'readonly') as hdul:
        field, nside, nest = _edenhofer_field(hdul, sample)     # (nshell, npix)
        logmap = np.log(field)
        radii = np.asarray(
            hdul['RADIAL PIXEL CENTERS'].data['radial pixel centers'],
            dtype=np.float64)

    x = np.linspace(x0, x1, nx, dtype=np.float32)
    y = np.linspace(y0, y1, ny, dtype=np.float32)
    z = np.linspace(z0, z1, nz, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')          # (nx, ny)
    cube = np.full((nx, ny, nz), np.nan, dtype=np.float32)

    for zs in range(0, nz, z_chunk):
        ze = min(zs + z_chunk, nz)
        if verbose:
            print(f'edenhofer interp z {ze:4d}/{nz}', flush=True)
        r, lon, lat = _cart2sph(X[:, :, None], Y[:, :, None],
                                z[None, None, zs:ze])   # each (nx, ny, nzc)
        lon = np.broadcast_to(lon, r.shape)   # lon has no z-dependence
        out = np.full(r.shape, np.nan, dtype=np.float64).ravel()
        rflat = r.ravel()
        lon_d = np.degrees(lon).ravel()
        lat_d = np.degrees(lat).ravel()

        # Bin each voxel into the shell interval [radii[i], radii[i+1]); drop
        # voxels inside/outside the shell stack, then group by shell so each
        # HEALPix interpolation runs once per shell over its members.
        idx = np.searchsorted(radii, rflat, side='right') - 1
        vv = np.where((idx >= 0) & (idx < radii.size - 1))[0]
        order = vv[np.argsort(idx[vv], kind='stable')]
        gidx = idx[order]
        uniq, starts = np.unique(gidx, return_index=True)
        bnds = np.append(starts, gidx.size)
        for gi, i_l in enumerate(uniq):
            sl = order[bnds[gi]:bnds[gi + 1]]
            lo = _hp_bilinear(logmap[i_l],     lon_d[sl], lat_d[sl], nside, nest)
            hi = _hp_bilinear(logmap[i_l + 1], lon_d[sl], lat_d[sl], nside, nest)
            wr = (rflat[sl] - radii[i_l]) / (radii[i_l + 1] - radii[i_l])
            out[sl] = (1.0 - wr) * lo + wr * hi
        cube[:, :, zs:ze] = np.exp(out).reshape(r.shape).astype(np.float32)
    return cube


def _edenhofer_mass_build(healpix_path, extent, shp, max_nside, verbose,
                          sample=None):
    """Mass-conserving scatter build: deposit each HEALPix voxel's extinction
    'mass' (density * volume) into the Cartesian voxel holding its centre, then
    divide by the voxel volume -> volume-weighted mean density, conserving
    total density*volume inside the box.  Coarse cells are subdivided so every
    voxel is reached: radially into <= dx/2 virtual sub-shells (each carrying
    its exact sub-volume's mass) and angularly by ud_grade to the nside whose
    pixel is <= dx/2 across (capped at `max_nside`).  No geometric voxel overlap
    is computed.  Linear throughout, so no log/Jensen bias.  Returns raw ZGR23
    density; voxels no ray reaches (the central hole, and far corners beyond
    max_nside) are NaN."""
    import healpy as hp
    from astropy.io import fits
    with fits.open(healpix_path, 'readonly') as hdul:
        data, nside0, nest = _edenhofer_field(hdul, sample)    # (nshell, npix0)
        if not nest:
            raise ValueError('mass build expects NESTED HEALPix ordering')
        bounds = np.asarray(
            hdul['RADIAL PIXEL BOUNDARIES'].data['radial pixel boundaries'],
            dtype=np.float64)

    (x0, x1), (y0, y1), (z0, z1) = extent
    nx, ny, nz = shp
    dx, dy, dz = (x1 - x0) / (nx - 1), (y1 - y0) / (ny - 1), (z1 - z0) / (nz - 1)
    half = 0.5 * min(dx, dy, dz)
    xa, ya, za = max(abs(x0), abs(x1)), max(abs(y0), abs(y1)), max(abs(z0), abs(z1))
    r_corner = np.sqrt(xa**2 + ya**2 + za**2)
    mass = np.zeros((nx, ny, nz), dtype=np.float64)

    def _pow2_ceil(v):
        return 1 << int(np.ceil(np.log2(max(v, 1.0))))

    unitvec = {}                        # nside -> (3, npix) pixel unit vectors
    n_shell = data.shape[0]
    for i in range(n_shell):
        r_lo, r_hi = bounds[i], bounds[i + 1]
        if r_lo > r_corner:             # shell entirely outside the box
            continue
        if verbose and i % 40 == 0:
            print(f'edenhofer mass regrid shell {i:3d}/{n_shell}', flush=True)
        nside = min(int(max_nside), max(nside0, _pow2_ceil(2.05 * r_hi / half)))
        if nside not in unitvec:
            unitvec[nside] = np.asarray(
                hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True))
        ux, uy, uz = unitvec[nside]
        dens = (data[i] if nside == nside0 else
                hp.ud_grade(data[i], nside, order_in='NESTED', order_out='NESTED'))
        # keep only pixels that can land in the box (loosest bound at r_lo)
        cand = ((np.abs(ux) <= xa / r_lo) & (np.abs(uy) <= ya / r_lo) &
                (np.abs(uz) <= za / r_lo))
        uxs, uys, uzs, denss = ux[cand], uy[cand], uz[cand], dens[cand]
        omega = 4.0 * np.pi / dens.size
        n_r = max(1, int(np.ceil((r_hi - r_lo) / half)))
        redges = np.linspace(r_lo, r_hi, n_r + 1)
        for j in range(n_r):
            ra, rb = redges[j], redges[j + 1]
            rc = 0.5 * (ra + rb)
            m = denss * (omega * (rb**3 - ra**3) / 3.0)    # per-pixel sub-mass
            ix = np.round((rc * uxs - x0) / dx).astype(np.intp)
            iy = np.round((rc * uys - y0) / dy).astype(np.intp)
            iz = np.round((rc * uzs - z0) / dz).astype(np.intp)
            ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) &
                  (iz >= 0) & (iz < nz))
            np.add.at(mass, (ix[ok], iy[ok], iz[ok]), m[ok])

    dens_grid = (mass / (dx * dy * dz)).astype(np.float32)
    dens_grid[mass == 0.0] = np.nan     # central hole + unreached corners -> NaN
    return dens_grid


def load_edenhofer_cube(healpix_path=EDENHOFER_HEALPIX, box='leike',
                        shape=None, rebuild=False, units='av', method='interp',
                        fill_hole=1e-5, mass_max_nside=1024, z_chunk=27,
                        sample=None, samples_path=EDENHOFER_SAMPLES,
                        verbose=True):
    """Interpolate the Edenhofer+2023 HEALPix dust map onto a Cartesian cube
    matching the Leike `_load_cube` convention.

    Returns an (nx, ny, nz) float32 array of dust extinction density with x/y in
    the plane of the sky (centred on the Sun) and z the third axis; for the
    default 'leike' box the extent and (740, 740, 540) shape equal the
    Leike+2020 cube (voxel pitch ~1 pc), so the result is a drop-in for
    `_load_cube` / `leike_s2_curve` / `plot_echo_leike_sensitivity`.

    method    : 'interp' (default) reproduces the Zenodo interp2box.py export —
              per Cartesian voxel, bilinear-angular + linear-radial interpolation
              in log space.  It point-samples the density field: not mass
              conserving, and the log step biases peaks/small-scale power down.
              'mass' instead scatters each HEALPix voxel's extinction*volume into
              the Cartesian voxel holding its centre and divides by the voxel
              volume, conserving total extinction*volume (linear, no log bias).
              Coarse cells are subdivided (radial virtual sub-shells + angular
              ud_grade, both to <= half a voxel) so no gaps open where HEALPix is
              coarser than the grid; `mass_max_nside` caps the angular upsampling
              (far corners beyond it may keep small gaps -> NaN -> `fill_hole`).
              The two methods cache to separate files.

    Voxels inside the innermost shell (r < ~68.8 pc — the central hole of the
    reconstruction, coincident with the Local Bubble) are interpolation-empty.
    By default they are filled with `fill_hole` rather than left NaN: the region
    is genuinely low-density (few tracer stars, not real structure), so treating
    it as diffuse floor is more faithful than masking it — and it avoids the hole
    imprinting a spurious knee in the direct power spectrum (its mask scale,
    ~1/68.8 pc ≈ 0.015 cyc/pc, otherwise shows up as a break).

    box     : 'leike' alias, or an ((x0,x1),(y0,y1),(z0,z1), (nx,ny,nz)) pair
              of (extent, shape).
    shape   : override the alias voxel shape (extent unchanged).
    fill_hole : value written into the central hole (and any other non-finite
              voxels), in the returned `units`; below a typical log_floor it just
              reads as the diffuse floor.  None leaves them NaN (mask behaviour).
    units   : 'av' (default) -> A_V mag/pc (native ZGR23 E * ZGR23_AV_PER_E = 2.8),
              the shared dust unit that matches the Leike reader; 'raw' returns
              native ZGR23 'E'/pc.  Only linear-transform amplitude depends on
              this; the log transform / all slopes are invariant.
    sample  : None (default) regrids the posterior mean from `healpix_path`; an
              int regrids that posterior sample from `samples_path` instead (its
              slope carries the inferred power the Wiener-mean smooths away —
              prefer a sample for slope work, as with Leike).  Each sample caches
              to its own .npy (see `cache`), parallel to the mean cube.
    samples_path : posterior-sample HEALPix file read when `sample` is not None
              (default EDENHOFER_SAMPLES).
    rebuild : force a fresh regrid that overwrites the .npy cache instead of
              loading it when present.  Run this after changing the cube
              construction (the loader can't tell a cache is stale); normal
              calls leave it False and stay cache-fast.  The cache sits next to
              the source file — `_edenhofer_cache_path` gives distinct files for
              the mean, each sample, and the 'mass' regrid — and always holds
              raw ZGR23 values; the `units` rescale is applied on return.
    z_chunk : z-planes interpolated per batch (caps peak memory).
    """
    src_path = healpix_path if sample is None else _existing_path(samples_path)
    cache = _edenhofer_cache_path(src_path, method, sample)
    scale = np.float32(ZGR23_AV_PER_E) if units == 'av' else None

    def _finalize(cube):
        """Apply the units rescale and (unless None) fill the hole."""
        if scale is not None:
            cube *= scale
        if fill_hole is not None:
            cube[~np.isfinite(cube)] = np.float32(fill_hole)
        return cube

    if not rebuild and os.path.exists(cache):
        return _finalize(np.load(cache))

    extent, shp = _EDENHOFER_BOXES[box] if isinstance(box, str) else box
    if shape is not None:
        shp = shape
    if method == 'mass':
        cube = _edenhofer_mass_build(src_path, extent, shp, mass_max_nside,
                                     verbose, sample=sample)
    elif method == 'interp':
        cube = _edenhofer_interp_build(src_path, extent, shp, z_chunk, verbose,
                                       sample=sample)
    else:
        raise ValueError("method must be 'interp' or 'mass'")

    np.save(cache, cube)              # cache holds raw ZGR23 values (NaN hole)
    return _finalize(cube)


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


def echo_ps_curve(fullsky=None, epoch=0, clip_threshold=0.09, stride=1,
                  data_dir='data', transform='log', norm=1.0, noise_floor=0.0):
    """Direct 2D power spectrum of the light echo.  Returns (k, P), k in
    cycles/pc — the field->PS analogue of echo_s2_curve, via
    image_power_spectrum (masked periodogram) instead of the SF + s2_to_ps.

    The mask is the finite footprint of the epoch image; clip/transform set the
    field values only.  The pixel scale is read from the (possibly strided) U
    grid, so stride is handled automatically.
    """
    if fullsky is None:
        fullsky = read_fullmap(data_dir=data_dir, stride=stride)
    flux = fullsky['flux_epochs'][epoch]
    field = _echo_field(flux, clip_threshold, norm, transform)
    U = fullsky['U_grid']
    pixel_ly = float(abs(U[0, 1] - U[0, 0]))
    return image_power_spectrum(field, pixel_ly=pixel_ly, noise_floor=noise_floor)


def leike_s2_curve(cube=None, leike_h5='leike2020/mean_std.h5',
                   footprint='full', leike_axis=2, slices=0,
                   log_floor=3e-3, stride=1, transform='log'):
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


def leike_ps_curve(cube=None, leike_h5='leike2020/mean_std.h5',
                   footprint='full', leike_axis=2, slices=0,
                   log_floor=3e-3, stride=1, transform='log', noise_floor=0.0):
    """Direct 2D power spectrum of the Leike/Edenhofer cube, averaged over the
    selected slices.  Returns (k, P), k in cycles/pc — the field->PS analogue of
    leike_s2_curve.  Every slice shares the same shape and voxel size, hence the
    same k grid, so P(k) is just the per-bin mean over slices.

    footprint/leike_axis/slices/stride/log_floor/transform: as leike_s2_curve.
    The mask is each slice's finite pixels (Edenhofer's central hole drops out);
    the pixel scale is `stride` pc, matching the SF route's dx_pc.
    """
    if cube is None:
        cube = _load_cube(leike_h5)

    # footprint restriction + in-plane downsampling (same as leike_s2_curve)
    n = cube.shape[leike_axis]
    if footprint != 'full':
        mid = n // 2
        rng = slice(mid, n) if footprint == 'above' else slice(0, mid)
        slc = [slice(None)] * 3
        slc[leike_axis] = rng
        cube = cube[tuple(slc)]
    if stride != 1:
        slc = [slice(None, None, stride)] * 3
        slc[leike_axis] = slice(None)
        cube = cube[tuple(slc)]

    n_total = cube.shape[leike_axis]
    if slices is None:
        indices = np.arange(n_total)
    else:
        indices = n_total // 2 + np.atleast_1d(np.asarray(slices, dtype=int))
        indices = indices[(indices >= 0) & (indices < n_total)]
    if indices.size == 0:
        raise ValueError('slices select no planes within the cube')

    k_ref, P_sum = None, None
    for idx in indices:
        slc = [slice(None)] * 3
        slc[leike_axis] = int(idx)
        slab = np.asarray(cube[tuple(slc)], dtype=np.float64)   # 2D
        field = _leike_slice_field(slab, log_floor, transform)
        k, P = image_power_spectrum(field, pixel_pc=stride, noise_floor=noise_floor)
        if P_sum is None:
            k_ref, P_sum = k, P.copy()
        else:
            P_sum += P
    return k_ref, P_sum / indices.size


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


WEBB_PIXEL_LY = 0.0016   # read_fullmap resampled_epochs pixel scale [ly]


def _apodize_mask(valid, apod_pix):
    """Cosine-ramp a boolean valid-pixel mask from 0 at its boundary up to 1 a
    distance `apod_pix` pixels inside, to suppress edge leakage.  apod_pix<=0
    returns the bare 0/1 mask."""
    valid = np.asarray(valid, dtype=bool)
    if apod_pix <= 0:
        return valid.astype(float)
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(valid)
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(d / apod_pix, 0.0, 1.0))


def image_power_spectrum(image, pixel_pc=None, pixel_ly=WEBB_PIXEL_LY,
                         n_bins=100, taper_frac=0.05, detrend='mean',
                         noise_floor=0.0, mask=None, mask_apod_pix=50):
    """Azimuthally-averaged 2D power spectrum of an echo image (masked or square).

    A direct-FFT companion to s2_to_ps: matches Kim+2008's estimator (their 5%
    cosine taper) so the slope is comparable to their beta, and returns the same
    (k, P) form as s2_to_ps so the two overlay directly (k in cycles/pc =
    1/scale[pc]).  Because P(k) and S2(r) are a Fourier pair, this adds no
    physics over the SF route — its value is a like-for-like Kim comparison and
    a cross-check that the SF->PS chain (window handling included) is right.

    Recipe: masked detrend -> window (outer taper * apodized footprint) -> FFT
    -> |.|^2 / sum(w^2) -> ring-average (mean power per mode = per-mode P_2D,
    slope beta_2D) -> subtract a white noise floor.

    image      : 2D float array.  A fully-populated square, or the whole
                 footprint with NaNs (no need to fill them — see `mask`).  Pass
                 whatever field you want: linear flux to match Kim, or the SF's
                 log/clipped field to match s2_to_ps.
    pixel_pc   : pixel size [pc].  If None, derived from pixel_ly (default is the
                 read_fullmap resampled scale, 0.0016 ly ~ 4.9e-4 pc).
    taper_frac : fraction of each outer edge given a raised-cosine (Tukey) taper;
                 Ingalls+2004 used 0.05.  0 disables it.
    detrend    : 'mean' subtracts the (masked) mean.  Without windowing the mean
                 only sets k=0, but the window's sidelobes leak a nonzero DC into
                 low k, so remove it first.  'plane' also removes a fitted
                 gradient; None leaves the field.
    mask       : optional 2D bool/0-1 array of *valid* pixels.  If None it is
                 taken from the finite pixels of `image`, so you can pass the raw
                 masked footprint instead of median-filling.  The (boundary-
                 apodized) mask is folded into the FFT window and into the
                 sum(w^2) normalization, which corrects the ~f_sky amplitude bias
                 and keeps the footprint edge from ringing.  Use the SAME mask
                 for the log and linear fields so their windowing is identical
                 (log pins the level via Leike continuity; linear then carries
                 the intensity factor).
    mask_apod_pix : cosine-taper width [pixels] inward from the mask boundary
                 (0 disables the boundary apodization).
    noise_floor: white-noise power subtracted from every P(k) (flat in k).  For
                 per-pixel noise sigma on empty sky the floor is ~sigma^2 *
                 pixel_pc^2 in these units; where P dips below it the result goes
                 negative and simply drops off a log plot.

    Returns (k_pc, P): k in cycles/pc, P ring-averaged power [field^2 pc^2] in
    the Welch/periodogram convention |FFT(f*w)|^2 * pixel_pc^2 / sum(w^2).
    """
    im = np.asarray(image, dtype=np.float64)
    if im.ndim != 2:
        raise ValueError('image must be 2D')
    if pixel_pc is None:
        pixel_pc = pixel_ly / LY_PER_PC
    ny, nx = im.shape

    valid = np.isfinite(im)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError('mask leaves no valid pixels')

    # detrend on valid pixels only, then zero the rest (they get window w=0)
    im = im.copy()
    if detrend == 'mean':
        im -= im[valid].mean()
    elif detrend == 'plane':
        yy, xx = np.mgrid[0:ny, 0:nx]
        A = np.column_stack([np.ones(int(valid.sum())),
                             xx[valid].ravel(), yy[valid].ravel()])
        coef, *_ = np.linalg.lstsq(A, im[valid], rcond=None)
        im -= coef[0] + coef[1] * xx + coef[2] * yy
    elif detrend is not None:
        raise ValueError("detrend must be 'mean', 'plane', or None")
    im[~valid] = 0.0

    # window = outer raised-cosine (Tukey) taper * boundary-apodized footprint
    if taper_frac > 0:
        from scipy.signal.windows import tukey
        w = np.outer(tukey(ny, 2 * taper_frac), tukey(nx, 2 * taper_frac))
    else:
        w = np.ones((ny, nx))
    if not valid.all():
        w = w * _apodize_mask(valid, mask_apod_pix)

    F = np.fft.fft2(im * w)
    power = (F.real**2 + F.imag**2) * pixel_pc**2 / np.sum(w**2)

    kx = np.fft.fftfreq(nx, d=pixel_pc)          # cycles/pc
    ky = np.fft.fftfreq(ny, d=pixel_pc)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')  # match power shape (ny, nx)
    kmag = np.sqrt(KX**2 + KY**2).ravel()
    power = power.ravel()

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
    return np.array(k_cen), np.array(P_k) - noise_floor


def plot_echo_leike_s2(fullsky=None, cube=None,
                       leike_h5='leike2020/mean_std.h5', use_sample=0,
                       samples_h5='leike2020/samples.h5',
                       clip_threshold=0.09, epoch=0, stride=1, norm=1.0,
                       footprint='full', slices=0, leike_axis=2,
                       log_floor=3e-3, transform='log', domain='sf', ax=None,
                       echo_label='echo', leike_label='Leike'):
    """Overlay one light-echo and one Leike+2020 curve.

    fullsky  : data dict from read_fullmap(); loaded (per `stride`) if None.
    cube     : Leike extinction cube (nx,ny,nz); if None, loaded per use_sample.
    use_sample : Leike posterior sample index (int, default 0) or 'mean'; the
               mean's slope is Wiener-steepened, so a sample is the default.
    transform: 'log' -> S2 of log(field); 'linear' -> S2 of field itself.
    norm     : echo field normalization (scales linear echo S2 by norm²).
    domain   : 'sf' -> S2(r) vs r [ly]; 'ps(sf)' -> P(k) via s2_to_ps; 'ps' ->
               P(k) direct from the field via image_power_spectrum.  The two PS
               domains share a normalization and differ only in estimator.
    See echo_s2_curve / leike_s2_curve for the meaning of the remaining
    parameters.  plot_echo_leike_sensitivity sweeps any one of them.
    """
    if ax is None:
        _, ax = plt.subplots()
    if cube is None:
        cube = _load_leike(leike_h5, use_sample, samples_h5)

    if domain == 'ps':
        x_e, y_e = echo_ps_curve(fullsky=fullsky, clip_threshold=clip_threshold,
                                 epoch=epoch, stride=stride, transform=transform,
                                 norm=norm)
        x_l, y_l = leike_ps_curve(cube=cube, leike_h5=leike_h5,
                                  footprint=footprint, leike_axis=leike_axis,
                                  slices=slices, log_floor=log_floor,
                                  stride=stride, transform=transform)
    else:
        r_e, s2_e = echo_s2_curve(fullsky=fullsky, clip_threshold=clip_threshold,
                                  epoch=epoch, stride=stride, transform=transform,
                                  norm=norm)
        r_l, s2_l = leike_s2_curve(cube=cube, leike_h5=leike_h5,
                                   footprint=footprint, leike_axis=leike_axis,
                                   slices=slices, log_floor=log_floor,
                                   stride=stride, transform=transform)
        if domain == 'ps(sf)':
            x_e, y_e = s2_to_ps(r_e, s2_e)
            x_l, y_l = s2_to_ps(r_l, s2_l)
        else:
            x_e, y_e = r_e * LY_PER_PC, s2_e
            x_l, y_l = r_l * LY_PER_PC, s2_l

    ax.loglog(x_e, y_e, label=echo_label)
    ax.loglog(x_l, y_l, label=leike_label)
    if domain in ('ps', 'ps(sf)'):
        ax.set_xlabel('k  [1/pc]')
        ax.set_ylabel('P(k)  ' + ('(from S₂)' if domain == 'ps(sf)' else '(direct)'))
    else:
        ax.set_xlabel('r  [ly]')
        ax.set_ylabel(_s2_ylabel(transform))
    ax.legend()
    return ax


# Choices varied by plot_echo_leike_sensitivity, one at a time (the rest held
# at nominal).  ECHO_PARAMS reshape the echo curve; LEIKE_PARAMS the Leike one.
DEFAULT_SWEEPS = {
    'clip_threshold': [0.06, 0.09, 0.12, 0.15],   # ~2-5x the ~0.03 noise
    'epoch':          [0, 2, 4],
    'log_floor':      [1e-3, 3e-3, 1e-2],   # A_V mmag/10pc: 10/30/100 (~n_H 0.6-6)
    'footprint':      ['full', 'above', 'below'],
}
ECHO_PARAMS  = ('clip_threshold', 'epoch')
LEIKE_PARAMS = ('log_floor', 'footprint')


def _footprint_z_pc(footprint, cube, leike_axis):
    """z-offset [pc] of the centre slice a `footprint` selects (native 1-pc
    voxels; the slice axis is never strided), for legend labels — e.g. 'above'
    of the 540-plane cube is +135 pc, 'below' -135 pc, 'full' 0."""
    n = cube.shape[leike_axis]
    mid = n // 2
    if footprint == 'above':
        c = mid + (n - mid) // 2
    elif footprint == 'below':
        c = mid // 2
    else:
        c = mid
    return c - mid


def _two_column_legend(ax, left, right, **kw):
    """Two-column legend with `left` handles in the left column and `right` in
    the right.  matplotlib fills columns as contiguous chunks, so we pad the
    shorter group with blank rows to keep the two groups in separate columns.
    """
    from matplotlib.lines import Line2D
    n = max(len(left), len(right), 1)

    def pad(g):
        return list(g) + [Line2D([], [], linestyle='none')
                          for _ in range(n - len(g))]

    handles = pad(left) + pad(right)
    labels = [' ' if h.get_label().startswith('_') else h.get_label()
              for h in handles]
    ax.legend(handles=handles, labels=labels, ncol=2, **kw)


# Leike+2020 3D power-spectrum spectral index beta (Fig. 4 linear / Fig. 13 log);
# the 2D-slice PS slope we compare against is beta - 1.
LEIKE_PS_SLOPE = {
    'linear': 1.72,   # hand-steepened; Leike 2D-slice value is 1.52 (= 2.52-1)
    'log':    1.82,   # = 2.82 - 1
}


def _ps_guide(ax, echo_line, leike_line, slope, band=100.0):
    """Draw a guide-the-eye power law (P ∝ k^-slope) plus a normalization band
    on a PS panel.

    The slope is Leike's 2D-slice value (beta-1) for the current transform,
    anchored to the Leike nominal curve at the smallest scale plotted (its
    largest k) and extrapolated across the full k range toward the echo.
    band : half-width of the shaded normalization envelope.  A 10x field-
        normalization uncertainty is 100x in the (amplitude²) power spectrum, so
        the default 100 shades guide/100 .. guide*100.
    """
    def _valid(line):
        k = np.asarray(line.get_xdata(), float)
        p = np.asarray(line.get_ydata(), float)
        m = np.isfinite(k) & np.isfinite(p) & (k > 0) & (p > 0)
        return k[m], p[m]

    ke, _ = _valid(echo_line)
    kl, pl = _valid(leike_line)
    if ke.size == 0 or kl.size == 0:
        return

    ia = np.argmax(kl)                           # Leike smallest scale = max k
    kg = np.logspace(np.log10(min(ke.min(), kl.min())),
                     np.log10(max(ke.max(), kl.max())), 200)
    guide = pl[ia] * (kg / kl[ia]) ** (-slope)
    ax.fill_between(kg, guide / band, guide * band, color='0.6', alpha=0.15,
                    lw=0, zorder=0)
    ax.loglog(kg, guide, ls='--', color='0.4', lw=1.3, zorder=1)


def plot_echo_leike_sensitivity(ax=None, stride=2,
                                leike_h5='leike2020/mean_std.h5',
                                dust_source='edenhofer',
                                edenhofer_path=EDENHOFER_HEALPIX,
                                edenhofer_sample=0,
                                leike_samples_h5='leike2020/samples.h5',
                                leike_sample=0, use_sample=0,
                                clip_threshold=0.09, epoch=0, norm=1.0,
                                fill_factor=1.0,
                                footprint='full', slices=0, leike_axis=2,
                                log_floor=3e-3, transform='log', domain='sf',
                                guide=True, guide_slope=None, band=100.0):
    """Overlay echo vs Leike, each as a spread of sensitivity lines.

    Every choice in DEFAULT_SWEEPS is varied one at a time (the others held at
    the nominal values passed here), so the single echo and Leike curves of
    plot_echo_leike_s2 each become a cloud showing how robust they are.  Each
    nominal curve is drawn bold; its variations thin and faded, in the same
    colour.  Data are loaded once and reused across all variations.

    dust_source : 'edenhofer' (default) or 'leike' — which map forms the dust
                cloud (nominal + log_floor + footprint sweep).  A single Leike
                posterior-sample line is drawn in either mode (see leike_sample).
                With 'leike', an extra Edenhofer-nominal line is also added for
                comparison.  Both dust cubes are read in the shared A_V mag/pc.
    edenhofer_sample : which Edenhofer cube is the nominal (both the cloud in
                'edenhofer' mode and the comparison line in 'leike' mode) — a
                posterior-sample index (int, default 0) or None for the mean.
                A SAMPLE is the default for the same reason as Leike's: the
                posterior mean's power-spectrum slope is Wiener-steepened.
    leike_sample : also draw one extra Leike posterior-sample line (index into
                samples.h5) at the nominal choices; None skips it (also skipped
                if it would duplicate the cloud, or samples.h5 is absent).
    use_sample : which Leike cube forms the dust_source='leike' cloud — a sample
                index (int, default 0) or 'mean'.  Default is a SAMPLE because
                the posterior mean's power-spectrum slope is Wiener-steepened
                (~0.3-0.6) vs the inferred/published beta.  No effect in
                'edenhofer' mode — the Edenhofer cloud's mean/sample choice is
                `edenhofer_sample`.

    transform : 'log' -> S2 of log(field) [fill-factor sensitive]; 'linear' ->
                S2 of the clipped field itself [intensity/A(V) sensitive].  The
                clip / floor are applied first in both cases.
    norm      : echo field normalization applied after clipping.  Meaningful for
                'linear' (scales echo S2 by norm² — the intensity/A(V) match to
                Leike); a no-op under 'log'.
    fill_factor: multiply the echo S2/P by this factor (a vertical shift of the
                echo cloud relative to Leike, in both domains and transforms).
                Its physical interpretation is still under discussion; here it is
                purely a display-time relative normalization knob.
    domain    : 'sf' plots S2(r) vs r [ly].  'ps(sf)' plots P(k) obtained from
                each S2 via s2_to_ps (Wiener-Khinchin).  'ps' plots P(k) taken
                directly from the field with image_power_spectrum (a masked
                periodogram — no SF).  Both PS domains share the same physical
                normalization (variance = ∫P d²k), so they overlay; 'ps(sf)' and
                'ps' differ only in estimator (they diverge a little at the band
                edges).  A clean power-law dust cloud recovers the 2D-slice slope
                β-1 ≈ 1.82 (log), not the 3D β ≈ 2.82.
    guide     : draw a guide-the-eye power law + normalization band (PS only;
                see _ps_guide).  Uses Leike's 2D-slice slope for the current
                transform (β-1: 1.52 linear / 1.82 log; override via guide_slope),
                anchored to Leike at the smallest scale.
    band      : half-width of the shaded normalization envelope (PS, default
                100x = a 10x field-normalization uncertainty squared).
    stride    : in-plane downsampling for both sides (default 2 for speed).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    nominal = dict(clip_threshold=clip_threshold, epoch=epoch,
                   log_floor=log_floor, footprint=footprint)
    faint = dict(lw=1, alpha=0.7)
    echo_handles, leike_handles = [], []

    is_ps = domain in ('ps', 'ps(sf)')

    def _draw(x, y, group, **kw):
        group.append(ax.loglog(x, y, **kw)[0])

    # ---- echo cloud ----  (nominal C0, variations cycle C1, C2, ...)
    fullsky = read_fullmap(stride=stride)
    # SF domains reuse one nominal-clip SF across the epoch sweep; 'ps' has none.
    sf_nom = (None if domain == 'ps' else
              compute_s2_echo(fullsky, clip_threshold=clip_threshold,
                              transform=transform, norm=norm))

    def _echo_xy(epoch_, clip_):
        if domain == 'ps':
            x, y = echo_ps_curve(fullsky=fullsky, epoch=epoch_,
                                 clip_threshold=clip_, stride=stride,
                                 transform=transform, norm=norm)
        else:
            if clip_ == clip_threshold:                 # reuse the nominal SF
                r, s2 = echo_s2_curve(sf=sf_nom, epoch=epoch_)
            else:
                r, s2 = echo_s2_curve(fullsky=fullsky, clip_threshold=clip_,
                                      epoch=epoch_, transform=transform, norm=norm)
            x, y = (s2_to_ps(r, s2) if domain == 'ps(sf)'
                    else (r * LY_PER_PC, s2))
        return x, np.asarray(y, float) * fill_factor

    _draw(*_echo_xy(epoch, clip_threshold), echo_handles, color='C0', lw=2.5,
          label='echo (nominal)')
    ci = 1
    for param in ECHO_PARAMS:
        for v in DEFAULT_SWEEPS[param]:
            if v == nominal[param]:
                continue
            x, y = (_echo_xy(v, clip_threshold) if param == 'epoch'
                    else _echo_xy(epoch, v))
            _draw(x, y, echo_handles, color=f'C{ci}', label=f'{param}={v}',
                  **faint)
            ci += 1

    # ---- Leike / Edenhofer cloud ----  (nominal C0, variations C1, C2, ...)
    if dust_source == 'edenhofer':
        cube = load_edenhofer_cube(edenhofer_path, sample=edenhofer_sample)
        dust_label = ('Edenhofer mean' if edenhofer_sample is None
                      else f'Edenhofer sample {edenhofer_sample}')
    else:
        cube = _load_leike(leike_h5, use_sample, leike_samples_h5)
        dust_label = ('Leike mean' if use_sample == 'mean'
                      else f'Leike sample {use_sample}')

    def _dust_xy(cube_, **over):
        opts = dict(footprint=footprint, leike_axis=leike_axis, slices=slices,
                    log_floor=log_floor, stride=stride, transform=transform)
        opts.update(over)
        if domain == 'ps':
            return leike_ps_curve(cube=cube_, **opts)
        r, s2 = leike_s2_curve(cube=cube_, **opts)
        return s2_to_ps(r, s2) if domain == 'ps(sf)' else (r * LY_PER_PC, s2)

    _draw(*_dust_xy(cube), leike_handles, color='C0', lw=2.5,
          label=f'{dust_label} (nominal)')
    ci = 1
    for param in LEIKE_PARAMS:
        for v in DEFAULT_SWEEPS[param]:
            if v == nominal[param]:
                continue
            lbl = (f'z={_footprint_z_pc(v, cube, leike_axis):+d} pc'
                   if param == 'footprint' else f'{param}={v}')
            _draw(*_dust_xy(cube, **{param: v}), leike_handles, color=f'C{ci}',
                  label=lbl, **faint)
            ci += 1

    # one extra line: the same nominal choices but Edenhofer instead of Leike
    # (uses the adjacent .npy cache when present, else builds from the FITS)
    if dust_source == 'leike':
        eden = load_edenhofer_cube(edenhofer_path, sample=edenhofer_sample)
        lbl = ('Edenhofer mean' if edenhofer_sample is None
               else f'Edenhofer sample {edenhofer_sample}')
        _draw(*_dust_xy(eden), leike_handles, color=f'C{ci}',
              label=f'{lbl} (nominal)', **faint)
        ci += 1

    # one extra line: a single Leike posterior sample at the nominal choices
    # (drawn against either cloud).  Where it lifts above the mean's curve the
    # reconstruction is prior- (not data-) driven.  Skip it if the Leike cloud
    # is already this very sample (use_sample), to avoid a duplicate line.
    redundant = (dust_source == 'leike' and use_sample != 'mean'
                 and leike_sample is not None
                 and int(use_sample) == int(leike_sample))
    if leike_sample is not None and not redundant:
        if os.path.exists(_existing_path(leike_samples_h5)):
            scube = _load_leike_sample(leike_samples_h5, leike_sample)
            _draw(*_dust_xy(scube), leike_handles, color=f'C{ci}',
                  label=f'Leike sample {leike_sample}', **faint)
            ci += 1

    if is_ps and guide:
        slope = LEIKE_PS_SLOPE[transform] if guide_slope is None else guide_slope
        _ps_guide(ax, echo_handles[0], leike_handles[0], slope, band=band)
        # keep the y-axis on the data (not the wide band); base it on the two
        # nominal curves so both clouds stay in view and faint/pathological
        # variations don't set the range
        nom = np.concatenate([np.asarray(echo_handles[0].get_ydata(), float),
                              np.asarray(leike_handles[0].get_ydata(), float)])
        nom = nom[np.isfinite(nom) & (nom > 0)]
        if nom.size:
            ax.set_ylim(nom.min() / 3, nom.max() * 3)

    if is_ps:
        ax.set_xlabel('k  [1/pc]')
        ax.set_ylabel('P(k)  ' + ('(from S₂)' if domain == 'ps(sf)' else '(direct)'))
    else:
        ax.set_xlabel('r  [ly]')
        ax.set_ylabel(_s2_ylabel(transform))
    ax.set_title(f'Echo vs {dust_label} — sensitivity '
                 f'({transform}, {domain}, stride={stride})')
    # PS inverts the x-axis (Leike at low k on the left), so put Leike first.
    left, right = ((leike_handles, echo_handles) if is_ps
                   else (echo_handles, leike_handles))
    _two_column_legend(ax, left, right, fontsize=7)
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
                         use_sample=0, samples_h5='leike2020/samples.h5',
                         n_bins=30, octants=True, fit_pc=(2.3, 125),
                         out_png='leike2020_fig4.png', ax=None):
    """Reproduce Leike+2020 Fig. 4: the 3D power spectrum of the dust cube.

    Splits the released (740, 740, 540) cube at its centre into eight octants
    and plots each octant's spectrum (octant 3 highlighted), then prints the
    full-volume slope against the published beta (2.52 linear / 2.82 log).

    use_sample : posterior sample index (int, default 0) or 'mean'.  The MEAN's
                empirical spectrum is biased steep (~3.1/3.7 vs the published
                2.52/2.82) because the Wiener-smoothed mean loses small-scale
                power; a SAMPLE carries that power and recovers a slope near the
                published (inferred) beta.  So samples reproduce Fig. 4, the mean
                does not.
    field_kind : 'linear' (Fig. 4) or 'log' (Fig. 13)
    octants    : if False, only the full-volume curve is computed/plotted
    fit_pc     : (lo, hi) scale range [pc] for the slope fit
    """
    cube = _load_leike(leike_h5, use_sample, samples_h5, units='raw')
    src = 'mean' if use_sample == 'mean' else f'sample {use_sample}'
    print(f"Loaded {src} cube {cube.shape}, "
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
    ax.set_title(f'Leike+2020 Fig. 4 [{src}]' +
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
