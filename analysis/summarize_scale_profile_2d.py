"""Turn a scale_profile_2d run into per-band and per-window tables, and compare
the measured in-plane anisotropy against what the 3D fits predict.

Three CSVs are written:

  results/scale_profile_2d_d<dex>_s<stride>_bands.csv
      one row per (window, band): b1, b2b1, pa2d, alpha and their 2x2
      block-jackknife standard errors.

  results/scale_profile_2d_d<dex>_s<stride>_slopes.csv
      one row per window: d(b2/b1)/dlog10(r) etc, with jackknife errors.

  results/scale_profile_2d_d<dex>_s<stride>_vs3d.csv
      one row per (window, band) joining the 2D MEASUREMENT to the 3D fit's
      IMPLIED dW=0 slice for the same window and band.  This is the table that
      answers the science question, so it carries both b2b1 columns and their
      difference.

The comparison this enables
---------------------------
A fixed 3D ellipsoid sliced at different orientations presents different
in-plane ellipses, so some of the window-to-window spread in 2D anisotropy is
pure geometry rather than real variation in the structures.  The 3D fits predict
each window's slice exactly (see structure_function.principal_axes_2d), so:

  spread(2D measured)   -- total observed range of in-plane anisotropy
  spread(3D-implied)    -- the part explained by slicing the fitted ellipsoids
  spread(residual)      -- what the 3D model does NOT account for

Degeneracy rejection uses the same philosophy as the 3D summarizer but on two
axes: b2 below the smallest sampled lag radius is unresolved, and b2/b1 below
RATIO_FLOOR is the same degeneracy floor used elsewhere in this project.
Because the 2D fit drops ~2/3 of the lag pairs, degenerate fits are expected to
be somewhat MORE common here than in 3D, so the usable flag matters more.

Usage:  python analysis/summarize_scale_profile_2d.py [stride] [band_dex]
"""
import csv
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

from summarize_scale_profile import RATIO_FLOOR, _jk_se, _slope_of, _write  # noqa: E402

KEYS = ('b2b1', 'b1', 'pa2d', 'alpha')


def is_degenerate_2d(band, r_min_global):
    """True if this in-plane band fit sits on a boundary."""
    if not band.get('fit_success'):
        return True
    b1, b2 = band.get('b1'), band.get('b2')
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in (b1, b2)):
        return True
    if b1 <= 0 or b2 <= 0:
        return True
    if b2 < r_min_global:          # minor axis below the resolution of the data
        return True
    if b2 / b1 < RATIO_FLOOR:
        return True
    return False


def _load(stride, band_dex, kind='2d', profile=None):
    """Per-window JSON for one run.  The 2D and 3D trees must share a profile:
    this module's whole purpose is a like-for-like comparison of the two."""
    import scale_split as ss
    if profile is None:
        profile = ss.CANONICAL_PROFILE
    suffix = ss.profile_suffix(profile)
    sub = ('scale_profile_2d_d%g_s%d%s' % (band_dex, stride, suffix)
           if kind == '2d'
           else 'scale_profile_d%g_s%d%s' % (band_dex, stride, suffix))
    files = sorted(glob.glob(f'{_ROOT}/data/{sub}/*.json'))
    bad = [f for f in files if json.load(open(f)).get('profile') != profile]
    if bad:
        raise RuntimeError('%d of %d files under data/%s were not fit with '
                           'profile %r (e.g. %s)'
                           % (len(bad), len(files), sub, profile,
                              os.path.basename(bad[0])))
    return files


def summarize(stride=2, band_dex=0.6, profile=None):
    files = _load(stride, band_dex, '2d', profile)
    if not files:
        raise SystemExit('no scale_profile_2d output found')
    r_min_global = min(json.load(open(f))['bands'][0]['r_lo'] for f in files)
    band_rows, slope_rows = [], []
    for fn in files:
        d = json.load(open(fn))
        chunk = 'r%d_c%d' % (d['row'], d['col'])
        cen, samples = d['bands'], d['samples']
        for b in cen:
            b['_ok'] = not is_degenerate_2d(b, r_min_global)
        for s_ in samples:
            for b in s_['bands']:
                b['_ok'] = not is_degenerate_2d(b, r_min_global)

        for bi, b in enumerate(cen):
            row = dict(chunk=chunk, band=bi, r_lo=b['r_lo'], r_hi=b['r_hi'],
                       r_mid=b['r_mid'], fit_success=bool(b.get('fit_success')),
                       usable=bool(b['_ok']), b2=b.get('b2', np.nan),
                       rms_resid=b.get('rms_resid', np.nan),
                       n_fit=b.get('n_fit', 0))
            for k in KEYS:
                row[k] = b.get(k, np.nan)
                jk = [s['bands'][bi].get(k, np.nan) for s in samples
                      if s['bands'][bi].get('_ok')]
                row[f'se_{k}'] = _jk_se(jk)
            row['n_ok_jk'] = int(sum(1 for s in samples if s['bands'][bi].get('_ok')))
            band_rows.append(row)

        row = dict(chunk=chunk, n_bands=len(cen),
                   n_ok=int(sum(1 for b in cen if b['_ok'])),
                   n_degenerate=int(sum(1 for b in cen if not b['_ok'])),
                   r_min=cen[0]['r_lo'], r_max=cen[-1]['r_hi'])
        for k in KEYS:
            row[f'slope_{k}'] = _slope_of(cen, k)
            row[f'se_slope_{k}'] = _jk_se([_slope_of(s['bands'], k) for s in samples])
            ok = [b for b in cen if b['_ok'] and np.isfinite(b.get(k, np.nan))]
            row[f'{k}_first'] = ok[0][k] if ok else np.nan
            row[f'{k}_last'] = ok[-1][k] if ok else np.nan
        slope_rows.append(row)
    return band_rows, slope_rows


def compare_to_3d(stride=2, band_dex=0.6, profile=None):
    """Join the 2D measurement to the 3D fit's implied dW=0 slice, per band.

    The 3D run stores b1/b2/b2b1/pa2d for every band via sf._fit_scalars, so the
    predicted slice needs no refitting -- it is already the same quantity
    computed by the same function.
    """
    from summarize_scale_profile import is_degenerate as is_degenerate_3d
    import structure_function as sf

    def _slice_of(band):
        """b2/b1 and PA of the dW=0 slice implied by a 3D band fit.

        Runs made before principal_axes_2d existed do not carry b1/b2/pa2d, but
        they DO store s11/s22/l12, and the slice is a pure function of those --
        so it is recomputed here rather than requiring the 3D run to be redone.
        """
        if np.isfinite(band.get('b2b1', np.nan)):
            return band['b2b1'], band.get('pa2d', np.nan), band.get('b1', np.nan)
        if not all(np.isfinite(band.get(k, np.nan)) for k in ('s11', 's22', 'l12')):
            return np.nan, np.nan, np.nan
        ax = sf.principal_axes_2d(band)
        r = ax['b2'] / ax['b1'] if ax['b1'] > 0 else np.nan
        return r, float(np.degrees(ax['pa'])), ax['b1']

    # Both sides resolve through the same profile, so the comparison can never
    # silently pit a 2D power-law fit against a 3D Weibull one.
    f2 = _load(stride, band_dex, '2d', profile)
    f3 = _load(stride, band_dex, '3d', profile)
    if not f3:
        print('no 3D run to compare against; skipping vs3d table')
        return []
    r2min = min(json.load(open(f))['bands'][0]['r_lo'] for f in f2)
    r3min = min(json.load(open(f))['bands'][0]['r_lo'] for f in f3)

    def _by_window(files):
        out = {}
        for fn in files:
            d = json.load(open(fn))
            out['r%d_c%d' % (d['row'], d['col'])] = d
        return out

    D2, D3 = _by_window(f2), _by_window(f3)
    rows = []
    for chunk in sorted(set(D2) & set(D3)):
        b2s, b3s = D2[chunk]['bands'], D3[chunk]['bands']
        for bi in range(min(len(b2s), len(b3s))):
            a, b = b2s[bi], b3s[bi]
            # bands are constructed from the same edges recipe but on different
            # point sets (2D keeps only dW=0), so record both radii rather than
            # assuming they coincide
            ok2 = not is_degenerate_2d(a, r2min)
            ok3 = not is_degenerate_3d(b, r3min)
            pred_ratio, pred_pa, pred_b1 = _slice_of(b)
            rows.append(dict(
                chunk=chunk, band=bi,
                r_mid_2d=a['r_mid'], r_mid_3d=b['r_mid'],
                b2b1_meas=a.get('b2b1', np.nan),
                b2b1_pred=pred_ratio,
                b1_meas=a.get('b1', np.nan), b1_pred=pred_b1,
                pa2d_meas=a.get('pa2d', np.nan),
                pa2d_pred=pred_pa,
                a2a1_3d=(b['a2'] / b['a1']) if b.get('a1') else np.nan,
                a3a2_3d=(b['a3'] / b['a2']) if b.get('a2') else np.nan,
                usable_2d=ok2, usable_3d=ok3, usable_both=bool(ok2 and ok3),
                d_b2b1=(a.get('b2b1', np.nan) - pred_ratio),
            ))
    return rows


def main():
    import scale_split as ss
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    band_dex = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    profile = sys.argv[3] if len(sys.argv) > 3 else ss.CANONICAL_PROFILE
    band_rows, slope_rows = summarize(stride, band_dex, profile)
    # The result filenames carry the profile for the same reason the data
    # trees do: an unsuffixed name is a claim that these are canonical fits.
    tag = f'd{band_dex:g}_s{stride}{ss.profile_suffix(profile)}'
    os.makedirs(f'{_ROOT}/results', exist_ok=True)
    _write(band_rows, f'{_ROOT}/results/scale_profile_2d_{tag}_bands.csv')
    _write(slope_rows, f'{_ROOT}/results/scale_profile_2d_{tag}_slopes.csv')
    _write(compare_to_3d(stride, band_dex, profile),
           f'{_ROOT}/results/scale_profile_2d_{tag}_vs3d.csv')


if __name__ == '__main__':
    main()
