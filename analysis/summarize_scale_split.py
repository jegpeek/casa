"""Summarize a scale_split.py run: are shape and orientation scale-independent?

Reads data/scale_split_s<stride>/*.json and produces one row per window with
the inner-band and outer-band axis ratios and principal-axis orientation,
jackknife errors on each, and the paired inner-minus-outer differences.

Two conventions matter here.

1.  Orientation is compared as the UNSIGNED angle between axis DIRECTIONS,
    arccos|n_A . n_B| in degrees.  Eigenvectors have arbitrary sign and the
    stored Euler angles (theta, phi, psi) flip discontinuously under axis
    relabelling, so differencing them directly manufactures huge spurious
    disagreements.  The unsigned inter-axis angle is invariant to both.

2.  The absolute axis length a1 is NOT compared between bands.  In the
    power-law regime S2 -> var_inf |L^-1 dr|^alpha, scaling L by c is absorbed
    by var_inf -> var_inf c^alpha, so a1 is near-degenerate within a narrow
    band.  Ratios and orientation are unaffected and are what we test.

Errors are the delete-one-block jackknife spread over the K*K samples,
var = (N-1)/N * sum (x_i - xbar)^2, matching _jackknife_stderr in
structure_function.py.  For the paired difference the jackknife is applied to
the DIFFERENCE per block, which correctly retains the strong positive
correlation between the two bands' errors (they share the same image).

Usage:  python analysis/summarize_scale_split.py [stride] [mode] [out_csv]
          mode = 'median' (equal counts) | 'logmid' (equal log-radius coverage)
"""
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def axes_from_angles(theta_deg, phi_deg, psi_deg):
    """Unit vectors (n1, n2, n3) in (U, V, W) from the stored Euler angles.

    Mirrors the construction in principal_axes_from_params, inverted.
    """
    th, ph, ps = np.radians([theta_deg, phi_deg, psi_deg])
    n1 = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    n2_0 = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    n3_0 = np.array([-np.sin(ph), np.cos(ph), 0.0])
    n2 = np.cos(ps) * n2_0 + np.sin(ps) * n3_0
    n3 = np.cross(n1, n2)
    return n1, n2, n3


def axis_angle(rec_a, rec_b, which=0):
    """Unsigned angle [deg] between the `which`-th principal axes of two fits."""
    A = axes_from_angles(rec_a['theta'], rec_a['phi'], rec_a['psi'])
    B = axes_from_angles(rec_b['theta'], rec_b['phi'], rec_b['psi'])
    c = abs(float(np.dot(A[which], B[which])))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def jk_stderr(vals):
    """Delete-one jackknife standard error, matching _jackknife_stderr."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    n = v.size
    if n < 2:
        return np.nan
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def _scalars(rec):
    """Derived comparables from one band's fit record."""
    if 'error' in rec or not rec.get('fit_success', False):
        return None
    a1, a2, a3 = rec['a1'], rec['a2'], rec['a3']
    if not all(np.isfinite([a1, a2, a3])) or a1 <= 0 or a2 <= 0:
        return None
    return dict(a1=a1, a2a1=a2 / a1, a3a2=a3 / a2, alpha=rec.get('alpha', np.nan),
                theta=rec['theta'], phi=rec['phi'], psi=rec['psi'])


def summarize(stride=2, mode='median'):
    suffix = '' if mode == 'median' else f'_{mode}'
    sys.path.insert(0, f'{_ROOT}/analysis')
    import preprocessing_mode as pm
    files = sorted(glob.glob(
        f'{_ROOT}/data/scale_split{suffix}_s{stride}{pm.variant_suffix()}/*.json'))
    rows = []
    for fn in files:
        d = json.load(open(fn))
        ci, co, cf = (_scalars(d['central'][k]) for k in ('inner', 'outer', 'full'))
        if ci is None or co is None:
            continue
        r = dict(chunk=f"r{d['row']}_c{d['col']}", size=d['size'],
                 r_split=d['r_split'], n_inner=d['n_inner'], n_outer=d['n_outer'],
                 w_frac_inner=d.get('w_frac_inner', np.nan),
                 r_min=d['r_min'], r_max=d['r_max'])
        for tag, c in (('in', ci), ('out', co)):
            for k in ('a1', 'a2a1', 'a3a2', 'alpha'):
                r[f'{k}_{tag}'] = c[k]
        r['ang1'] = axis_angle(d['central']['inner'], d['central']['outer'], 0)
        r['ang2'] = axis_angle(d['central']['inner'], d['central']['outer'], 1)
        if cf is not None:
            r['a2a1_full'] = cf['a2a1']; r['a3a2_full'] = cf['a3a2']
            r['alpha_full'] = cf['alpha']

        # jackknife: per-block band fits -> errors on each band and on the diff
        per = {k: [] for k in ('a2a1_in', 'a2a1_out', 'a3a2_in', 'a3a2_out',
                               'alpha_in', 'alpha_out', 'd_a2a1', 'd_a3a2',
                               'd_alpha', 'ang1', 'ang2')}
        nb_ok = 0
        for s in d['samples']:
            si, so = _scalars(s['inner']), _scalars(s['outer'])
            if si is None or so is None:
                continue
            nb_ok += 1
            for k in ('a2a1', 'a3a2', 'alpha'):
                per[f'{k}_in'].append(si[k]); per[f'{k}_out'].append(so[k])
                per[f'd_{k}'].append(si[k] - so[k])
            per['ang1'].append(axis_angle(s['inner'], s['outer'], 0))
            per['ang2'].append(axis_angle(s['inner'], s['outer'], 1))
        r['n_blocks_ok'] = nb_ok
        for k, v in per.items():
            r[f'se_{k}'] = jk_stderr(v)

        for k in ('a2a1', 'a3a2', 'alpha'):
            r[f'd_{k}'] = r[f'{k}_in'] - r[f'{k}_out']
            se = r[f'se_d_{k}']
            r[f'z_{k}'] = r[f'd_{k}'] / se if se and np.isfinite(se) and se > 0 else np.nan
        for k in ('ang1', 'ang2'):
            se = r[f'se_{k}']
            r[f'z_{k}'] = r[k] / se if se and np.isfinite(se) and se > 0 else np.nan
        rows.append(r)
    return rows


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mode = sys.argv[2] if len(sys.argv) > 2 else 'median'
    suffix = '' if mode == 'median' else f'_{mode}'
    # The preprocessing variant is part of the output name, so a raw-flux
    # summary cannot overwrite the arcsinh table the committed figures use.
    sys.path.insert(0, f'{_ROOT}/analysis')
    import preprocessing_mode as pm
    out = (sys.argv[3] if len(sys.argv) > 3
           else f'{_ROOT}/results/scale_split{suffix}_s{stride}'
                f'{pm.variant_suffix()}.csv')
    rows = summarize(stride, mode)
    if not rows:
        print('no usable windows'); return
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join('' if r.get(k) is None or
                              (isinstance(r.get(k), float) and not np.isfinite(r[k]))
                              else str(r.get(k)) for k in keys) + '\n')
    print(f'{len(rows)} windows -> {out}')


if __name__ == '__main__':
    main()
