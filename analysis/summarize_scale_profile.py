"""Turn a scale_profile run into per-band and per-window slope tables.

Two CSVs are written:

  results/scale_profile_d<dex>_s<stride>_bands.csv
      one row per (window, band): T, a2a1, a3a2, a3a1, alpha and their 2x2
      block-jackknife standard errors.

  results/scale_profile_d<dex>_s<stride>_slopes.csv
      one row per window: dT/dlog10(r) and the same for the other shape
      measures, each with a jackknife standard error.

Why the slope error must come from the jackknife
------------------------------------------------
The bands OVERLAP (step = band_dex/2) and every band is measured on the same
images, so the per-band values are strongly correlated and an ordinary
least-squares slope error would be badly underestimated.  Instead the slope is
refit inside each delete-one-block jackknife sample and the spread over the k^2
samples is taken -- exactly the convention used for paired differences in
summarize_scale_split.py, and for the same reason.

Usage:  python analysis/summarize_scale_profile.py [stride] [band_dex]
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

import scale_split as ss  # noqa: E402  (needs the sys.path above)

KEYS = ('T', 'p', 'a2a1', 'a3a2', 'a3a1', 'alpha')

# Degeneracy rejection.  fit_success=True is NOT sufficient: the optimiser can
# park on a boundary and report success with a collapsed axis.  Two independent
# symptoms are seen in the d0.6_s2 run (10 of 145 band fits):
#   a3 -> 0     (short axis annihilated; T then returns exactly 1.0, i.e. a
#                maximally confident "prolate" reading from a degenerate fit,
#                while p = ln(a1 a3/a2^2) correctly goes non-finite)
#   a2/a1 -> 0  (long axis runs away; a1 reached 920 ly for a 0.5 ly lag range)
# Both are rejected: an axis shorter than the smallest sampled lag radius is not
# resolved by the data, and axis ratios below 0.02 are the same degeneracy floor
# used elsewhere in this project.  The two criteria disagree on only 3 fits, so
# the union is applied for safety.
RATIO_FLOOR = 0.02


def is_degenerate(band, r_min_global):
    """True if this band fit sits on a boundary and must not enter a statistic."""
    if not band.get('fit_success'):
        return True
    a1, a2, a3 = band.get('a1'), band.get('a2'), band.get('a3')
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in (a1, a2, a3)):
        return True
    if a1 <= 0 or a2 <= 0 or a3 <= 0:
        return True
    if a3 < r_min_global:            # short axis below the resolution of the data
        return True
    if a2 / a1 < RATIO_FLOOR or a3 / a2 < RATIO_FLOOR:
        return True
    return False


def _slope_of(bands, key):
    """OLS slope of key vs log10(r_mid) over converged bands; nan if <3 points."""
    ok = [b for b in bands if b.get('_ok') and np.isfinite(b.get(key, np.nan))]
    x = [np.log10(b['r_mid']) for b in ok]
    y = [b[key] for b in ok]
    if len(x) < 3:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _jk_se(values):
    """Jackknife standard error from delete-one-block refits.

    For a delete-one jackknife with n blocks, se = sqrt((n-1)/n * sum (x_i - xbar)^2).
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return np.nan
    n = v.size
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def summarize(stride=2, band_dex=0.6, data_dir=None,
              profile=ss.CANONICAL_PROFILE):
    """Aggregate the per-window scale-profile JSON into band and slope rows.

    data_dir defaults to the repo's own data/ tree; pass one to summarize a
    rerun written elsewhere (the reproduction notebook points it at rerun/).
    `profile` selects which per-band fit form to read, and resolves to the
    directory suffix scale_profile.main writes (unsuffixed for the canonical
    profile).  Every file read is checked against it.
    """
    if data_dir is None:
        data_dir = os.path.join(_ROOT, 'data')
    import preprocessing_mode as pm
    suffix = ss.profile_suffix(profile) + pm.variant_suffix()
    pat = os.path.join(data_dir,
                       f'scale_profile_d{band_dex:g}_s{stride}{suffix}',
                       '*.json')
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError('no scale-profile JSON under %s' % pat)
    # A directory name is a claim; the per-file tag is the evidence.  Checking
    # here means a mislabelled or hand-moved tree cannot reach a results table.
    bad = {f: t for f, t in
           ((f, json.load(open(f)).get('profile')) for f in files)
           if t != profile}
    if bad:
        f0, t0 = sorted(bad.items())[0]
        raise RuntimeError(
            '%d of %d files under %s were fit with a different profile '
            '(e.g. %s has profile=%r, expected %r)'
            % (len(bad), len(files), os.path.dirname(pat),
               os.path.basename(f0), t0, profile))
    # global smallest sampled lag radius: the resolution limit for any axis
    r_min_global = min(json.load(open(f))['bands'][0]['r_lo'] for f in files)
    band_rows, slope_rows = [], []
    for fn in files:
        d = json.load(open(fn))
        chunk = 'r%d_c%d' % (d['row'], d['col'])
        cen = d['bands']
        samples = d['samples']
        # tag every band (central and jackknife) with its usability
        for b in cen:
            b['_ok'] = not is_degenerate(b, r_min_global)
        for s_ in samples:
            for b in s_['bands']:
                b['_ok'] = not is_degenerate(b, r_min_global)

        # ---- per band
        for bi, b in enumerate(cen):
            row = dict(chunk=chunk, band=bi, r_lo=b['r_lo'], r_hi=b['r_hi'],
                       r_mid=b['r_mid'], fit_success=bool(b.get('fit_success')),
                       usable=bool(b['_ok']))
            for k in KEYS:
                row[k] = b.get(k, np.nan)
                jk = [s['bands'][bi].get(k, np.nan) for s in samples
                      if s['bands'][bi].get('_ok')]
                row[f'se_{k}'] = _jk_se(jk)
            row['n_ok_jk'] = int(sum(1 for s in samples if s['bands'][bi].get('_ok')))
            band_rows.append(row)

        # ---- per window slope
        row = dict(chunk=chunk, n_bands=len(cen),
                   n_ok=int(sum(1 for b in cen if b['_ok'])),
                   n_degenerate=int(sum(1 for b in cen if not b['_ok'])),
                   r_min=cen[0]['r_lo'], r_max=cen[-1]['r_hi'])
        for k in KEYS:
            row[f'slope_{k}'] = _slope_of(cen, k)
            row[f'se_slope_{k}'] = _jk_se([_slope_of(s['bands'], k) for s in samples])
            # end-to-end change across the fitted range, for interpretability
            ok = [b for b in cen if b['_ok'] and np.isfinite(b.get(k, np.nan))]
            row[f'{k}_first'] = ok[0][k] if ok else np.nan
            row[f'{k}_last'] = ok[-1][k] if ok else np.nan
        slope_rows.append(row)
    return band_rows, slope_rows


def _write(rows, path):
    if not rows:
        print(f'no rows for {path}')
        return
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} rows -> {path}')


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    band_dex = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    profile = sys.argv[3] if len(sys.argv) > 3 else ss.CANONICAL_PROFILE
    band_rows, slope_rows = summarize(stride, band_dex, profile=profile)
    # The preprocessing variant is part of the tag, so a raw-flux summary can
    # never overwrite the arcsinh table the committed figures were built from.
    import preprocessing_mode as pm
    tag = (f'd{band_dex:g}_s{stride}{ss.profile_suffix(profile)}'
           f'{pm.variant_suffix()}')
    _write(band_rows, f'{_ROOT}/results/scale_profile_{tag}_bands.csv')
    _write(slope_rows, f'{_ROOT}/results/scale_profile_{tag}_slopes.csv')


if __name__ == '__main__':
    main()
