"""The five headline results, published (arcsinh) against linear flux units.

Every quantity is computed with the SAME estimators the paper uses -- imported
from make_tier_figures / shape_center, not reimplemented -- so the only thing
that differs between the two columns is the preprocessing:

    published : background=0.03, arcsinh_scale=0.03   (results/..._k4.csv)
    linear    : background=0,    arcsinh_scale=None   (results/..._k4_linear.csv)

and, because the SNR tiering is itself an S2 plateau-over-floor ratio measured
in whatever units the field was transformed to, the q4 sample as well.  Those
two effects are confounded by construction; `--frozen-tiers` additionally
reports the linear fits tiered by the PUBLISHED SNR, which isolates the refit.

    python analysis/compare_linear_vs_arcsinh.py

Writes results/linear_vs_arcsinh_headline.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

import make_tier_figures as mtf  # noqa: E402
import shape_center as sc  # noqa: E402

K = 4


def _tables(variant, tier_from=None):
    """(usable 3D rows, q4 subset, raw table) for one preprocessing variant.

    `tier_from` overrides which SNR table supplies the tiering, so the linear
    fits can be tiered by the published SNR to separate the refit effect from
    the sample-change effect.
    """
    d = mtf.load(k=K, variant=variant)[0]
    if tier_from is not None:
        snr = pd.read_csv(os.path.join(
            _ROOT, 'results', 'noise_audit_table%s.csv' % tier_from))
        q75, q50 = np.percentile(snr.snr, [75, 50])
        snr['tier'] = np.where(snr.snr >= q75, 'q4',
                               np.where(snr.snr >= q50, 'q3', 'bottom_half'))
        d = d.drop(columns=['snr', 'tier']).merge(
            snr[['row', 'col', 'snr', 'tier']], on=['row', 'col'], how='left')
    u = d[mtf.usable(d)]
    return u, u[u.tier == 'q4']


def _raw2d(variant):
    tag = 'r%g_s%d_k%d' % (mtf.RCUT, mtf.STRIDE, K)
    raw = pd.read_csv(os.path.join(
        _ROOT, 'results', 'singleband_powerlaw_%s%s.csv' % (tag, variant)))
    return raw[(raw['mode'] == '2d') & (raw.profile == 'powerlaw')]


def headline(variant, tier_from=None):
    """The five results as a flat dict of scalars."""
    u, q4 = _tables(variant, tier_from=tier_from)
    out = {'n_usable': len(u), 'n_q4': len(q4)}

    # ---- result 1: one shape, and it is prolate
    for c in ('a2a1', 'a3a2'):
        mu, sig, se_mu, _ = mtf.ml_center_and_scatter(
            q4[c].values, q4['se_' + c].values)
        out['common_' + c] = 10 ** mu
        out['sigint_%s_dex' % c] = sig
    out['axis_a3a1'] = out['common_a2a1'] * out['common_a3a2']

    P = np.log10(q4.a3a2.values) - np.log10(q4.a2a1.values)
    sP = np.hypot((q4.se_a2a1 / q4.a2a1 / np.log(10)).values,
                  (q4.se_a3a2 / q4.a3a2 / np.log(10)).values)
    muP, sigP = sc.ml_center_and_scatter(P, sP)[:2]
    seP = 1 / np.sqrt((1 / (sP ** 2 + sigP ** 2)).sum())
    out.update(prolateness_dex=muP, prolateness_sigma=muP / seP,
               n_prolate=int((P > 0).sum()))

    # ---- result 2: the shapes are not identical
    for c in ('a2a1', 'a3a2'):
        lv = np.log10(q4[c].values)
        sl = (q4['se_' + c] / q4[c] / np.log(10)).values
        sig, lo, hi, dchi2 = sc.profile_ci(lv, sl)
        out['scatter_%s_pct' % c] = 100 * (10 ** sig - 1)
        out['p_zero_scatter_' + c] = stats.chi2.sf(dchi2, 1)

    # ---- result 3: the 2D appearance is slicing, not structure
    j = q4[['row', 'col', 'incl']].merge(
        _raw2d(variant)[['row', 'col', 'b2b1']], on=['row', 'col'])
    rho, p = stats.spearmanr(j.incl, j.b2b1)
    out.update(spearman_rho=rho, spearman_p=p,
               b2b1_min=j.b2b1.min(), b2b1_max=j.b2b1.max())

    # ---- result 4: one small-lag slope
    mu, sig, se_mu, _ = mtf.ml_center_and_scatter(
        q4.alpha.values, q4.se_alpha.values)
    out.update(alpha_common=10 ** mu, alpha_median=float(np.median(q4.alpha)),
               alpha_sigint_dex=sig)

    # ---- result 5: the long axis prefers the echo plane
    v = q4.incl.dropna()
    ks = stats.kstest(np.cos(np.radians(v)), 'uniform')
    out.update(incl_median_deg=float(v.median()),
               frac_within30_of_plane=float((v > 60).mean()),
               frac_within30_of_W=float((v < 30).mean()),
               ks_p_isotropic=ks.pvalue)
    return out


LABELS = [
    ('n_usable', 'usable windows', '%.0f'),
    ('n_q4', 'windows in q4 (top-SNR quartile)', '%.0f'),
    ('common_a2a1', 'R1  common a2/a1', '%.4f'),
    ('common_a3a2', 'R1  common a3/a2', '%.4f'),
    ('axis_a3a1', 'R1  implied a3/a1', '%.4f'),
    ('prolateness_dex', 'R1  prolateness [dex]', '%.4f'),
    ('prolateness_sigma', 'R1  prolate significance [sigma]', '%.2f'),
    ('n_prolate', 'R1  windows individually prolate', '%.0f'),
    ('sigint_a2a1_dex', 'R2  intrinsic scatter a2/a1 [dex]', '%.4f'),
    ('sigint_a3a2_dex', 'R2  intrinsic scatter a3/a2 [dex]', '%.4f'),
    ('scatter_a2a1_pct', 'R2  intrinsic scatter a2/a1 [%]', '%.1f'),
    ('scatter_a3a2_pct', 'R2  intrinsic scatter a3/a2 [%]', '%.1f'),
    ('p_zero_scatter_a2a1', 'R2  p(one exact shape) a2/a1', '%.2e'),
    ('p_zero_scatter_a3a2', 'R2  p(one exact shape) a3/a2', '%.2e'),
    ('spearman_rho', 'R3  rho(incl vs b2/b1)', '%.4f'),
    ('spearman_p', 'R3  p value', '%.2e'),
    ('b2b1_min', 'R3  measured b2/b1 min', '%.4f'),
    ('b2b1_max', 'R3  measured b2/b1 max', '%.4f'),
    ('alpha_common', 'R4  common alpha', '%.4f'),
    ('alpha_median', 'R4  median alpha', '%.4f'),
    ('alpha_sigint_dex', 'R4  alpha intrinsic scatter [dex]', '%.4f'),
    ('incl_median_deg', 'R5  median inclination [deg]', '%.1f'),
    ('frac_within30_of_plane', 'R5  fraction within 30 deg of plane', '%.3f'),
    ('frac_within30_of_W', 'R5  fraction within 30 deg of W', '%.3f'),
    ('ks_p_isotropic', 'R5  KS p vs isotropic', '%.2e'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        _ROOT, 'results', 'linear_vs_arcsinh_headline.csv'))
    args = ap.parse_args()

    pub = headline('')
    lin = headline('_linear')
    frz = headline('_linear', tier_from='')

    rows = []
    for key, label, fmt in LABELS:
        a, b, c = pub[key], lin[key], frz[key]
        rows.append(dict(
            quantity=label, key=key,
            published=fmt % a, linear=fmt % b, linear_frozen_tiers=fmt % c,
            pct_change=(100 * (b - a) / abs(a)) if a not in (0,) else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False, max_colwidth=42))
    print('\nwrote %s' % os.path.relpath(args.out, _ROOT))


if __name__ == '__main__':
    main()
