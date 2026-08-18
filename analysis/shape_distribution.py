"""Is the ellipsoid SHAPE the same everywhere in the echo?

This is the quantitative form of the project's central hypothesis: structures
that look different in the plane of the echo are similar in three dimensions.
If that holds, the axis ratios a2/a1 and a3/a2 should be consistent across
windows once measurement error is accounted for, even though the absolute
sizes and the 2D appearance vary.

The test is a constant-model chi-square with an intrinsic-scatter term:

    x_i = mu + N(0, sigma_int^2) + N(0, se_i^2)

sigma_int is estimated by maximum likelihood.  sigma_int consistent with zero
means the ratios are universal within the errors; sigma_int > 0 means there is
real window-to-window variation in shape beyond what the jackknife allows.

Reported alongside is the same statistic for a1 (the absolute size), which we
already know varies strongly -- it is the control that shows the test can
detect variation when variation is there.

Usage:  python analysis/shape_distribution.py <table.csv> <out_prefix>
"""
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats


def inverse_variance_mean(x, se):
    w = 1.0 / se ** 2
    mu = np.sum(w * x) / np.sum(w)
    return mu, np.sqrt(1.0 / np.sum(w))


def fit_intrinsic_scatter(x, se):
    """ML estimate of mu and sigma_int for x_i ~ N(mu, sigma_int^2 + se_i^2)."""
    def nll(theta):
        mu, log_s = theta
        v = np.exp(2 * log_s) + se ** 2
        return 0.5 * np.sum(np.log(2 * np.pi * v) + (x - mu) ** 2 / v)

    mu0, _ = inverse_variance_mean(x, se)
    s0 = max(np.std(x, ddof=1) ** 2 - np.mean(se ** 2), 1e-4) ** 0.5
    r = optimize.minimize(nll, [mu0, np.log(s0)], method='Nelder-Mead')
    mu, sig = r.x[0], float(np.exp(r.x[1]))

    # likelihood-ratio test against sigma_int = 0
    nll_free = nll(r.x)
    r0 = optimize.minimize_scalar(lambda m: nll([m, np.log(1e-8)]))
    nll_null = nll([r0.x, np.log(1e-8)])
    lr = 2 * (nll_null - nll_free)
    # boundary case: 50:50 mixture of chi2_0 and chi2_1
    p = 0.5 * stats.chi2.sf(max(lr, 0), 1) if lr > 0 else 1.0
    return dict(mu=mu, sigma_int=sig, lr=lr, p=p)


def constant_chi2(x, se):
    mu, _ = inverse_variance_mean(x, se)
    c2 = float(np.sum(((x - mu) / se) ** 2))
    dof = len(x) - 1
    return dict(chi2=c2, dof=dof, chi2_dof=c2 / dof,
                p=float(stats.chi2.sf(c2, dof)))


def analyse(df, cols=(('a2a1', 'se_a2a1'), ('a3a2', 'se_a3a2'), ('a1', 'se_a1'))):
    rows = []
    for val, err in cols:
        d = df[[val, err]].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[d[err] > 0]
        x, se = d[val].to_numpy(), d[err].to_numpy()
        rec = dict(quantity=val, n=len(x), median=float(np.median(x)),
                   raw_std=float(np.std(x, ddof=1)),
                   median_se=float(np.median(se)))
        rec.update({'chi2_' + k: v for k, v in constant_chi2(x, se).items()})
        rec.update(fit_intrinsic_scatter(x, se))
        rec['scatter_ratio'] = rec['sigma_int'] / rec['median_se']
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    clean = df[~df.degen]
    out = analyse(clean)
    out.to_csv(sys.argv[2] + '_shape_stats.csv', index=False)
    print(out.to_string(index=False))
