"""The common shape: where its centre is, and how much real scatter is around it.

Two questions that must not be conflated, because they want DIFFERENT centres:

  * "What is the typical shape?"  -- wants an UNBIASED centre.  Inverse-variance
    weighting in log space is biased on these data, because windows with larger
    a2/a1 have smaller log-errors (rho = -0.36), which pulls the weighted mean
    high.  Use the median, or the ML centre fitted jointly with intrinsic
    scatter (`ml_center_and_scatter`), which downweights nothing systematically.

  * "Is one exact shape consistent with the data?" -- wants the chi^2-MINIMISING
    centre.  Testing a common value against a non-minimising centre spuriously
    rejects it.  This trap was hit in this project.

The model throughout is

    x_i = mu + N(0, sigma_int^2) + N(0, se_i^2)          [x in log10 space]

sigma_int is the INTRINSIC scatter: real window-to-window shape variation, over
and above measurement error.  This project's result is that sigma_int is
non-zero -- 0.120 dex in a2/a1 and 0.096 dex in a3/a2 -- so the honest headline
is NOT "the shapes are identical" but "the shapes differ by a measured 25-32%,
and most of the apparent spread is measurement noise".

Do NOT derive an error-inflation factor from sqrt(chi^2/n) of the S2 fits and
apply it here.  Those residuals are weighted by 1/r, not by a measurement
error, so that chi^2 is not a reduced chi^2 and its root is not an inflation
factor.  Inflating with a factor derived from the same chi^2 you then test with
is circular and returns p ~ 0.46 by construction.  Estimate sigma_int
independently instead -- that is what this module is for.

Usage
-----
    from shape_center import ml_center_and_scatter, profile_ci, common_value_test

    mu, sig = ml_center_and_scatter(np.log10(d.a2a1), d.se_a2a1 / d.a2a1 / np.log(10))
    sig, lo, hi, dchi2 = profile_ci(lv, sl)
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats


def weighted_center(lv, sl, sig=0.0):
    """Inverse-variance centre in log space, including intrinsic scatter.

    With sig=0 this is the plain inverse-variance mean -- the NULL model, in
    which all scatter is measurement error.  Note the returned SE is the SE on
    the centre, not the scatter of the population.
    """
    lv = np.asarray(lv, float)
    sl = np.asarray(sl, float)
    w = 1.0 / (sl ** 2 + sig ** 2)
    mu = float(np.sum(w * lv) / np.sum(w))
    return mu, float(1.0 / np.sqrt(np.sum(w)))


def _nll(ls, lv, sl):
    """-log L of (mu profiled out, sigma_int = 10**ls)."""
    s2t = sl ** 2 + (10 ** ls) ** 2
    w = 1.0 / s2t
    mu = np.sum(w * lv) / np.sum(w)
    return 0.5 * np.sum((lv - mu) ** 2 / s2t + np.log(s2t))


def ml_center_and_scatter(lv, sl, bounds=(-4, 1)):
    """Joint ML centre and intrinsic scatter in log space.

    Returns
    -------
    (mu, sigma_int) -- both in dex.  mu is the common log10 value; sigma_int is
    the real window-to-window scatter with measurement error removed.
    """
    lv = np.asarray(lv, float)
    sl = np.asarray(sl, float)
    r = minimize_scalar(_nll, bounds=bounds, method='bounded',
                        args=(lv, sl))
    sig = float(10 ** r.x)
    mu, _ = weighted_center(lv, sl, sig)
    return mu, sig


def profile_ci(lv, sl, bounds=(-4, 1), ngrid=4001):
    """Profile-likelihood interval on sigma_int, and the test against zero.

    Returns
    -------
    sig      : ML intrinsic scatter [dex]
    lo, hi   : 1-sigma profile interval (delta-chi2 <= 1, one parameter)
    dchi2_0  : delta-chi2 of sigma_int = 0 against the ML value.  This is the
               test that there IS real shape variation; compare to chi2_1.
    """
    lv = np.asarray(lv, float)
    sl = np.asarray(sl, float)
    r = minimize_scalar(_nll, bounds=bounds, method='bounded', args=(lv, sl))
    n0 = _nll(r.x, lv, sl)
    g = np.linspace(bounds[0], bounds[1], ngrid)
    dn = 2 * np.array([_nll(x, lv, sl) for x in g]) - 2 * n0
    ok = dn <= 1.0
    return (float(10 ** r.x), float(10 ** g[ok].min()),
            float(10 ** g[ok].max()), float(dn[0]))


def zero_scatter_pvalue(lv, sl):
    """p-value for "sigma_int = 0" via the profile likelihood ratio (1 dof)."""
    _, _, _, dchi2 = profile_ci(lv, sl)
    return float(stats.chi2.sf(dchi2, 1))


def common_value_test(lv, sl, sig=0.0):
    """chi^2 of one common value, at the chi^2-MINIMISING centre.

    Use this for the hypothesis test, never with an externally chosen centre.
    Pass sig>0 to test "one shape plus this much intrinsic scatter".
    """
    lv = np.asarray(lv, float)
    sl = np.asarray(sl, float)
    mu, _ = weighted_center(lv, sl, sig)
    chi2 = float(np.sum((lv - mu) ** 2 / (sl ** 2 + sig ** 2)))
    dof = lv.size - 1
    return chi2, dof, float(stats.chi2.sf(chi2, dof))


def prolateness(a2a1, a3a2):
    """log-symmetric prolate/oblate measure: log10(a3/a2) - log10(a2/a1).

    Replaces triaxiality T, which is the wrong statistic for this cloud: T
    saturates at a2/a1 ~ 0.29 (every window lands at T ~ 0.94), responds ~4x
    more strongly to the SCALE-INVARIANT ratio a2/a1 than to the varying
    a3/a2, and returns exactly 1.0 for degenerate fits with a3 -> 0 (maximum
    confidence in "prolate" from a collapsed fit).

    This measure is scale-free, non-saturating, weights the two log ratios
    equally, and in the log-log shape plane its zero is exactly the y = x
    diagonal.  > 0 prolate, = 0 maximally triaxial, < 0 oblate.

    CAVEAT: the SE on this quantity has been computed by adding se_a2a1 and
    se_a3a2 in quadrature, which assumes they are uncorrelated.  They come from
    the same fit and the jackknife samples give the true covariance (median
    rho = -0.43 per window).  Do that properly before publishing the
    significance.
    """
    return np.log10(np.asarray(a3a2, float)) - np.log10(np.asarray(a2a1, float))


def mahalanobis_2d(a2a1, a3a2, se21, se32, c21, c32, sig21=0.0, sig32=0.0):
    """Per-window distance in the log-log shape plane, in units of its own error.

    The right test statistic for "is this distribution consistent with one
    shape (plus intrinsic scatter)": compare the MEDIAN of the returned d
    against the chi2_2 median of 1.386, or KS-test the whole set against chi2_2.

    Do NOT use "fraction of windows inside their own 1-sigma ellipse" instead.
    Per-window SEs span a factor of 19 here, so large-SE windows fall inside by
    construction: the measurement-only null holds 34% of windows within 1 sigma
    while being rejected at median d = 2.45.
    """
    lu = np.log10(np.asarray(a2a1, float)) - np.log10(c21)
    lv = np.log10(np.asarray(a3a2, float)) - np.log10(c32)
    su = np.hypot(np.asarray(se21, float), sig21)
    sv = np.hypot(np.asarray(se32, float), sig32)
    return np.hypot(lu / su, lv / sv)
