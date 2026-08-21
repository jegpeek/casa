"""The geometric slicing forward model: what a fixed 3D shape looks like in 2D.

This is the machinery behind the project's central claim.  A single epoch of
the echo measures the dW=0 central SLICE through the 3D ellipsoid (the Schur
complement -- see `structure_function.principal_axes_2d`), not its projection.
A slice through a fixed triaxial ellipsoid has an in-plane axis ratio b2/b1
that depends on the viewing orientation, so windows that look very different
in the plane of the echo can be the same 3D structure seen at different
angles.

Two orientation angles are unconstrained by the data (the rolls phi and psi);
only theta, the angle of a1 to the echo normal W, is measured.  Every function
here therefore Monte-Carlos phi and psi uniformly and returns percentiles.

UNITS: theta/phi/psi are RADIANS throughout, matching
`structure_function.params_from_principal_axes`.  Passing degrees is a silent
error that produces a non-monotonic, wrapped b2/b1 sweep -- it cost this
project a full round of wrong numbers.  `tests/test_slicing_model.py` pins the
units with an endpoint check: over theta = 0 -> pi/2 the median b2/b1 must run
from exactly a3/a2 (face-on) to a value bracketed by a3/a1 and a2/a1 (edge-on,
roll deciding which short axis lies in the plane).

Sampling: use >= 1500 roll draws.  At 200 the median curve is visibly jagged.
The median is genuinely NON-monotonic in theta -- do not draw or describe it
as monotonic.

Usage
-----
    from slicing_model import b2b1_at, roll_band, slice_ratio_from

    lo, med, hi = b2b1_at(70.0, 0.281, 0.596)      # inclination in DEGREES
    band = roll_band(0.281, 0.596, np.linspace(40, 90, 60))
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import structure_function as sf  # noqa: E402

# Default roll-marginalisation sample size.  200 leaves the median jagged.
NROLL = 4000


def slice_ratio_from(a1, a2, a3, theta, phi, psi):
    """b2/b1 of the dW=0 slice through the ellipsoid at orientation (rad).

    Parameters
    ----------
    a1, a2, a3 : float
        Semi-axes, a1 >= a2 >= a3.  Only ratios matter for the output.
    theta, phi, psi : float [RADIANS]
        theta is the polar angle of a1 from W; phi and psi are the rolls.

    Returns
    -------
    float -- b2/b1 of the in-plane slice, or nan if the slice degenerates.
    """
    p = sf.params_from_principal_axes(a1, a2, a3, theta, phi, psi)
    ax2 = sf.principal_axes_2d(p)
    return ax2['b2'] / ax2['b1'] if ax2['b1'] > 0 else np.nan


def b2b1_at(inc_deg, a2a1, a3a2, n=NROLL, rng=None, pcts=(16, 50, 84)):
    """Percentiles of b2/b1 at one inclination, marginalising both rolls.

    `inc_deg` is in DEGREES (converted internally) because it is the angle we
    plot against; the underlying geometry call takes radians.
    """
    rng = np.random.default_rng(31) if rng is None else rng
    phi = rng.uniform(0, 2 * np.pi, n)
    psi = rng.uniform(0, 2 * np.pi, n)
    th = np.radians(inc_deg)
    v = np.array([slice_ratio_from(1.0, a2a1, a2a1 * a3a2, th, f, s)
                  for f, s in zip(phi, psi)])
    v = v[np.isfinite(v) & (v > 0)]
    return np.percentile(v, list(pcts))


def roll_band(a2a1, a3a2, inc_grid, n=NROLL, seed=31):
    """The 16/50/84 b2/b1 band vs inclination for ONE fixed 3D shape.

    This is the curve overlaid on the b2/b1-vs-inclination figure.  Its width
    is PURELY the spread from the two unconstrained roll angles: it excludes
    measurement error entirely, and any figure showing it must say so.
    Freezing both rolls collapses the width to exactly zero, which is the
    check that no other variance has leaked in.

    The SAME roll draws are reused at every inclination (common random
    numbers).  Drawing fresh rolls per inclination adds independent sampling
    noise to each point on the curve, which at n=4000 is ~2e-3 in b2/b1 --
    about a hundred times the genuine non-monotonic structure of ~2e-5 (see
    test_median_curve_is_not_monotonic) -- so it both roughens the drawn curve
    and buries the real feature in jitter.  Sharing the draws makes differences
    ALONG the curve reflect geometry rather than resampling.
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    psi = rng.uniform(0, 2 * np.pi, n)

    lo, mid, hi = [], [], []
    for th in np.atleast_1d(inc_grid):
        v = np.array([slice_ratio_from(1.0, a2a1, a2a1 * a3a2,
                                       np.radians(th), f, s)
                      for f, s in zip(phi, psi)])
        v = v[np.isfinite(v) & (v > 0)]
        a, b, c = np.percentile(v, [16, 50, 84])
        lo.append(a)
        mid.append(b)
        hi.append(c)
    return np.array(lo), np.array(mid), np.array(hi)


def predicted_spread(inc_deg, a2a1, a3a2, sig21=0.0, sig32=0.0,
                     nrep=400, seed=19):
    """Predicted sd of log10(b2/b1) across a window sample, by forward model.

    Draws the rolls fresh each replicate and, if sig21/sig32 > 0, also scatters
    the 3D axis ratios in log space by that intrinsic scatter (dex).  Compare
    the returned distribution against the OBSERVED sd of log10(b2/b1) to ask
    how much 2D spread a given amount of 3D shape variation would produce.

    With sig21 = sig32 = 0 this is the pure slicing prediction: the spread you
    get from viewing ONE shape at the observed inclinations.

    Returns
    -------
    ndarray of length nrep -- the sd from each replicate.
    """
    inc_deg = np.asarray(inc_deg, float)
    rg = np.random.default_rng(seed)
    out = []
    for _ in range(nrep):
        l21 = np.log10(a2a1) + rg.normal(0, sig21, inc_deg.size)
        l32 = np.log10(a3a2) + rg.normal(0, sig32, inc_deg.size)
        r21 = 10 ** l21
        r32 = 10 ** l32
        phi = rg.uniform(0, 2 * np.pi, inc_deg.size)
        psi = rg.uniform(0, 2 * np.pi, inc_deg.size)
        th = np.radians(inc_deg)
        b = np.array([slice_ratio_from(1.0, x2, x2 * x3, t, f, s)
                      for x2, x3, t, f, s in zip(r21, r32, th, phi, psi)])
        b = b[np.isfinite(b) & (b > 0)]
        out.append(np.std(np.log10(b), ddof=1))
    return np.array(out)


def observed_intrinsic_spread(b2b1, se_b2b1):
    """INTRINSIC sd of log10(b2/b1) across windows, measurement error removed.

    This is the only quantity the forward model may be compared against, and
    getting it wrong is an easy and serious error.  `predicted_spread` returns
    the spread of NOISE-FREE b2/b1 values, so comparing it to the raw observed
    sd double-counts measurement error and understates how well slicing does.

    On the 29 top-quartile windows the raw sd is 0.234 dex with a median
    per-window SE of 0.121 dex, which deflates to an intrinsic 0.185 dex.  The
    raw and intrinsic numbers differ by more than 25%, so the comparison must
    be intrinsic-vs-intrinsic.

    Returns
    -------
    (sig_int, lo, hi) -- ML intrinsic spread and its 1-sigma profile interval.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from shape_center import profile_ci

    b2b1 = np.asarray(b2b1, float)
    se = np.asarray(se_b2b1, float)
    m = np.isfinite(b2b1) & (b2b1 > 0) & np.isfinite(se)
    lv = np.log10(b2b1[m])
    sl = se[m] / b2b1[m] / np.log(10.0)
    sig, lo, hi, _ = profile_ci(lv, sl)
    return sig, lo, hi


def sensitivity_to_3d_scatter(inc_deg, a2a1, a3a2, b2b1, se_b2b1,
                              sig_grid=None, nrep=400, seed=19):
    """At what 3D intrinsic scatter would the 2D spread have been detectable?

    Sweeps an assumed 3D sigma_int through `predicted_spread` and reports where
    the predicted 2D spread departs from the OBSERVED INTRINSIC one by 2 sigma.
    This is the honest statement of the project's central limitation: the
    measured 3D scatter (0.120 dex) sits a factor ~2.3 BELOW this threshold, so
    "the 2D spread is explained by slicing" is true but weak -- 2D never had the
    power to see the real 3D variation that the 3D fits do detect.

    Takes the b2/b1 values and their SEs rather than a precomputed sd, so the
    intrinsic-vs-intrinsic comparison is guaranteed by construction; passing a
    raw sd here was a live footgun.  Both ratios are scattered by the same sig.

    Returns
    -------
    (sig_grid, med_pred, tension, obs_int) -- tension in sigma at each grid
    point, and the observed intrinsic spread it was measured against.
    """
    if sig_grid is None:
        sig_grid = np.linspace(0.0, 0.5, 26)
    obs, olo, ohi = observed_intrinsic_spread(b2b1, se_b2b1)
    obs_se = 0.5 * (ohi - olo)
    med, tens = [], []
    for sg in sig_grid:
        d = predicted_spread(inc_deg, a2a1, a3a2, sig21=sg, sig32=sg,
                             nrep=nrep, seed=seed)
        m = float(np.median(d))
        med.append(m)
        tens.append(abs(m - obs) / obs_se)
    return (np.asarray(sig_grid), np.asarray(med), np.asarray(tens), obs)
