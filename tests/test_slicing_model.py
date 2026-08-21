"""Pin the slicing forward model -- above all, its ANGLE UNITS.

`params_from_principal_axes` takes RADIANS.  Passing degrees is silent: the
sweep wraps and goes non-monotonic, and it produced a whole round of wrong
numbers in this project (predicted 2D spread 0.188/0.204 instead of the correct
0.172/0.192, moving a published sensitivity threshold from 0.27 to 0.30 dex).

The diagnostic that settles it is the endpoint behaviour, which is pure
geometry and needs no data:

    theta = 0    (a1 along W)      -> the dW=0 slice is the a2-a3 plane, so
                                      b2/b1 = a3/a2 EXACTLY, for any roll.
    theta = pi/2 (a1 in the plane) -> the slice contains a1 and one short axis,
                                      so b2/b1 lies between a3/a1 and a2/a1
                                      depending on the roll.

Read as degrees, theta = 90 is ~1.57 full turns and lands nowhere near either.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'analysis'))

from slicing_model import b2b1_at, slice_ratio_from, roll_band  # noqa: E402

A2A1, A3A2 = 0.281, 0.596          # the project's common shape (k=4)
A3A1 = A2A1 * A3A2


def test_face_on_is_exactly_a3_over_a2():
    """theta=0: slice is the a2-a3 plane, so b2/b1 = a3/a2 for EVERY roll."""
    for phi in np.linspace(0, 2 * np.pi, 7):
        for psi in np.linspace(0, 2 * np.pi, 5):
            b = slice_ratio_from(1.0, A2A1, A3A1, 0.0, phi, psi)
            assert abs(b - A3A2) < 1e-9, (phi, psi, b)


def test_edge_on_is_bracketed_by_the_two_short_ratios():
    """theta=pi/2: roll decides which short axis lies in the plane."""
    v = [slice_ratio_from(1.0, A2A1, A3A1, np.pi / 2, f, s)
         for f in np.linspace(0, 2 * np.pi, 40)
         for s in np.linspace(0, 2 * np.pi, 40)]
    v = np.array([x for x in v if np.isfinite(x)])
    assert v.min() > A3A1 - 1e-6, v.min()
    assert v.max() < A2A1 + 1e-6, v.max()
    # both extremes must actually be approached, or the roll is not being used
    assert v.min() < A3A1 * 1.05, v.min()
    assert v.max() > A2A1 * 0.95, v.max()


def test_degrees_would_fail_this():
    """Guard the units: reading theta as degrees must NOT reproduce face-on."""
    b_deg = slice_ratio_from(1.0, A2A1, A3A1, 90.0, 0.3, 0.7)   # 90 rad!
    assert abs(b_deg - A3A2) > 1e-3
    b_rad = slice_ratio_from(1.0, A2A1, A3A1, np.pi / 2, 0.3, 0.7)
    assert abs(b_rad - b_deg) > 1e-3


def test_b2b1_at_takes_degrees_and_matches_geometry():
    """The plotting-facing helper converts for you; endpoints must still hold."""
    _, med0, _ = b2b1_at(0.0, A2A1, A3A2, n=200)
    assert abs(med0 - A3A2) < 1e-6, med0
    lo90, _, hi90 = b2b1_at(90.0, A2A1, A3A2, n=3000)
    assert lo90 > A3A1 - 1e-6 and hi90 < A2A1 + 1e-6


def test_band_width_is_purely_roll_spread():
    """Freezing both rolls must collapse the band to zero width.

    This is the check that no measurement error has leaked into the shaded
    band on the inclination figure -- its width is roll spread ONLY.
    """
    grid = np.linspace(40, 90, 8)
    lo, med, hi = roll_band(A2A1, A3A2, grid, n=1)   # n=1: one roll draw
    assert np.allclose(lo, hi, atol=1e-12)
    assert np.allclose(lo, med, atol=1e-12)


def test_median_curve_is_not_monotonic():
    """Documented behaviour: the roll-marginalised median is NOT monotonic.

    Anything that draws or describes this curve as monotonic is wrong.
    """
    grid = np.linspace(0, 90, 40)
    _, med, _ = roll_band(A2A1, A3A2, grid, n=2000)
    d = np.diff(med)
    assert (d > 0).any() and (d < 0).any(), 'expected a non-monotonic median'
