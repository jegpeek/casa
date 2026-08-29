"""Invariants of the Science shape-plane figure's caption statistics.

These pin two things that were briefly stated wrongly in the source and that
an arcsinh-only check cannot catch, because under arcsinh the three
top-quartile counts happen to coincide (29/29/28) and the bug is invisible.
Every test here therefore runs BOTH preprocessing variants.

The invariants:

1. Coverage is measured on the non-degenerate top quartile, NOT on the drawn
   set.  This is the project's standing rule -- a quoted statistic must not
   move when the display floor is retuned.  Test 1 retunes the floor hard and
   demands the coverage fractions do not budge.

2. The caption's stated denominator must equal the sample coverage was
   actually computed on.  The three counts (fitted / non-degenerate / drawn)
   are distinct under raw flux, so describing the denominator as "inside the
   plotted axes" or as the fitted quartile is false there.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'analysis')):
    if p not in sys.path:
        sys.path.insert(0, p)

VARIANTS = ('', '_linear')


def _data(variant, floor=None):
    import make_science_figure as m
    import make_tier_figures as mtf
    kw = {} if floor is None else {'floor': floor}
    return m.shape_plane_data(variant=variant, **kw)


def _have(variant):
    """Skip loudly if this variant's input table is absent."""
    try:
        _data(variant)
    except Exception as exc:                                # pragma: no cover
        pytest.skip('variant %r unavailable: %s' % (variant, exc))


@pytest.mark.parametrize('variant', VARIANTS)
def test_coverage_is_floor_independent(variant):
    """Retuning the display floor must not move a quoted coverage fraction."""
    _have(variant)
    base = _data(variant)
    for floor in (0.02, 0.15, 0.25):
        alt = _data(variant, floor=floor)
        assert alt['cov_meas'] == base['cov_meas'], (
            'measurement coverage moved with floor=%s under variant %r'
            % (floor, variant))
        assert alt['cov_tot'] == base['cov_tot'], (
            'total coverage moved with floor=%s under variant %r'
            % (floor, variant))


@pytest.mark.parametrize('variant', VARIANTS)
def test_caption_denominator_matches_coverage_sample(variant):
    """The number the caption quotes must be the denominator actually used."""
    _have(variant)
    S = _data(variant)
    assert S['cov_meas'][1] == S['n_q4_cov']
    assert S['cov_tot'][1] == S['n_q4_cov']
    # and it must appear in the caption prose
    import make_science_figure as m
    txt = m.caption(S)
    assert str(S['n_q4_cov']) in txt


@pytest.mark.parametrize('variant', VARIANTS)
def test_coverage_denominator_is_nondegenerate_not_drawn(variant):
    """Pin WHICH sample the denominator is, distinguishing all three counts."""
    _have(variant)
    S = _data(variant)
    d = S['d']
    q4 = d[d.tier == 'q4']
    lo = S['floor']
    n_drawn = int(((q4.a2a1 >= lo) & (q4.a3a2 >= lo)).sum())
    assert S['n_q4_cov'] == len(q4), 'denominator is the non-degenerate quartile'
    assert S['n_q4_cov'] <= S['n_q4_fit']
    assert n_drawn <= S['n_q4_cov']


def test_raw_flux_actually_separates_the_three_counts():
    """Guard the guard: under raw flux the counts must genuinely differ.

    If this ever collapses to a single number, the tests above stop being able
    to catch a denominator mix-up and the arcsinh-only blind spot is back.
    """
    _have('_linear')
    S = _data('_linear')
    d = S['d']
    q4 = d[d.tier == 'q4']
    lo = S['floor']
    n_drawn = int(((q4.a2a1 >= lo) & (q4.a3a2 >= lo)).sum())
    assert S['n_q4_fit'] > S['n_q4_cov'] > n_drawn, (
        'expected fitted > non-degenerate > drawn, got %d, %d, %d'
        % (S['n_q4_fit'], S['n_q4_cov'], n_drawn))
