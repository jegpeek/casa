"""Tests for power_law_log_s2: the r -> 0 limit of the Weibull profile.

These pin down three properties that the single-band (r < 0.1 ly) analysis
relies on:

  1. weibull_log_s2 -> power_law_log_s2 as r -> 0, with NO beta dependence.
     This is why dropping to 2 parameters is legitimate on a lag range that
     stops well short of the saturation scale a1.
  2. beta = 1 is NOT the power law.  (1 - e^-r)^a != r^a at finite r, so a
     separate profile is genuinely needed rather than a frozen Weibull.
  3. The power law has an exact one-parameter scale degeneracy: scaling every
     geometric axis by k and A by k^alpha leaves log10 S2 unchanged.  Callers
     must freeze A; axis RATIOS survive, absolute a1 does not.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import structure_function as sf  # noqa: E402


def test_weibull_tends_to_power_law_at_small_r():
    """The limit holds, and it is independent of beta."""
    r = np.logspace(-4, -2, 60)
    for beta in (1.5, 2.0, 4.0, 9.0):
        w = sf.weibull_log_s2(r, [0.6, beta, 1.0])
        # var_inf * r^alpha, i.e. the beta-free limit
        pl = sf.power_law_log_s2(r, [0.6, 1.0])
        assert np.abs(w - pl).max() < 1e-3, (beta, np.abs(w - pl).max())


def test_beta_one_is_not_the_power_law():
    """A frozen beta=1 Weibull is a DIFFERENT curve -- hence a real profile."""
    r = np.logspace(-2, -1, 40)
    w1 = sf.weibull_log_s2(r, [0.6, 1.0, 1.0])
    pl = sf.power_law_log_s2(r, [0.6, 1.0])
    dev = np.abs(w1 - pl).max()
    assert dev > 5e-3, dev


def test_power_law_scale_degeneracy_is_exact():
    """Scaling all axes by k and A by k^alpha is an exact symmetry."""
    rng = np.random.default_rng(3)
    lags = rng.normal(0.0, 0.03, size=(400, 3))
    alpha, k = 0.6, 2.7
    kw = dict(theta=1.0, phi=0.4, psi=0.9)
    p1 = sf.params_from_principal_axes(a1=0.8, a2=0.25, a3=0.10, **kw)
    p2 = sf.params_from_principal_axes(a1=0.8 * k, a2=0.25 * k, a3=0.10 * k, **kw)
    v1 = np.array([p1[key] for key in sf._GEOM_KEYS] + [alpha, 1.0])
    v2 = np.array([p2[key] for key in sf._GEOM_KEYS] + [alpha, k ** alpha])
    m1 = sf.log_s2_model(v1, lags, profile=sf.power_law_log_s2)
    m2 = sf.log_s2_model(v2, lags, profile=sf.power_law_log_s2)
    assert np.abs(m1 - m2).max() < 1e-12, np.abs(m1 - m2).max()


def test_power_law_metadata():
    """The pluggable-profile contract fit_s2 relies on."""
    p = sf.power_law_log_s2
    assert p.n_params == 2
    assert p.param_names == ['alpha', 'A']
    assert len(p.default_guess) == 2 and len(p.param_bounds) == 2
    assert 'A' in p.param_names, 'freeze=("A",) must be expressible'
