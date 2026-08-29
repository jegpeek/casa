"""Pin the conventions of results/scale_split_summary*.csv.

That table had no producer in the tree for most of this project's life: it was
built once by an ad-hoc step that was never committed.  The conventions were
later recovered by reverse-engineering the committed table, and
analysis/summarize_scale_split_table.py now reproduces it to float precision.

These tests exist so the recovery cannot be silently lost again.  The angle
convention in particular is easy to "fix" into a signed-rank test by someone
reading only the column name `wilcoxon_p`, which would be wrong: the angles are
unsigned and non-negative by construction, so the meaningful null is isotropy,
not zero.  A wrong test here does not raise -- it just quietly reports a
different p value into a paper section.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

sst = pytest.importorskip('summarize_scale_split_table')


def test_angle_null_is_isotropy_not_zero():
    """P(angle < T) = 1 - cos T for two independent random axes in 3D.

    Verified by simulation rather than asserted, because getting this wrong is
    the failure this whole test file is about.
    """
    rng = np.random.default_rng(0)
    n = 200000
    a = rng.normal(size=(n, 3))
    b = rng.normal(size=(n, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.abs(np.einsum('ij,ij->i', a, b))))
    T = sst.AGREE_DEG
    got = float((ang < T).mean())
    want = 1.0 - np.cos(np.radians(T))
    assert abs(got - want) < 3e-3, (got, want)


def test_angle_row_uses_one_sided_binomial():
    """Perfect agreement must be significant; isotropic input must not be."""
    n = 29
    agree = np.full(n, 2.0)          # all axes agree to 2 deg
    r = sst.angle_row('t', 'ang1', agree)
    assert r['n_positive'] == n
    assert r['wilcoxon_p'] < 1e-20

    # Angles drawn FROM the isotropic null must not look significant.
    rng = np.random.default_rng(1)
    iso = np.degrees(np.arccos(np.abs(rng.uniform(-1, 1, size=n))))
    r_iso = sst.angle_row('t', 'ang1', iso)
    assert r_iso['wilcoxon_p'] > 0.01, r_iso

    # The interval is deliberately blank for angle rows.
    assert np.isnan(r['lo68']) and np.isnan(r['hi68'])


def test_shape_row_is_a_paired_signed_rank_test():
    d = np.array([0.1, 0.2, -0.05, 0.3, 0.15, 0.22, 0.08])
    r = sst.shape_row('t', 'a3a2', d)
    assert r['n'] == d.size
    assert r['n_positive'] == int((d > 0).sum())
    assert r['median_inner_minus_outer'] == pytest.approx(np.median(d))
    assert r['wilcoxon_p'] == pytest.approx(stats.wilcoxon(d).pvalue)
    assert r['lo68'] < r['median_inner_minus_outer'] < r['hi68']


def test_bootstrap_interval_is_deterministic():
    """A fixed seed is what makes the committed table reproducible."""
    d = np.linspace(-0.3, 0.5, 23)
    a = sst.shape_row('t', 'alpha', d)
    b = sst.shape_row('t', 'alpha', d)
    assert (a['lo68'], a['hi68']) == (b['lo68'], b['hi68'])


def _run(script, args, env_var):
    env = dict(os.environ)
    env.pop('CASA_LINEAR_UNITS', None)
    env.pop('CASA_ARCSINH_UNITS', None)
    env[env_var] = '1'
    env['PYTHONPATH'] = _ROOT + os.pathsep + os.path.join(_ROOT, 'analysis')
    return subprocess.run(
        [sys.executable, os.path.join(_ROOT, 'analysis', script)] + args,
        capture_output=True, text=True, env=env, cwd=_ROOT)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_ROOT, 'results', 'scale_split_summary.csv')),
    reason='committed arcsinh summary table not present')
def test_reproduces_committed_arcsinh_table():
    """The check that establishes this is the same computation, not a similar one."""
    p = _run('summarize_scale_split_table.py', ['--check'], 'CASA_ARCSINH_UNITS')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'MATCH' in p.stdout, p.stdout


def test_output_path_follows_the_variant():
    """A raw-flux summary must never land on the arcsinh table's name."""
    import preprocessing_mode as pm
    seen = {}
    for var in ('CASA_ARCSINH_UNITS', 'CASA_LINEAR_UNITS'):
        env = dict(os.environ)
        env.pop('CASA_LINEAR_UNITS', None)
        env.pop('CASA_ARCSINH_UNITS', None)
        env[var] = '1'
        env['PYTHONPATH'] = _ROOT + os.pathsep + os.path.join(_ROOT, 'analysis')
        p = subprocess.run(
            [sys.executable, '-c',
             'import sys; sys.path.insert(0, "analysis");'
             'import preprocessing_mode as pm; print(pm.variant_suffix())'],
            capture_output=True, text=True, env=env, cwd=_ROOT)
        seen[var] = p.stdout.strip()
    assert seen['CASA_ARCSINH_UNITS'] == ''
    assert seen['CASA_LINEAR_UNITS'] == pm.LINEAR_SUFFIX
    assert seen['CASA_ARCSINH_UNITS'] != seen['CASA_LINEAR_UNITS']
