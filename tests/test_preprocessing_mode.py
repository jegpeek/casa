"""Pin the preprocessing switch: the two layers must never disagree.

Which preprocessing is running is consumed in two places that live in different
import layers:

    analysis/scale_split.py       COMPUTE_KW      -> what the fits COMPUTE
    make_tier_figures.py          default_variant -> which FILES are read/written

If those ever disagree, the failure is silent and produces a genuinely wrong
measurement rather than an error: raw-flux fits tiered by arcsinh signal-to-noise
(the SNR is an S2 plateau-over-floor ratio, and the arcsinh compresses the
plateau relative to the floor asymmetrically), or a figure built from one
variant's table and labelled with the other's. Both now resolve through
analysis/preprocessing_mode.py; this test is the alarm if a future edit
reintroduces a second copy of the logic.

The layers are checked in SUBPROCESSES because the switch is read at import
time -- mutating os.environ in-process would not reach an already-imported
module, which is exactly why it is an env var rather than an argument (the
compute drivers re-import inside `spawn`-ed pool workers).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (env overrides, expect_linear_units)
CASES = [
    ({}, True),                                          # raw flux is DEFAULT
    ({'CASA_ARCSINH_UNITS': '1'}, False),
    ({'CASA_LINEAR_UNITS': '0'}, False),                 # legacy spelling
    ({'CASA_LINEAR_UNITS': '1'}, True),
    # An explicit request for the non-default variant must win over a stale
    # variable left in the environment.
    ({'CASA_LINEAR_UNITS': '1', 'CASA_ARCSINH_UNITS': '1'}, False),
]

_PROBE = r"""
import json
import scale_split as ss
import make_tier_figures as mtf
import preprocessing_mode as pm
print(json.dumps({
    'compute_linear': ss.LINEAR_UNITS,
    'compute_arcsinh_scale': ss.COMPUTE_KW['arcsinh_scale'],
    'compute_background': ss.COMPUTE_KW['background'],
    'suffix': mtf.default_variant(),
    'shared_linear': pm.use_linear_units(),
    'shared_suffix': pm.variant_suffix(),
}))
"""


def _probe(overrides):
    env = dict(os.environ)
    for k in ('CASA_LINEAR_UNITS', 'CASA_ARCSINH_UNITS'):
        env.pop(k, None)
    env.update(overrides)
    env['PYTHONPATH'] = os.pathsep.join(
        [ROOT, os.path.join(ROOT, 'analysis'), env.get('PYTHONPATH', '')])
    out = subprocess.run([sys.executable, '-c', _PROBE], capture_output=True,
                         text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    import json
    return json.loads(out.stdout)


def test_layers_agree_in_every_case():
    for overrides, expect_linear in CASES:
        r = _probe(overrides)
        label = overrides or '(unset: the default)'
        assert r['compute_linear'] == expect_linear, (
            'compute layer picked the wrong variant for %s' % (label,))
        assert r['shared_linear'] == expect_linear, (
            'shared resolver disagrees with expectation for %s' % (label,))
        # The whole point: suffix and compute settings move TOGETHER.
        assert r['suffix'] == ('_linear' if expect_linear else ''), (
            'filename suffix does not match the compute variant for %s'
            % (label,))
        assert r['suffix'] == r['shared_suffix'], (
            'figure layer and shared resolver disagree for %s' % (label,))


def test_compute_settings_match_the_variant():
    """Raw flux means BOTH the transform and the floor are off.

    These are one coupled choice, not two knobs: with arcsinh_scale=None the
    chain reduces to (f - background), and S2 is a function of pixel
    DIFFERENCES, so a constant offset cancels exactly.  What the arcsinh
    actually receives in the original runs is x = (f - 0.03)/0.03, which is why
    the two published values are the same number.
    """
    lin = _probe({})
    assert lin['compute_arcsinh_scale'] is None
    assert lin['compute_background'] == 0.0

    arc = _probe({'CASA_ARCSINH_UNITS': '1'})
    assert arc['compute_arcsinh_scale'] == 0.03
    assert arc['compute_background'] == 0.03


def test_default_is_raw_flux():
    """Guard the DEFAULT itself, so flipping it back is never accidental."""
    assert _probe({})['compute_linear'] is True
    assert _probe({})['suffix'] == '_linear'


_PATHS_PROBE = r"""
import json, os
import preprocessing_mode as pm
print(json.dumps({
    'suffix': pm.variant_suffix(),
    'q4': os.path.basename(pm.windows_file('q4', '.')),
    'all': os.path.basename(pm.windows_file('all', '.')),
}))
"""


def _paths(overrides):
    env = dict(os.environ)
    for k in ('CASA_LINEAR_UNITS', 'CASA_ARCSINH_UNITS'):
        env.pop(k, None)
    env.update(overrides)
    env['PYTHONPATH'] = os.pathsep.join(
        [ROOT, os.path.join(ROOT, 'analysis'), env.get('PYTHONPATH', '')])
    out = subprocess.run([sys.executable, '-c', _PATHS_PROBE],
                         capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    import json
    return json.loads(out.stdout)


def test_variant_trees_are_disjoint():
    """A raw-flux run must never write into the arcsinh output tree.

    The arcsinh caches under data/ are UNTRACKED, so an overwrite is
    unrecoverable -- the committed tables could not be rebuilt.  The arcsinh
    variant must keep the historical unsuffixed paths (nothing already on disk
    changes meaning) and raw flux must be suffixed.
    """
    assert _paths({})['suffix'] == '_linear'
    assert _paths({'CASA_ARCSINH_UNITS': '1'})['suffix'] == ''


def test_q4_window_list_follows_the_variant():
    """q4 is SNR-selected, so its membership is variant-specific.

    The two quartiles share 28 of 29 windows but each holds one the other does
    not, so reading the wrong list silently bootstraps a window this variant's
    tiering excludes.  `all` is the full grid and must NOT be suffixed.
    """
    assert _paths({})['q4'] == 'q4_windows_linear.json'
    assert _paths({'CASA_ARCSINH_UNITS': '1'})['q4'] == 'q4_windows.json'
    for ov in ({}, {'CASA_ARCSINH_UNITS': '1'}):
        assert _paths(ov)['all'] == 'all115_windows.json'


def test_figure_layer_stays_importable_without_bulk_deps():
    """Tier-A reproduction must work from a clone with no data and no util_efs.

    make_tier_figures is deliberately isolated from structure_function (which
    imports h5py and reaches for util_efs), so default_variant() must not drag
    those in -- that is why the shared resolver has no dependencies beyond the
    standard library.
    """
    probe = (
        'import sys, make_tier_figures as mtf;'
        'mtf.default_variant();'
        'heavy = [m for m in ("structure_function", "h5py", "util_efs")'
        ' if m in sys.modules];'
        'print(",".join(heavy))'
    )
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [ROOT, os.path.join(ROOT, 'analysis'), env.get('PYTHONPATH', '')])
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                         text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '', (
        'tier-A import pulled in bulk deps: %s' % out.stdout.strip())


if __name__ == '__main__':
    test_layers_agree_in_every_case()
    test_compute_settings_match_the_variant()
    test_default_is_raw_flux()
    test_variant_trees_are_disjoint()
    test_q4_window_list_follows_the_variant()
    test_figure_layer_stays_importable_without_bulk_deps()
    print('preprocessing mode: all invariants hold')
