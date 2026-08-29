"""Single source of truth for which preprocessing the pipeline is running in.

RAW FLUX is the default.  The arcsinh transform applied by the original runs is
dynamic-range compression with no measurement-model justification, and every
conclusion survives without it (see README_FORK.md), so the untransformed data
is what the analysis now uses by default.

Two things are selected by this choice and they must never disagree:

  * the COMPUTE settings handed to the structure-function code
    (`analysis/scale_split.py`: COMPUTE_KW), and
  * the filename suffix of the tables and figures
    (`make_tier_figures.default_variant()` -> `_linear` or '').

They live in different layers -- one is imported by the compute drivers, the
other by figure code that must stay importable without h5py or util_efs for
tier-A reproduction -- so the RESOLUTION lives here, in a module with no
dependencies beyond the standard library, and both layers import it.  Before
this module existed the same env-var logic was written out twice, which agreed
by inspection and had no alarm if it ever stopped agreeing.

This is deliberately an environment variable rather than a function argument:
the drivers reach the compute settings through module state inside `spawn`-ed
pool workers, which re-import the module and would not observe a mutation made
in the parent process.
"""

import os

#: Filename suffix used by the raw-flux (default) variant.  NB the suffix was
#: deliberately NOT inverted when raw flux became the default: unsuffixed
#: committed files remain the arcsinh run they have always been, so no tracked
#: path silently changes meaning between commits.
LINEAR_SUFFIX = '_linear'
ARCSINH_SUFFIX = ''


def use_linear_units():
    """True when the pipeline should run in raw flux units (the default).

    `CASA_ARCSINH_UNITS=1` selects the original arcsinh preprocessing.  The
    legacy `CASA_LINEAR_UNITS=0` does the same, so scripts written before raw
    flux became the default keep working.  If both are set, arcsinh wins: an
    explicit request for the non-default variant should never be silently
    overridden by a stale variable left in the environment.
    """
    if os.environ.get('CASA_ARCSINH_UNITS', '') not in ('', '0'):
        return False
    return os.environ.get('CASA_LINEAR_UNITS', '1') not in ('', '0')


def variant_suffix():
    """Filename suffix for the current variant: '_linear' or ''."""
    return LINEAR_SUFFIX if use_linear_units() else ARCSINH_SUFFIX


def windows_file(which, root='.'):
    """Path to a window list, variant-aware for the SNR-selected lists.

    `q4` is the top signal-to-noise QUARTILE, so its membership depends on the
    preprocessing: the arcsinh and raw-flux quartiles share 28 of 29 windows but
    each contains one the other does not (arcsinh r1600_c2400, raw flux
    r2000_c4000).  Reading the wrong one silently bootstraps a window the
    tiering excludes and omits one it includes.

    `all`/`all115` is the full grid and so is variant-independent.
    """
    if which in ('all', 'all115'):
        return os.path.join(root, 'handoff', 'all115_windows.json')
    path = os.path.join(
        root, 'handoff', '%s_windows%s.json' % (which, variant_suffix()))
    if not os.path.exists(path):
        raise SystemExit(
            'missing %s -- the %s list is SNR-selected and so is specific to '
            'this preprocessing variant; regenerate it rather than falling '
            'back to the other variant\'s list.' % (path, which))
    return path
