"""Regenerate the per-window SNR tiering table from the raw arrays.

`noise_audit.py`'s own `__main__` reads `data/sf_fits/*.h5`, a cache of
`compute_s2` outputs that is not tracked in the fork, so it cannot be used to
re-tier under a different preprocessing choice.  This driver goes through
`noise_audit.audit_window`, which calls `read_window` + `compute_s2` directly
and is documented to reproduce `results/noise_audit_table.csv` to 5+ decimals.

The point of it is that the published tiering is computed in ARCSINH units:
`snr` is the S2 plateau-over-floor excess, and the arcsinh compresses the
plateau relative to the floor, so the tier boundaries are not flux SNR.  With
CASA_LINEAR_UNITS=1 this writes the same statistic measured on raw flux.

    python analysis/noise_audit_windows.py [--procs 4]

Output: results/noise_audit_table_linear.csv (linear mode) or
        results/noise_audit_table_regen.csv (default mode, for validation
        against the published table -- it never overwrites it).
"""
import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

import noise_audit as na  # noqa: E402
import scale_split as ss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--size', type=int, default=400)
    args = ap.parse_args()

    w = pd.read_csv(os.path.join(_ROOT, 'data', 'chunk_windows.csv'))
    suffix = '_linear' if ss.LINEAR_UNITS else '_regen'
    cache = os.path.join(_ROOT, 'data', 'noise_audit_cache' + suffix)
    specs = [(int(r.row), int(r.col), args.size,
              os.path.join(_ROOT, 'data'), cache) for r in w.itertuples()]

    print('%d windows, compute_kw=%s' % (len(specs), ss.COMPUTE_KW), flush=True)
    rows = []
    with Pool(args.procs) as pool:
        for i, (rec, status) in enumerate(
                pool.imap_unordered(na.audit_window, specs), 1):
            rows.append(rec)
            if i % 10 == 0:
                print('  %d/%d %s' % (i, len(specs), status), flush=True)

    out = pd.DataFrame(rows).sort_values(['row', 'col'])
    path = os.path.join(_ROOT, 'results', 'noise_audit_table%s.csv' % suffix)
    out.to_csv(path, index=False)
    print('wrote %s (%d rows, %d finite snr)'
          % (path, len(out), int(out.snr.notna().sum())))


if __name__ == '__main__':
    main()
