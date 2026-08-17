"""Smoke test: one official chunk through read_window -> compute_s2 -> fit_s2."""
import os
import sys
import time

import numpy as np

import structure_function as sf

if __name__ == '__main__':
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    spec = sf.chunk_window(cid)
    print('chunk', cid, 'window (row, col, size) =', spec)

    t = time.time()
    d = sf.read_window(spec[0], spec[1], spec[2], spec[2])
    print('read_window        %6.1fs' % (time.time() - t))
    print('  flux_epochs', d['flux_epochs'].shape,
          'finite frac %.3f' % np.mean(np.isfinite(d['flux_epochs'])))
    print('  W_values', d['W_values'])
    print('  U range %.3f .. %.3f ly' % (d['U_grid'].min(), d['U_grid'].max()))

    t = time.time()
    s = sf.compute_s2(d)
    print('compute_s2         %6.1fs' % (time.time() - t))
    print('  s2', s['s2'].shape, 'lag_du %.3f .. %.3f ly'
          % (s['lag_du'][0], s['lag_du'][-1]))

    t = time.time()
    f = sf.fit_s2(s)
    print('fit_s2             %6.1fs' % (time.time() - t))
    for k, v in f.items():
        if np.isscalar(v):
            print('  %-16s %s' % (k, v))
    a1, a2, a3 = f.get('a1'), f.get('a2'), f.get('a3')
    if a1 is not None:
        print('  axes (ly)        %.4f %.4f %.4f' % (a1, a2, a3))
