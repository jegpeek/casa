#!/usr/bin/env python3
"""Run process_chunks on all uvw_chunk_*_products.h5 files and save results."""
import glob
import sys
import os
sys.path.insert(0, os.path.expanduser('~/projects/util_efs/python'))

import structure_function as sf

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <output.pdf>')
        sys.exit(1)

    pdf_path = sys.argv[1]
    fns = sorted(glob.glob('data/uvw_chunk_*_products.h5'))
    fns = [f for f in fns if 'sf_angular' not in f]
    print(f'Processing {len(fns)} chunks...')

    res = sf.process_chunks(fns)
    print(f'Completed {len(res)} / {len(fns)} chunks')

    for fn, val in res.items():
        sf.save_chunk_result(fn, val['sf'], val['fit'])
    print('Saved all results.')

    sf.make_chunk_plots_pdf(res, pdf_path)
    print(f'PDF written to {pdf_path}')
