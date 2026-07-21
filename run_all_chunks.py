#!/usr/bin/env python3
"""Run process_chunks on all official chunks (data/chunk_windows.csv) and save
results."""
import sys
import os
sys.path.insert(0, os.path.expanduser('~/projects/util_efs/python'))

import structure_function as sf

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <output.pdf>')
        sys.exit(1)

    pdf_path = sys.argv[1]
    specs = sf.official_windows()
    print(f'Processing {len(specs)} chunks...')

    res = sf.process_chunks(specs)
    print(f'Completed {len(res)} / {len(specs)} chunks')

    for val in res.values():
        sf.save_chunk_result(val['sf'], val['fit'])
    print('Saved all results.')

    sf.make_chunk_plots_pdf(res, pdf_path)
    print(f'PDF written to {pdf_path}')
