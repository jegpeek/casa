#!/usr/bin/env python3
"""Run process_chunks on all official chunks (data/chunk_windows.csv) and save
results."""
import sys
import os
sys.path.insert(0, os.path.expanduser('~/projects/util_efs/python'))

import numpy as np
import structure_function as sf

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <output.pdf>')
        sys.exit(1)

    pdf_path = sys.argv[1]
    specs = sf.official_windows()
    print(f'Processing {len(specs)} chunks...')

    # process_chunks streams each sf_fit_*.h5 to data/ and returns the summary
    summary = sf.process_chunks(specs)
    print(f'Completed {len(summary)} / {len(specs)} chunks')

    np.save('data/chunk_summary.npy', summary)
    print('Wrote data/chunk_summary.npy')

    sf.make_chunk_plots_pdf(summary, pdf_path)
    print(f'PDF written to {pdf_path}')
