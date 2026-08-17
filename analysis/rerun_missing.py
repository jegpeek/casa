"""Re-run only the official windows that have no result file in save_dir yet,
then rebuild the summary and PDF from every result on disk.

Usage: rerun_missing.py <output.pdf> [save_dir]
"""
import os
import sys

import numpy as np

import structure_function as sf

if __name__ == '__main__':
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'chunk_fits.pdf'
    save_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/sf_fits'

    specs = sf.official_windows()
    missing = [s for s in specs
               if not os.path.exists(sf.window_result_path(*s, save_dir))]
    print(f'{len(specs) - len(missing)} / {len(specs)} already on disk; '
          f'running {len(missing)} missing windows')

    if missing:
        sf.process_chunks(missing, save_dir=save_dir)

    paths = sf.result_paths(save_dir)
    print(f'{len(paths)} result files on disk; building summary')
    summary = sf.summarize_chunks(paths, save_dir=save_dir)
    np.save(f'{save_dir}/chunk_summary.npy', summary)
    print(f'Wrote {save_dir}/chunk_summary.npy with {len(summary)} rows')

    sf.make_chunk_plots_pdf(summary, pdf_path)
    print(f'PDF written to {pdf_path}')
