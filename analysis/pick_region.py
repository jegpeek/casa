#!/usr/bin/env python
"""Interactive polygon picker for defining an interior region of the Cas A
light-echo cloud in the U-V (sky) plane.

Run this in the `casa` conda env on a machine with a display:

    python analysis/pick_region.py                 # default stride 2
    python analysis/pick_region.py --stride 1      # full res (slower)
    python analysis/pick_region.py --out data/interior_region.json

A matplotlib window opens on the 3-colour U-V image (epochs 2/3/4, one global
asinh stretch — the same rendering as knee_field_montage.png).  Click to add
polygon vertices; the outline updates live.

    left-click : add a vertex
    u          : undo the last vertex
    c          : clear all vertices
    Enter      : finish and save
    q / close  : cancel without saving

The polygon is written as full-field pixel coordinates [col, row] (0-indexed,
origin bottom-left, matching numpy row/col after origin='lower') so the
structure-function stage can build a pixel mask directly.
"""
import argparse, json, os, sys
import numpy as np

# Force an INTERACTIVE backend before pyplot is imported.  The casa env ships
# with MPLBACKEND=Agg (non-interactive: draws but runs no event loop, so the
# window ignores clicks/keys).  Prefer macOS-native 'macosx', fall back to Tk.
import matplotlib
os.environ.pop('MPLBACKEND', None)
for _bk in ('macosx', 'tkagg'):
    try:
        matplotlib.use(_bk, force=True)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
print('matplotlib backend:', matplotlib.get_backend())

# repo importable regardless of where python is launched from
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]

DATA_DIR   = os.path.join(REPO, 'data')
RGB_EPOCHS = (2, 3, 4)     # 0-indexed epochs -> R, G, B (matches the montage)
PCT        = (1, 99)       # global percentile clip for the asinh stretch
SOFT       = 0.30          # asinh softening


def load_rgb(stride):
    """Full-field 3-colour image with one global asinh stretch, decimated by
    `stride` for display.  Returns (img[H,W,3], H_full, W_full, stride)."""
    flux_mm = np.load(f'{DATA_DIR}/resampled_epochs_noclip.npy', mmap_mode='r')
    n_full  = flux_mm.shape[1:]                    # (H, W) full field
    chans = []
    for e in RGB_EPOCHS:
        chans.append(np.asarray(flux_mm[e, ::stride, ::stride], dtype=float))
    cube = np.stack(chans, axis=-1)               # (h, w, 3)
    finite = cube[np.isfinite(cube)]
    lo, hi = np.percentile(finite, PCT)
    scaled = (cube - lo) / max(hi - lo, 1e-12)
    img = np.arcsinh(scaled / SOFT) / np.arcsinh(1.0 / SOFT)
    img = np.clip(img, 0, 1)
    img[~np.isfinite(img)] = 0.0
    return img, n_full[0], n_full[1], stride


class PolygonPicker:
    def __init__(self, img, H, W, stride, outpath):
        self.stride, self.outpath = stride, outpath
        self.H, self.W = H, W
        self.verts = []                           # full-field [col, row]
        self.fig, self.ax = plt.subplots(figsize=(11, 11))
        # extent in FULL-field pixel coords so clicks map straight to col/row
        self.ax.imshow(img, origin='lower', interpolation='nearest',
                       extent=[0, W, 0, H])
        self.ax.set_xlabel('U pixel (col)'); self.ax.set_ylabel('V pixel (row)')
        self.ax.set_title('click vertices  |  u=undo  c=clear  Enter=save  q=cancel',
                          fontsize=11)
        (self.line,) = self.ax.plot([], [], '-o', color='cyan', lw=1.6, ms=5,
                                    mfc='white', mec='cyan')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def _redraw(self):
        if self.verts:
            xs = [v[0] for v in self.verts]; ys = [v[1] for v in self.verts]
            if len(self.verts) >= 3:              # show closing edge
                xs = xs + [xs[0]]; ys = ys + [ys[0]]
            self.line.set_data(xs, ys)
        else:
            self.line.set_data([], [])
        self.ax.set_title('%d vertices  |  u=undo  c=clear  Enter=save  q=cancel'
                          % len(self.verts), fontsize=11)
        self.fig.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        self.verts.append([float(ev.xdata), float(ev.ydata)])
        self._redraw()

    def on_key(self, ev):
        if ev.key == 'u' and self.verts:
            self.verts.pop(); self._redraw()
        elif ev.key == 'c':
            self.verts = []; self._redraw()
        elif ev.key in ('enter', 'return'):
            self.save(); plt.close(self.fig)
        elif ev.key == 'q':
            print('cancelled — nothing saved'); plt.close(self.fig)

    def save(self):
        if len(self.verts) < 3:
            print('need >= 3 vertices to define a region; not saving'); return
        out = {
            'coord_system': 'full_field_pixels',
            'note': 'vertices are [col, row] 0-indexed, origin bottom-left '
                    '(numpy array indexing after origin="lower")',
            'field_shape_HxW': [self.H, self.W],
            'display_stride':  self.stride,
            'rgb_epochs':      list(RGB_EPOCHS),
            'vertices_col_row': [[round(c, 2), round(r, 2)] for c, r in self.verts],
        }
        os.makedirs(os.path.dirname(self.outpath), exist_ok=True)
        with open(self.outpath, 'w') as f:
            json.dump(out, f, indent=2)
        print('saved %d-vertex polygon -> %s' % (len(self.verts), self.outpath))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=2,
                    help='display decimation (clicks still map to full-res pixels)')
    ap.add_argument('--out', default=os.path.join(DATA_DIR, 'interior_region.json'))
    a = ap.parse_args()
    img, H, W, stride = load_rgb(a.stride)
    print('field %d x %d  |  displaying at stride %d  (%d x %d)'
          % (H, W, stride, img.shape[0], img.shape[1]))
    # Keep a STRONG reference: matplotlib's callback registry holds only weak
    # refs to the handlers, so without this the picker is garbage-collected the
    # instant this line returns — the window survives but all clicks/keys die.
    picker = PolygonPicker(img, H, W, stride, a.out)
    plt.show()
    return picker


if __name__ == '__main__':
    main()
