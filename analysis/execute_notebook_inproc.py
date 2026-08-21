"""Execute a notebook in-process and write the outputs back into the .ipynb.

Why this exists: the usual route (`jupyter nbconvert --execute`, or nbclient)
starts a kernel over a local TCP socket, and this sandbox does not permit
binding one.  This runs every code cell with `exec` in one shared namespace --
the same execution model a kernel provides for a top-to-bottom run -- captures
stdout and any `IPython.display` calls, and stores them as cell outputs.

Limits, stated so nobody mistakes this for a kernel: no interactivity, no
`In[]`/`Out[]` expression echo (only what a cell prints), and cell execution
stops at the first exception, which is reported.  For a linear walkthrough
notebook that is sufficient, and the committed .ipynb ends up with real
outputs rather than empty cells.

    python analysis/execute_notebook_inproc.py notebooks/topline_results.ipynb
"""
import base64
import contextlib
import io
import os
import sys
import traceback

import nbformat as nbf


class _Display:
    """Minimal stand-in for IPython.display that records what a cell shows."""

    def __init__(self):
        self.captured = []

    def _image(self, filename=None, data=None, width=None, **kw):
        if filename is not None:
            with open(filename, 'rb') as fh:
                data = fh.read()
        return ('image/png', base64.b64encode(data).decode('ascii'), width)

    def make_module(self):
        import types
        mod = types.ModuleType('IPython.display')
        sink = self.captured

        class Image:
            def __init__(self, filename=None, data=None, width=None, **kw):
                self.payload = _Display._image(None, filename=filename,
                                               data=data, width=width)

        def display(*objs):
            for o in objs:
                if isinstance(o, Image):
                    sink.append(o.payload)

        mod.Image = Image
        mod.display = display
        return mod


def execute(path):
    nb = nbf.read(path, as_version=4)
    workdir = os.path.dirname(os.path.abspath(path))
    disp = _Display()

    # Make `from IPython.display import Image, display` resolve to the recorder,
    # so figure cells produce embedded PNGs without a live kernel.
    import IPython  # noqa: F401  (ensure the parent package exists)
    sys.modules['IPython.display'] = disp.make_module()

    ns = {'__name__': '__main__', '__builtins__': __builtins__}
    cwd = os.getcwd()
    os.chdir(workdir)
    n_run = n_err = 0
    try:
        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue
            n_run += 1
            disp.captured.clear()
            buf = io.StringIO()
            outputs = []
            try:
                with contextlib.redirect_stdout(buf), \
                        contextlib.redirect_stderr(buf):
                    exec(cell.source, ns)
            except Exception:
                n_err += 1
                outputs.append(nbf.v4.new_output(
                    'error', ename='Error', evalue='see traceback',
                    traceback=traceback.format_exc().splitlines()))
                print('ERROR in cell %d:\n%s' % (n_run, traceback.format_exc()),
                      file=sys.stderr)
            text = buf.getvalue()
            if text:
                outputs.insert(0, nbf.v4.new_output('stream', name='stdout',
                                                    text=text))
            for mime, b64, width in disp.captured:
                meta = {mime: {'width': width}} if width else {}
                outputs.append(nbf.v4.new_output(
                    'display_data', data={mime: b64}, metadata=meta))
            cell.outputs = outputs
            cell.execution_count = n_run
    finally:
        os.chdir(cwd)

    nbf.write(nb, path)
    print('executed %d code cells, %d errors -> %s' % (n_run, n_err, path))
    return n_err


if __name__ == '__main__':
    sys.exit(1 if execute(sys.argv[1]) else 0)
