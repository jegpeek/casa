"""Execute a notebook in-process and write the outputs back into the .ipynb.

Why this exists: the usual route (`jupyter nbconvert --execute`, or nbclient)
starts a kernel over a local TCP socket, and this sandbox does not permit
binding one.  This runs every code cell with `exec` in one shared namespace --
the same execution model a kernel provides for a top-to-bottom run -- captures
stdout and any `IPython.display` calls, and stores them as cell outputs.

A cell's trailing bare expression is echoed the way a kernel echoes `Out[]`: a
matplotlib Figure is rendered to PNG, and anything else becomes an
`execute_result` carrying `text/plain` plus `text/html` when the object provides
`_repr_html_`, so a cell ending in a bare DataFrame renders as a table.

Limits, stated so nobody mistakes this for a kernel: no interactivity, only the
*final* expression of a cell is echoed (a kernel echoes nothing else either),
and cell execution stops at the first exception, which is reported.  For a
linear walkthrough notebook that is sufficient, and the committed .ipynb ends
up with real outputs rather than empty cells.

    python analysis/execute_notebook_inproc.py notebooks/topline_results.ipynb
"""
import base64
import contextlib
import io
import os
import sys
import traceback

import nbformat as nbf


def _repr_bundle(value):
    """MIME bundle for a cell's trailing expression, as a kernel would build it.

    Includes text/html when the object offers `_repr_html_` (pandas objects do),
    so a cell ending in a bare DataFrame renders as a real table.
    """
    data = {'text/plain': repr(value)}
    html = getattr(value, '_repr_html_', None)
    if callable(html):
        try:
            out = html()
        except Exception:
            out = None
        if isinstance(out, str):
            data['text/html'] = out
    return data


class _Display:
    """Minimal stand-in for IPython.display that records what a cell shows."""

    def __init__(self):
        self.captured = []
        self.results = []

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


def _exec_cell(source, ns, disp):
    """exec a cell, then render a trailing bare expression the way Jupyter does.

    A cell ending in `fig` displays that figure under a real kernel.  Plain
    exec() discards the value, so without this the notebook silently loses the
    figures whose cells end that way -- the failure mode is a missing image, not
    an error, which is exactly the kind of thing that ships unnoticed.
    """
    import ast

    tree = ast.parse(source)
    tail = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tail = ast.Expression(tree.body.pop().value)
    exec(compile(tree, '<cell>', 'exec'), ns)
    if tail is None:
        return
    value = eval(compile(tail, '<cell>', 'eval'), ns)
    if value is None:
        return
    fig = getattr(value, 'savefig', None)
    if fig is not None:                      # a matplotlib Figure
        buf = io.BytesIO()
        value.savefig(buf, format='png', dpi=110, bbox_inches='tight')
        disp.captured.append(('image/png',
                              base64.b64encode(buf.getvalue()).decode(),
                              None))
        return
    # Anything else is an Out[] value: echo it the way a kernel does, so a cell
    # ending in a bare DataFrame renders as a table instead of showing nothing.
    disp.results.append(_repr_bundle(value))


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
            disp.results.clear()
            buf = io.StringIO()
            outputs = []
            try:
                with contextlib.redirect_stdout(buf), \
                        contextlib.redirect_stderr(buf):
                    _exec_cell(cell.source, ns, disp)
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
            for data in disp.results:
                outputs.append(nbf.v4.new_output(
                    'execute_result', data=data, metadata={},
                    execution_count=n_run))
            cell.outputs = outputs
            cell.execution_count = n_run
    finally:
        os.chdir(cwd)

    nbf.write(nb, path)
    print('executed %d code cells, %d errors -> %s' % (n_run, n_err, path))
    return n_err


if __name__ == '__main__':
    sys.exit(1 if execute(sys.argv[1]) else 0)
