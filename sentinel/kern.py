"""One scalar source, two compile targets.

A ``@spec`` function is the single definition of a bit-exact kernel.  On the CPU it is
compiled in place by ``numba.njit``; :func:`build` recompiles the same code object for
``"cuda"`` as a device function whose callees resolve to the CUDA builds of the other
specs.  Only the launch layer (:mod:`sentinel.kern_cuda` and the ``prange`` loops of the
callers) differs per target, so no second implementation can drift.  The subset a spec may
use is enforced by ``sentinel/tests/test_kern.py``.
"""

import functools
import importlib.util
import os
import sys
import types

import numpy as np

from sentinel import jitcache

jitcache.install()  # must precede numba: the cache key carries the kernel sources

try:
    import numba
    from numba.core.dispatcher import Dispatcher
except ImportError:  # pragma: no cover - numba absent -> specs stay plain Python
    numba = None
    Dispatcher = ()

BACKENDS = ("cpu", "cuda")
_SPECS = {}
TABLE_BYTES = 32
_DEVICE_TABLES = {}


def spec(fn=None, *, inline=False):
    """Register ``fn`` as a spec and return its CPU build (``fn`` itself without numba).

    ``inline`` keeps numba's IR-level ``inline="always"`` for the CPU build."""

    def wrap(f):
        if numba is None:  # pragma: no cover
            return f
        disp = numba.njit(cache=True, inline="always" if inline else "never")(f)
        _SPECS[disp] = f
        return disp

    return wrap if fn is None else wrap(fn)


def specs(module=None):
    """The registered ``{name: dispatcher}``, optionally of one module only."""
    return {
        d.py_func.__name__: d
        for d in _SPECS
        if module is None or d.py_func.__module__ == module
    }


@functools.cache
def _cuda_module(name):
    """``{spec name: CUDA device function}`` for module ``name``.

    The twins share one globals dict (a copy of the module's) holding the CUDA builds of
    every spec, so callees resolve to the same target; specs imported from other modules
    are swapped for their CUDA builds too."""
    from numba import cuda  # pylint: disable=import-outside-toplevel

    scope = dict(vars(sys.modules[name]))
    for key, val in scope.items():
        if isinstance(val, Dispatcher) and val in _SPECS:
            if val.py_func.__module__ != name:
                scope[key] = _cuda_module(val.py_func.__module__)[val.py_func.__name__]
        elif (
            scope.get("DEVICE_TABLES")
            and isinstance(val, np.ndarray)
            and val.nbytes > TABLE_BYTES
        ):
            scope[key] = _device_table(val)
    mine = {}
    for disp in _SPECS:
        fn = disp.py_func
        if fn.__module__ == name:
            twin = types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__)
            twin.__qualname__ = fn.__qualname__
            mine[fn.__name__] = scope[fn.__name__] = cuda.jit(device=True)(twin)
    return mine


def _device_table(table):
    """``table`` as one device array shared by every spec that reads it: a table past
    :data:`TABLE_BYTES` would otherwise be frozen into each function as constant data,
    which the device caps at 64 KB per kernel."""
    from numba import cuda  # pylint: disable=import-outside-toplevel

    key = id(table)
    if key not in _DEVICE_TABLES:
        _DEVICE_TABLES[key] = (table, cuda.to_device(table))
    return _DEVICE_TABLES[key][1]


def _cuda_build(disp):
    return _cuda_module(disp.py_func.__module__)[disp.py_func.__name__]


def build(fn, target):
    """``fn`` (a spec) compiled for ``target`` in :data:`BACKENDS`."""
    if fn not in _SPECS:
        raise ValueError(f"{fn!r} is not a spec")
    if target == "cpu":
        return fn
    if target == "cuda":
        return _cuda_build(fn)
    raise ValueError(f"unknown target {target!r}")


def _touch(cell):
    cell[0] = 1


@functools.cache
def cuda_available():
    """Whether the numba-cuda package can compile and launch a kernel (device or
    simulator); the older ``numba.cuda`` bundled with numba is not a target."""
    if importlib.util.find_spec("numba_cuda") is None:
        return False
    try:
        from numba import cuda  # pylint: disable=import-outside-toplevel

        cell = cuda.to_device(np.zeros(1, dtype=np.int32))
        cuda.jit(_touch)[1, 1](cell)
        return bool(cell.copy_to_host()[0] == 1)
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def backend():
    """The active backend from ``SENTINEL_BACKEND`` (``cpu``|``cuda``|``auto``)."""
    want = os.environ.get("SENTINEL_BACKEND", "auto")
    if want == "auto":
        return "cuda" if cuda_available() else "cpu"
    if want not in BACKENDS:
        raise ValueError(f"SENTINEL_BACKEND={want!r}: expected cpu, cuda or auto")
    if want == "cuda" and not cuda_available():
        raise RuntimeError("SENTINEL_BACKEND=cuda but numba-cuda has no device")
    return want
