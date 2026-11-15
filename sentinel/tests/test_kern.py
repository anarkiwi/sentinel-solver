"""The spec layer: the subset every ``@spec`` kernel stays inside, backend selection, and
CPU-versus-CUDA bit identity (real GPU when present, numba's simulator in CI)."""

import ast
import importlib
import inspect
import os
import textwrap

import numpy as np
import pytest

from sentinel import kern, landtable, los

FLOAT_OK = {"_cell_span", "_cell_last"}
CASTS = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
BUILTINS = {"min", "max", "int", "range", "abs"}
BANNED = (
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Assert,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.Pow,
)
for _mod in (
    "sight_jit",
    "exposure_jit",
    "objectcost",
    "pricing_jit",
    "projector_jit",
    "rendercost",
):
    importlib.import_module(f"sentinel.{_mod}")
SPECS = sorted(kern.specs().items())


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    """Tests pick a backend explicitly; the reference side is always the CPU path."""
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")


def violations(source, name, peers=()):
    """The ways a spec's ``source`` leaves the subset."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    floats = name in FLOAT_OK
    out = []
    for node in ast.walk(tree):
        if isinstance(node, BANNED):
            out.append(type(node).__name__)
        elif isinstance(node, ast.Div) and not floats:
            out.append("float division")
        elif isinstance(node, ast.Constant) and isinstance(node.value, float):
            if not floats:
                out.append("float constant")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id == name:
                    out.append("recursion")
                elif func.id not in BUILTINS and func.id not in peers:
                    out.append(f"call {func.id}")
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                cast = func.value.id == "np" and func.attr in CASTS
                if not cast and not (floats and func.value.id == "math"):
                    out.append(f"call {func.value.id}.{func.attr}")
            else:
                out.append("indirect call")
    return out


def test_specs_registered():
    names = set(kern.specs())
    assert {"march", "_march_one", "_stop_one", "_can_see_object"} <= names
    assert kern.specs("sentinel.landtable").keys() == {"_cell_last", "_stop_one"}


@pytest.mark.parametrize("name,disp", SPECS, ids=[n for n, _ in SPECS])
def test_spec_subset(name, disp):
    assert not disp.targetoptions.get("fastmath")
    assert not violations(inspect.getsource(disp.py_func), name, kern.specs())


@pytest.mark.parametrize(
    "source,expect",
    [
        ("def f(a):\n    return np.zeros(a)\n", "call np.zeros"),
        ("def f(a):\n    raise ValueError(a)\n", "Raise"),
        ("def f(a):\n    return a / 2\n", "float division"),
        ("def f(a):\n    return f(a - 1)\n", "recursion"),
        ("def f(a):\n    return [a]\n", "List"),
        ("def f(a):\n    return a * 0.5\n", "float constant"),
        ("def f(a):\n    return g(a)\n", "call g"),
    ],
)
def test_lint_rejects(source, expect):
    assert expect in violations(source, "f")


def test_backend_env(monkeypatch):
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")
    assert kern.backend() == "cpu"
    monkeypatch.setenv("SENTINEL_BACKEND", "bogus")
    with pytest.raises(ValueError):
        kern.backend()
    monkeypatch.setenv("SENTINEL_BACKEND", "auto")
    assert kern.backend() == ("cuda" if kern.cuda_available() else "cpu")
    if not kern.cuda_available():
        monkeypatch.setenv("SENTINEL_BACKEND", "cuda")
        with pytest.raises(RuntimeError):
            kern.backend()


_NEEDS = pytest.mark.skipif(not kern.cuda_available(), reason="needs numba-cuda")


def gpu(fn):
    """Mark a test as needing numba-cuda: skipped without a device or the simulator."""
    return pytest.mark.gpu(_NEEDS(fn))


SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
SLICE = 32 if SIM else 20000
BOARDS = (335, 42) if SIM else (0, 42, 335, 1442)


def _params(st, slot):
    return landtable._march_params(st, slot, None, landtable.MAX_STEPS)


def _rays(st, slot, grids, rng):
    """Random lattice indices, half of them from the landing candidates."""
    n = los._lattice_vectors(*grids)[0].shape[0]
    cand = landtable.landing_rays(st, slot, None, grids)
    return np.unique(
        np.concatenate([rng.integers(0, n, SLICE // 2), rng.choice(cand, SLICE // 2)])
    )
