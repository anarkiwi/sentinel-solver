"""The spec layer: the subset every ``@spec`` kernel stays inside, backend selection, and
CPU-versus-CUDA bit identity (real GPU when present, numba's simulator in CI)."""

import ast
import importlib
import inspect
import os
import textwrap

import numpy as np
import pytest

from sentinel import kern, landtable, los, los_jit

FLOAT_OK = {"_cell_span"}
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
    "actions_jit",
    "badline_jit",
    "enemies_jit",
    "exposure_jit",
    "objectcost",
    "pricing_jit",
    "projector_jit",
    "rendercost",
    "rows_jit",
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
    assert {"march", "_march_one", "_cell_span", "_crosses", "_stop_one"} <= names
    assert kern.specs("sentinel.landtable").keys() == {
        "_cell_span",
        "_crosses",
        "_stop_one",
    }


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


def test_build_cpu_is_identity_and_rejects_strangers():
    assert kern.build(landtable._crosses, "cpu") is landtable._crosses
    with pytest.raises(ValueError):
        kern.build(landtable.stop_cells, "cpu")
    with pytest.raises(ValueError):
        kern.build(landtable._crosses, "tpu")


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


@gpu
@pytest.mark.parametrize("board", BOARDS)
def test_cuda_march_matches_cpu(board, new_game):
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    st = new_game(board).state
    slot = st.player
    grids = landtable.lattice()
    idx = _rays(st, slot, grids, np.random.default_rng(board))
    vecs = tuple(np.ascontiguousarray(a[idx]) for a in landtable.vectors(grids))
    params = _params(st, slot)
    mem = np.frombuffer(st.mem, dtype=np.uint8)
    status, tx, ty, _ = los_jit.march_batch(mem, *vecs, *params)
    clear = np.flatnonzero(status == los_jit.LOS_CLEAR)
    got_idx, got_tile = kern_cuda.landing(mem, vecs, params)
    assert clear.size
    assert np.array_equal(got_idx, clear)
    assert np.array_equal(got_tile, tx[clear] * 32 + ty[clear])


@gpu
@pytest.mark.parametrize("board", BOARDS)
def test_cuda_stop_matches_cpu(board, new_game):
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    st = new_game(board).state
    slot = st.player
    grids = landtable.lattice()
    idx = _rays(st, slot, grids, np.random.default_rng(board + 1))
    comps = tuple(
        np.ascontiguousarray(c[idx]) for c in landtable.cached_components(grids)
    )
    lo, hi, slope = landtable.surface_maps(st)
    args = (
        int(st.obj_x[slot]),
        int(st.obj_y[slot]),
        landtable.seed_z(st, slot, None),
        landtable.MAX_STEPS,
        landtable.RADIUS,
    )
    _first, ncand = landtable.stop_cells(*comps, lo, hi, slope, *args)
    mask = kern_cuda.stop_mask(comps, lo, hi, slope, *args)
    assert (ncand > 0).any() and not (ncand > 0).all()
    assert np.array_equal(mask, ncand > 0)


@gpu
@pytest.mark.skipif(SIM, reason="whole-lattice sweeps need a real device")
@pytest.mark.parametrize("board", (42, 335))
def test_backends_agree_on_public_queries(board, new_game, monkeypatch):
    st = new_game(board).state
    slot = st.player
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")
    lands = sorted(landtable.landable_set(st, slot))
    tiles = lands[:: max(1, len(lands) // 12)] + [(2, 2), (30, 30), (15, 1)]
    seen = {}
    for backend in ("cpu", "cuda"):
        monkeypatch.setenv("SENTINEL_BACKEND", backend)
        seen[backend] = (
            [landtable.landable_view(st, t, slot) for t in tiles],
            landtable.landable_set(st, slot),
            landtable.landing_rays(st, slot).tolist(),
        )
    assert seen["cpu"] == seen["cuda"]
    assert lands and any(v is not None for v in seen["cpu"][0])
