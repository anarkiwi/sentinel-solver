"""The batched pricing path equals the scalar cost model, on the CPU threads and the device."""

import os

import numpy as np
import pytest

from sentinel import objmodel, pancost, pricing_jit, projector
from sentinel.tests.test_kern import gpu

pytestmark = pytest.mark.skipif(
    not objmodel.available(), reason="needs the game image for the object model"
)
SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
BOARDS = (335,) if SIM else (0, 335, 1442)


@pytest.fixture(autouse=True)
def cold_caches():
    for cache in (projector._COST_CACHE, pancost._NOTCH_CACHE):
        cache.clear()
    yield


def _requests(st, count, seed):
    """Random notch, play, strip-window and odd-column requests on ``st``."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        view = {
            "h_angle": int(rng.integers(256)),
            "v_angle": int(rng.integers(256)),
            "ref_lo": 0x80 if i % 7 == 3 else 0,
        }
        d = int(rng.integers(4))
        kind = i % 4
        if kind == 0:
            out.append(
                (view, st.player, pancost.PAN_MODE[d], None, pancost.PAN_ROWS[d])
            )
        elif kind == 1:
            out.append((view, None, projector.PLAY_MODE, None, None))
        elif kind == 2:
            cols = int(rng.integers(1, 21))
            win = projector.strip_window(cols)
            out.append((view, st.player, projector.PLAY_MODE, win, None))
        else:
            out.append((view, st.player, 2, None, pancost.PAN_ROWS[0]))
    return out


@pytest.mark.parametrize("board", BOARDS)
def test_prime_costs_equal_scalar(board, new_game):
    st = new_game(board).state
    reqs = _requests(st, 40 if not SIM else 6, board)
    projector.prime_costs(st, reqs)
    primed = len(projector._COST_CACHE)
    got = [projector.render_cost(st, *r) for r in reqs]
    assert len(projector._COST_CACHE) == primed
    projector._COST_CACHE.clear()
    assert got == [projector.render_cost(st, *r) for r in reqs]
    assert primed > 1


def test_prefetch_matches_scalar_and_stops_on_scalar_paths(new_game, monkeypatch):
    st = new_game(335).state
    plots = pancost.notch_plots(0x10, 0x00, 0x90, 0x20)
    calls = [
        lambda d=d, h=h, v=v: pancost.notch_frames(st, d, h, v) for d, h, v in plots
    ]
    projector.prefetch(calls)
    assert projector._COST_CACHE
    got = [c() for c in calls]
    projector._COST_CACHE.clear()
    pancost._NOTCH_CACHE.clear()
    assert got == [c() for c in calls]
    monkeypatch.setattr(projector, "_exact_selected", lambda: True)
    projector._COST_CACHE.clear()
    projector.prefetch(calls)
    assert not projector._COST_CACHE


def test_deferred_passes_through_broad_handlers(new_game):
    st = new_game(335).state
    view = {"h_angle": 3, "v_angle": 5}

    def swallow():
        try:
            projector.render_cost(st, view)
        except Exception:  # pylint: disable=broad-except
            return "swallowed"
        return "done"

    projector.prefetch([swallow])
    assert projector._COST_CACHE
    assert swallow() == "done"


def test_serial_and_parallel_launch_agree(new_game):
    st = new_game(42).state
    reqs = _requests(st, 8, 1)
    jobs = np.stack(
        [
            projector._cost_job(
                st,
                r[0]["h_angle"],
                r[0]["v_angle"],
                st.player if r[1] is None else r[1],
                r[2],
                r[3],
                r[4],
                r[0]["ref_lo"],
            )
            for r in reqs
        ]
    )
    vis = np.stack([projector._occlusion_memo(st, st.player)[1]])
    mem = np.frombuffer(st.mem, dtype=np.uint8)
    model = projector._object_backend()[1]
    scr = pricing_jit.scratch(3)
    one, many = np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int64)
    pricing_jit.render_serial(mem, jobs, vis, model, scr, one)
    pricing_jit.render_many(mem, jobs, vis, model, scr, many)
    assert (one == many).all() and one.min() > 0


@gpu
@pytest.mark.parametrize("board", (335,) if SIM else (0, 335))
def test_cuda_pricing_matches_cpu(board, new_game):
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    st = new_game(board).state
    wants = [
        (
            r[0]["h_angle"],
            r[0]["v_angle"],
            st.player if r[1] is None else r[1],
            r[2],
            r[3],
            r[4],
            r[0]["ref_lo"],
        )
        for r in _requests(st, 3 if SIM else 64, board + 9)
    ]
    jobs = np.stack([projector._cost_job(st, *w) for w in wants])
    vis = np.stack([projector._occlusion_memo(st, st.player)[1]])
    mem = np.frombuffer(st.mem, dtype=np.uint8)
    model = projector._object_backend()[1]
    want = np.zeros(len(jobs), dtype=np.int64)
    pricing_jit.render_many(mem, jobs, vis, model, pricing_jit.scratch(2), want)
    assert (kern_cuda.render_cycles_many(mem, jobs, vis, model) == want).all()
    assert want.min() > 0
