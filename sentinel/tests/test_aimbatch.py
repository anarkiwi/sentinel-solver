"""The array aim pricing equals the scalar aim model, bit for bit."""

import functools

import numpy as np
import pytest

from sentinel import aimbatch, aimcost, landtable, objmodel, pancost
from sentinel import projector, settlecost, views
from sentinel.tests import aimref

pytestmark = pytest.mark.skipif(
    not objmodel.available(), reason="needs the game image for the object model"
)
GRIDS = [landtable.lattice(v_primary=p) for p in (True, False)]


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")


@pytest.fixture(autouse=True)
def cold_caches():
    for cache in (projector._COST_CACHE, pancost._NOTCH_CACHE):
        cache.clear()
    yield


@functools.cache
def _board(number, boulders):
    return aimref.board(number, boulders)


def _mems(*states):
    return np.stack([np.frombuffer(st.mem, dtype=np.uint8) for st in states])


def test_press_plan_is_h_press_count():
    h0 = np.repeat(np.arange(0, 256, 8), 32)
    h1 = np.tile(np.arange(0, 256, 8), 32)
    turn, steps = aimbatch.press_plan(h0, h1)
    want = [aimcost.h_press_count(a, b) for a, b in zip(h0.tolist(), h1.tolist())]
    assert list(zip(turn.tolist(), steps.tolist())) == want


@pytest.mark.parametrize("seed", range(3))
def test_notch_plots_are_pancost_notch_plots(seed):
    rng = np.random.default_rng(seed)
    h0, h1 = (rng.integers(0, 32, 200) * 8 for _ in range(2))
    v0, v1 = ((rng.integers(-12, 12, 200) * 4) & 0xFF for _ in range(2))
    valid, direction, plot_h, plot_v, _steps = aimbatch.notch_plots(h0, v0, h1, v1)
    for i in range(200):
        want = pancost.notch_plots(*(int(a[i]) for a in (h0, v0, h1, v1)))
        got = list(
            zip(direction[i][valid[i]], plot_h[i][valid[i]], plot_v[i][valid[i]])
        )
        assert [tuple(map(int, g)) for g in got] == want


@pytest.mark.parametrize("number,boulders", [(42, 0), (335, 0), (42, 3)])
def test_aim_parts_are_aim_table(number, boulders):
    """Every cell of both lattices, from aligned and skewed facings, on two boards at
    once: the parts continue to :func:`sentinel.views.aim_table`'s own sum state."""
    boards = [_board(number, boulders), _board(1442, 0)]
    mems = _mems(*boards)
    reqs, want = [], []
    for row, st in enumerate(boards):
        me = st.player
        for lat, grids in enumerate(GRIDS):
            skew = (0, 0) if lat == 0 else (3, 0x14)
            aim_from = (
                (int(st.obj_h_angle[me]) + skew[0]) & 0xFF,
                (int(st.obj_v_angle[me]) + skew[1]) & 0xFF,
            )
            cells = np.arange(len(grids[0]) * len(grids[1]))[::3]
            reqs.append((row, lat, *aim_from, cells))
            want.append(views.aim_table(st, cells, grids, aim_from, me))
    row, lat, h0, v0, cells = zip(*reqs)
    pan, turn, uturn = aimbatch.aim_parts(
        mems, np.array(row), np.array(lat), np.array(h0), np.array(v0), cells, GRIDS
    )
    for q, table in enumerate(want):
        got = views._sum_step(
            *views._sum_step(np.full(len(cells[q]), views.TOGGLE_FRAMES), 0.0, turn[q]),
            pan[q],
        )
        assert np.array_equal(got[0], table[0]) and np.array_equal(got[1], table[1])
    for q, st in enumerate(boards[i] for i in row):
        assert uturn[q] == settlecost.uturn_settle_frames(st)


def test_aim_parts_with_no_cells_still_prices_the_uturn():
    mems = _mems(_board(42, 0), _board(1442, 0))
    none = np.zeros(0, dtype=np.int64)
    aim = np.array([0, 0])
    empty = aimbatch.aim_parts(mems, aim[:0], aim[:0], aim[:0], aim[:0], [], GRIDS)
    assert [len(part) for part in empty] == [0, 0, 0]
    row = np.array([0, 1, 1])
    lat = np.array([0, 0, 1])
    h0 = v0 = np.zeros(3, dtype=np.int64)
    cells = [np.arange(4), none, none]
    pan, turn, uturn = aimbatch.aim_parts(mems, row, lat, h0, v0, cells, GRIDS)
    assert [len(p) for p in pan] == [4, 0, 0] and [len(t) for t in turn] == [4, 0, 0]
    ref = aimbatch.aim_parts(mems, row[:1], lat[:1], h0[:1], v0[:1], cells[:1], GRIDS)
    assert np.array_equal(pan[0], ref[0][0]) and uturn[0] == ref[2][0]
    every = aimbatch.aim_parts(mems, row, lat, h0, v0, [none] * 3, GRIDS)
    assert uturn.tolist() == every[2].tolist() and all(len(p) == 0 for p in every[0])
