"""The array aim pricing equals the scalar aim model, bit for bit."""

import functools

import numpy as np
import pytest

from sentinel import aimbatch, aimcost, landtable, objmodel, pancost, playerbase
from sentinel import projector, settlecost
from sentinel.game import Game
from sentinel.phase_player import PhasePlayer

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
def _board(number, actions):
    player = PhasePlayer(Game.typed(number), verbose=False)
    if actions:
        player.run(max_actions=actions)
    return player.st


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


@pytest.mark.parametrize("number,actions", [(42, 0), (335, 0), (42, 6)])
def test_aim_parts_are_aim_table(number, actions):
    """Every cell of both lattices, from aligned and skewed facings, on two boards at
    once: the parts continue to :func:`playerbase.aim_table`'s own sum state."""
    boards = [_board(number, actions), _board(1442, 0)]
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
            want.append(playerbase.aim_table(st, cells, grids, aim_from, me))
    row, lat, h0, v0, cells = zip(*reqs)
    pan, turn, uturn = aimbatch.aim_parts(
        mems, np.array(row), np.array(lat), np.array(h0), np.array(v0), cells, GRIDS
    )
    for q, table in enumerate(want):
        got = playerbase._sum_step(
            *playerbase._sum_step(
                np.full(len(cells[q]), playerbase.TOGGLE_FRAMES), 0.0, turn[q]
            ),
            pan[q],
        )
        assert np.array_equal(got[0], table[0]) and np.array_equal(got[1], table[1])
    for q, st in enumerate(boards[i] for i in row):
        assert uturn[q] == settlecost.uturn_settle_frames(st)


@pytest.mark.parametrize("reuse", [False, True])
def test_aim_total_columns_are_aim_frames(reuse):
    st = _board(42, 6)
    me = st.player
    aim_from = (int(st.obj_h_angle[me]), int(st.obj_v_angle[me]))
    grids = GRIDS[0]
    per_cell = len(grids[2]) * len(grids[3])
    rng = np.random.default_rng(5)
    cells = np.arange(len(grids[0]) * len(grids[1]))
    rays = cells * per_cell + rng.integers(0, per_cell, cells.size)
    views = [playerbase._view_at(int(r), grids) for r in rays]
    cursor = (70, 101) if reuse else playerbase.SIGHTS_CENTRE
    pan, _turn, uturn = aimbatch.aim_parts(
        _mems(st),
        np.array([0]),
        np.array([0]),
        np.array(aim_from[:1]),
        np.array(aim_from[1:]),
        [cells],
        GRIDS,
    )
    turns, _steps = aimbatch.press_plan(
        np.full(cells.size, aim_from[0]), np.array([v["h_angle"] for v in views])
    )
    drive = np.array(
        [
            max(abs(v["cursor"][0] - cursor[0]), abs(v["cursor"][1] - cursor[1]))
            for v in views
        ]
    )
    got = aimbatch.aim_total_columns(
        np.full(cells.size, 0.0 if reuse else playerbase.TOGGLE_FRAMES),
        np.where(turns > 0, uturn[0], 0.0),
        pan[0],
        drive,
    )
    want = [
        playerbase.aim_frames(st, v, aim_from, me, cursor if reuse else None)
        for v in views
    ]
    assert got.tolist() == want


@pytest.mark.parametrize("number,actions", [(42, 0), (335, 0)])
def test_plan_spans_are_the_phase_players(number, actions):
    """Every priced move: the view the phase player's lookup gives, and the span its own
    aim-plus-settle chain gives, as the frames the beam books."""
    from sentinel import beam_player as bp  # pylint: disable=import-outside-toplevel

    game = Game.typed(number)
    if actions:
        PhasePlayer(game, verbose=False).run(max_actions=actions)
    st = game.state
    beam = {
        "cursor": np.array([playerbase.SIGHTS_CENTRE]),
        "bearing": np.full((1, 2), bp.NO_BEARING),
    }
    cols = bp.plan([st], _mems(st), bp.Board.of_states([st]), beam, 2)
    player = PhasePlayer(game, verbose=False)
    views = playerbase._Views(st)
    assert (cols["kind"] == bp.WAIT).sum() == 1 and len(cols["kind"]) > 1
    for i, kind in enumerate(cols["kind"].tolist()):
        k, tile = int(cols["k"][i]), int(cols["tile"][i])
        if kind == bp.WAIT:
            assert cols["span"][i] == bp.WAIT_QUANTUM
            continue
        xy = (tile // 32, tile % 32)
        view = views.get(xy, band=True)
        assert cols["view"][i].tolist() == [
            *(view[f] for f in ("h_angle", "v_angle")),
            *view["cursor"],
        ]
        span = {v: player._span(v, view, xy) for v in bp._verbs(kind, k)}
        assert cols["span"][i] == int(round(bp._span_of(kind, k, span)))
