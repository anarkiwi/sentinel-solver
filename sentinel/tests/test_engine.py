"""The search engine: write footprint, action kernels against the reference actions, pool
operations against single-state calls, and CPU-versus-CUDA identity of every device op.
"""

import ast

import numpy as np
import pytest

from sentinel import actions, engine, enemies, footprint, memmap as mm, rows_jit
from sentinel.tests.test_batch import _replot_advances, _same
from sentinel.tests.test_kern import SIM, gpu

BOARDS = (42, 335, 0, 1442)
ROBOT, BOULDER, TREE = mm.T_ROBOT, mm.T_BOULDER, mm.T_TREE


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")


def _span(expr, **scope):
    return footprint._span(ast.parse(expr, mode="eval").body, scope)


def _free_tile(st):
    """A tile of bare terrain next to the player, to build on."""
    px, py = st.player_xy()
    for dx in (1, -1, 2, -2, 3):
        tile = (px + dx, py)
        if tile[0] in range(31) and mm.OBJECT_TILE > st.mem[0x400 + tile[0] * 32 + py]:
            return tile
    raise AssertionError("no free tile")


@pytest.mark.parametrize(
    "expr,want",
    [
        ("7", (7, 7)),
        ("BASE + 3", (13, 13)),
        ("BASE + slot", (10, 265)),
        ("(slot & 0x3F) + 4", (4, 67)),
        ("np.int64(x) << 2", (0, 1020)),
        ("BASE + (slot >> 4)", (10, 25)),
        ("lut[slot]", (0, 255)),
        ("BASE - 2", (8, 8)),
    ],
)
def test_span_intervals(expr, want):
    assert _span(expr, BASE=10, np=np) == want


def test_loop_variables_bound_only_where_they_are_not_parameters():
    src = (
        "def f(mem, slot):\n    for k in range(3, 9):\n        mem[100 + k] = 0\n"
        "        mem[100 + slot] = 1\n"
    )
    fn = ast.parse(src).body[0]
    got = footprint._stores(fn, {})
    assert got == [(103, 108), (100, 355)]


def test_footprint_holds_every_write_and_stays_small(new_game):
    mask = footprint.mask()
    assert 1000 < mask.sum() < 4096
    for board in BOARDS:
        st = new_game(board).state
        before = bytes(st.mem)
        for tile in (_free_tile(st),):
            slot = actions.create(st, BOULDER, tile)
            top = actions.create(st, ROBOT, tile)
            assert slot is not None and top is not None
            actions.transfer(st, top)
            actions.absorb(st, slot)
        enemies.advance_frames(st, 4000)
        changed = np.frombuffer(before, np.uint8) != np.frombuffer(st.mem, np.uint8)
        assert changed.sum() > 8
        assert not (changed & ~mask).any()


def test_region_maps_addresses_to_row_columns():
    addrs, amap = footprint.region()
    assert (amap[addrs] == np.arange(addrs.size)).all()
    assert (amap >= 0).sum() == addrs.size


@pytest.mark.parametrize("board", BOARDS)
def test_action_kernels_match_the_reference(board, new_game):
    ref = new_game(board).state
    tile = _free_tile(ref)
    packed = tile[0] * 32 + tile[1]
    seqs = [
        [(rows_jit.CREATE, BOULDER, packed)] * 2,
        [(rows_jit.CREATE, BOULDER, packed), (rows_jit.CREATE, ROBOT, packed)]
        + [(rows_jit.TRANSFER, rows_jit.LAST, 0)],
        [(rows_jit.CREATE, TREE, packed), (rows_jit.CREATE, BOULDER, packed)],
        [(rows_jit.ABSORB, 0, 0)],
        [(rows_jit.TRANSFER, 1, 0)],
        [(rows_jit.CREATE, ROBOT, packed)] * 40,
        [(rows_jit.TRANSFER, rows_jit.LAST, 0)],
    ]
    pool = engine.Pool.from_states([ref])
    kids, ok = pool.expand(np.zeros(len(seqs), dtype=int), engine.pad_ops(seqs))
    for seq, kid, took in zip(seqs, kids, ok):
        want = ref.clone()
        done = True
        last = None
        for op, a, b in seq:
            if op == rows_jit.CREATE:
                last = actions.create(want, a, (b >> 5, b & 31))
                done = last is not None
            elif op == rows_jit.ABSORB:
                done = actions.absorb(want, a)
            else:
                done = (
                    actions.transfer(want, last if a < 0 else a)
                    if (last is not None or a >= 0)
                    else False
                )
            if not done:
                break
        assert bool(took) == bool(done), seq
        assert bytes(pool.state(kid).mem) == bytes(want.mem), seq
    assert ok.sum() >= 2 and not ok.all()


def test_evaluate_reads_the_decision_fields(new_game):
    st = new_game(335).state
    pool = engine.Pool.from_states([st])
    got = pool.evaluate([0])
    assert got["energy"][0] == st.energy
    assert got["player"][0] == st.player
    assert got["eye"][0] == int(round(st.eye_z() * 256))
    assert got["alive"][0] == 1 and got["won"][0] == 0
    assert got["enemies"][0] == len(enemies.enemy_slots(st))
    assert (got["x"][0], got["y"][0]) == tuple(st.player_xy())
    st.mem[mm.PLAYER_ENERGY] = 0
    st.mem[mm.PLAYER_HAS_HYPERSPACED] = 0xC0
    pool.add([st])
    got = pool.evaluate([1])
    assert got["alive"][0] == 0 and got["won"][0] == 1


def test_pool_advance_matches_single_states_and_dedupes(new_game):
    base = new_game(42).state
    other = base.clone()
    enemies.advance_frames(other, 120)
    pool = engine.Pool.from_states([base, base.clone(), other])
    frames = [900, 900, 300]
    want = [st.clone() for st in (base, base, other)]
    for st, n in zip(want, frames):
        enemies.advance_frames(st, n)
    pool.advance(np.arange(3), frames)
    assert all(_same(pool.state(i), w) for i, w in enumerate(want))
    assert np.array_equal(pool.mut[0], pool.mut[1])


def test_pool_advance_prices_replots(monkeypatch):
    found = _replot_advances(monkeypatch, 3)
    monkeypatch.undo()
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")
    pool = engine.Pool.from_states([st for st, _ in found])
    want = [st.clone() for st, _ in found]
    for st, (_, n) in zip(want, found):
        enemies.advance_frames(st, n)
    pool.advance(np.arange(3), [n for _, n in found])
    assert all(_same(pool.state(i), w) for i, w in enumerate(want))


def test_fork_is_independent_and_keys_identify_rows(new_game):
    st = new_game(42).state
    pool = engine.Pool.from_states([st])
    kid = pool.fork([0, 0])
    assert kid.tolist() == [1, 2] and len(pool) == 3
    assert len(set(pool.keys([0, 1, 2]).tolist())) == 1
    pool.advance(kid[:1], 500)
    keys = pool.keys([0, 1, 2]).tolist()
    assert keys[0] == keys[2] != keys[1]
    assert _same(pool.state(0), st)
    pool.clock[2, 0] += 1
    assert pool.keys([0, 2])[0] != pool.keys([0, 2])[1]


def test_pool_grows_and_rejects_a_foreign_image(new_game):
    st = new_game(42).state
    pool = engine.Pool.from_states([st])
    for _ in range(5):
        pool.fork(np.zeros(20, dtype=int))
    assert len(pool) == 101 and np.array_equal(pool.mut[100], pool.mut[0])
    foreign = st.clone()
    foreign.mem[0x9000] ^= 0xFF
    with pytest.raises(ValueError):
        pool.add([foreign])


def test_view_lays_the_mutable_bytes_over_the_static_image(new_game):
    st = new_game(335).state
    pool = engine.Pool.from_states([st])
    assert bytes(pool.view([0])[0]) == bytes(st.mem)


def test_selection_primitives():
    assert engine.first(np.array([0, 0, 1, 1])) == 2
    assert engine.first(np.zeros(3)) == -1
    assert engine.argbest(np.array([1, 3, 3, 2]), np.array([0, 1, 2, 9])) == 2
    assert engine.argbest(np.array([5, 5])) == 0
    ops = engine.pad_ops([[(0, 1, 2)], [], [(0, 1, 2), (1, 3, 4)]])
    assert ops.shape == (3, 2, 3) and (ops[1] == engine.NOOP_ROW).all()


def test_landing_table_is_first_ray_per_tile(new_game):
    from sentinel import landtable  # pylint: disable=import-outside-toplevel

    grids = landtable.lattice(coarse=True)
    sts = [new_game(42).state]
    sts.append(sts[0].clone())
    pool = engine.Pool.from_states(sts)
    got = pool.landing(np.arange(2), grids)
    assert got.shape == (2, 1024) and (got[0] >= 0).any()
    assert np.array_equal(got[0], got[1])


@gpu
def test_device_pool_matches_host_pool(new_game):
    st = new_game(42).state
    tile = _free_tile(st)
    packed = tile[0] * 32 + tile[1]
    seqs = [
        [(rows_jit.CREATE, BOULDER, packed)],
        [(rows_jit.CREATE, ROBOT, packed), (rows_jit.TRANSFER, rows_jit.LAST, 0)],
        [(rows_jit.ABSORB, 0, 0)],
    ]
    frames = [20] * 3 if SIM else [1500, 400, 900]
    pools = [
        engine.Pool.from_states([st, st.clone()], device=dev) for dev in (False, True)
    ]
    out = []
    for pool in pools:
        kids, ok = pool.expand(np.zeros(3, dtype=int), engine.pad_ops(seqs))
        pool.advance(kids, frames)
        out.append((pool.keys(kids), ok, pool.evaluate(kids), pool.state(kids[1])))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])
    for field in rows_jit.FIELDS:
        assert np.array_equal(out[0][2][field], out[1][2][field])
    assert _same(out[0][3], out[1][3])
    assert len(set(out[1][0].tolist())) == 3
    dev = pools[1]
    dev.fork_into([0], [1])
    assert dev.keys([0, 1])[0] == dev.keys([0, 1])[1]


@gpu
@pytest.mark.skipif(SIM, reason="replot stops need whole-board runs")
def test_device_advance_prices_replots(monkeypatch):
    found = _replot_advances(monkeypatch, 3)
    monkeypatch.undo()
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")
    pool = engine.Pool.from_states([st for st, _ in found], device=True)
    want = [st.clone() for st, _ in found]
    for st, (_, n) in zip(want, found):
        enemies.advance_frames(st, n)
    pool.advance(np.arange(3), [n for _, n in found])
    assert all(_same(pool.state(i), w) for i, w in enumerate(want))
