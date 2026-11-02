"""The search engine: write footprint, action kernels against the reference actions, pool
operations against single-state calls, and CPU-versus-CUDA identity of every device op.
"""

import ast

import numpy as np
import pytest

from sentinel import (
    actions,
    engine,
    enemies,
    footprint,
    memmap as mm,
    objmodel,
    rows_jit,
)
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


def test_trackers_do_not_distinguish_rows_but_other_bytes_do(new_game):
    st = new_game(42).state
    pool = engine.Pool.from_states([st])
    kid = pool.fork([0, 0, 0])
    _addrs, amap = footprint.region()
    for tracker in footprint.LOS_TRACKERS:
        assert amap[tracker] >= 0
        pool.mut[kid[0], amap[tracker]] ^= 0xFF
    assert amap[mm.TARGETED_OBJECT_SLOT] >= 0
    pool.mut[kid[1], amap[mm.TARGETED_OBJECT_SLOT]] ^= 0xFF
    keys = pool.keys([0, *kid.tolist()]).tolist()
    assert keys[0] == keys[1] != keys[2]
    assert keys[3] == keys[0]


def test_unkept_tracker_bits_never_reach_a_row_field_or_the_clock(new_game):
    """Rows that differ only in the tracker bytes agree on every other byte, and on the
    clock, after the enemies have run: the property the key's mask relies on."""
    st = new_game(335).state
    st.mem[mm.PLAYER_NOT_ACTED] = 0
    pool = engine.Pool.from_states([st])
    kid = pool.fork([0])
    _addrs, amap = footprint.region()
    for tracker in footprint.LOS_TRACKERS:
        pool.mut[kid[0], amap[tracker]] = 0x5A
    pool.advance([0, kid[0]], 2000)
    keep = pool.keep.astype(bool)
    assert np.array_equal(pool.mut[0][keep], pool.mut[kid[0]][keep])
    assert np.array_equal(pool.clock[0], pool.clock[kid[0]])
    assert pool.keys([0])[0] == pool.keys([kid[0]])[0]


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
        [(rows_jit.FACE, 0x44, 0xF1), (rows_jit.CREATE, BOULDER, packed)],
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
        out[-1] += (pool.clocks(kids), pool.mutable(kids))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])
    for field in rows_jit.FIELDS:
        assert np.array_equal(out[0][2][field], out[1][2][field])
    assert _same(out[0][3], out[1][3])
    assert np.array_equal(out[0][4], out[1][4]) and out[0][4].any()
    assert np.array_equal(out[0][5], out[1][5])
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


@gpu
@pytest.mark.parametrize("who", ["device", "host", "race"])
def test_device_rows_are_the_same_whoever_runs_them(new_game, monkeypatch, who):
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    kernel = kern_cuda._advance_kernel

    class Halted:
        def __getitem__(self, cfg):
            launch = kernel[cfg]

            def run(*args):
                args[-2].copy_to_device(np.ones(args[-2].size, dtype=np.int32))
                launch(*args)

            return run

    def idle(b, *args):
        while not args[-2][b]:
            pass

    if who == "host":
        monkeypatch.setattr(kern_cuda, "_advance_kernel", Halted())
    if who == "device":
        monkeypatch.setattr(kern_cuda, "take_row", idle)
    st = new_game(42).state
    frames = [20, 7, 13, 3] if SIM else [1500, 90, 700, 3, 400]
    want = []
    for n in frames:
        ref = st.clone()
        enemies.advance_frames(ref, n)
        want.append(ref)
    pool = engine.Pool.from_states([st] * len(frames), device=True)
    pool.advance(np.arange(len(frames)), frames)
    assert all(_same(pool.state(i), w) for i, w in enumerate(want))


def test_default_device_wants_cuda_and_a_batch(monkeypatch):
    monkeypatch.setattr(engine.kern, "backend", lambda: "cuda")
    assert engine.default_device(2) and not engine.default_device(1)
    monkeypatch.setattr(engine.kern, "backend", lambda: "cpu")
    assert not engine.default_device(10**6)


def test_states_materialise_rows_with_their_clocks(new_game):
    base = new_game(42).state
    other = base.clone()
    enemies.advance_frames(other, 90)
    pool = engine.Pool.from_states([base, other])
    got = pool.states([1, 0])
    assert _same(got[0], other) and _same(got[1], base)
    assert pool.clocks([1])[0].tolist() == [getattr(other, f) for f in enemies.CLOCK]
    assert pool.mutable([0]).shape == (1, pool.addrs.size)


def test_advance_of_nothing_is_a_no_op(new_game):
    pool = engine.Pool.from_states([new_game(42).state])
    before = pool.mut.copy()
    pool.advance([], 100)
    assert np.array_equal(pool.mut, before)


def test_pool_overlay_read_and_states_agree(new_game):
    st = new_game(42).state
    pool = engine.Pool.from_states([st])
    mut = pool.mutable([0])
    img = pool.overlay(mut)
    assert bytes(img[0]) == bytes(st.mem)
    addrs = np.array([mm.PLAYER_OBJECT, mm.OBJECTS_X, 0x00E0, 0x9AF6])
    assert pool.read(mut, addrs)[0].tolist() == [st.mem[a] for a in addrs]
    got = pool.as_states(img, pool.clocks([0]))
    assert bytes(got[0].mem) == bytes(st.mem)
    assert [getattr(got[0], f) for f in enemies.CLOCK] == [
        getattr(st, f) for f in enemies.CLOCK
    ]


@pytest.mark.skipif(not objmodel.available(), reason="needs the object model")
@pytest.mark.parametrize("residual", [0, 10**7])
def test_batched_replot_pricing_is_the_scalar_one(new_game, residual):
    st = new_game(42).state
    st.cycle_residual = residual
    requests = [(3, 3, 20, 31), (3, 4, 20, 25), (3, 0, 12, 12), (3, 7, 9, 9)]
    fresh = [st.clone() for _ in requests]
    for state, req in zip(fresh, requests):
        enemies._price_replot(state, *req)
    mem = np.frombuffer(st.mem, dtype=np.uint8)
    seen = np.tile(mem[enemies.REPLOT_READS], (len(requests), 1))
    clock = np.tile([getattr(st, f) for f in enemies.CLOCK], (len(requests), 1))
    got, at, val = enemies.price_replots(seen, clock, np.array(requests))
    for i, state in enumerate(fresh):
        image = mem.copy()
        image[at[i]] = val[i]
        assert bytes(image) == bytes(state.mem)
        assert got[i].tolist() == [getattr(state, f) for f in enemies.CLOCK]


@gpu
def test_staged_buffers_persist_and_grow():
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    small = kern_cuda.stage("probe", np.arange(10))
    held = kern_cuda._BUFFERS["probe"]
    assert small.copy_to_host().tolist() == list(range(10))
    again = kern_cuda.stage("probe", np.arange(4).reshape(2, 2))
    assert kern_cuda._BUFFERS["probe"] is held
    assert again.copy_to_host().tolist() == [[0, 1], [2, 3]]
    big = kern_cuda.stage("probe", np.arange(100))
    assert kern_cuda._BUFFERS["probe"].size >= 100
    assert big.copy_to_host().tolist() == list(range(100))


@gpu
def test_peek_and_patch_kernels_touch_only_the_named_bytes():
    from numba import cuda  # pylint: disable=import-outside-toplevel
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(1)
    host = rng.integers(0, 256, (3, engine.IMAGE), dtype=np.uint8)
    imgs, clock = cuda.to_device(host.copy()), cuda.to_device(np.zeros((4, 12), "i8"))
    pos, addrs = np.array([2, 0]), np.array([5, 100, 9999])
    seen = kern_cuda.scratch("probe_seen", (2, 3), np.uint8)
    grid, block = kern_cuda.grid2(3, 2)
    kern_cuda._peek_kernel[grid, block](
        imgs,
        kern_cuda.stage("probe_pos", pos),
        kern_cuda.stage("probe_at", addrs),
        seen,
    )
    assert np.array_equal(seen.copy_to_host(), host[pos][:, addrs])
    at, val = np.array([[7, 8, 9], [70, 80, 90]]), np.array([[1, 2, 3], [4, 5, 6]])
    fresh = np.arange(24, dtype=np.int64).reshape(2, 12)
    grid, block = kern_cuda.grid2(12, 2)
    kern_cuda._patch_kernel[grid, block](
        imgs,
        clock,
        kern_cuda.stage("probe_pos", pos),
        kern_cuda.stage("probe_rows", np.array([3, 1])),
        kern_cuda.stage("probe_wat", at),
        kern_cuda.stage("probe_val", val),
        kern_cuda.stage("probe_clock", fresh),
    )
    want = host.copy()
    for p, a, v in zip(pos, at, val):
        want[p, a] = v
    assert np.array_equal(imgs.copy_to_host(), want)
    assert np.array_equal(clock.copy_to_host()[[3, 1]], fresh)
    assert not clock.copy_to_host()[[0, 2]].any()
