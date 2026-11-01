"""Batched queries return, per state, what the single-state call returns."""

import numpy as np
import pytest

from sentinel import batch, enemies, kern, landtable, los_jit
from sentinel.tests.test_kern import SIM, gpu

BOARDS = (42, 335, 0)
GRIDS = landtable.lattice(coarse=True)


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")


def _same(a, b):
    return bytes(a.mem) == bytes(b.mem) and all(
        getattr(a, f) == getattr(b, f) for f in enemies.CLOCK
    )


def test_first_of_maps_each_key_to_its_first_position():
    reps, back = batch.first_of(["a", "b", "a", "c", "b"])
    assert reps == [0, 1, 3]
    assert back == [0, 1, 0, 2, 1]
    assert batch.first_of([]) == ([], [])


@pytest.mark.parametrize("plotting", (False, True))
def test_advance_matches_each_state_alone(new_game, plotting):
    boards = BOARDS + (42, 335)
    frames = [90, 700, 4000, 90, 0]
    got = [new_game(b).state for b in boards]
    want = [st.clone() for st in got]
    for st, n in zip(want, frames):
        enemies.advance_frames(st, n, plotting=plotting)
    batch.advance(got, frames, plotting=plotting)
    assert all(_same(a, b) for a, b in zip(got, want))


def test_advance_one_count_for_all(new_game):
    got = [new_game(b).state for b in BOARDS]
    want = [st.clone() for st in got]
    for st in want:
        enemies.advance_frames(st, 300)
    batch.advance(got, 300)
    assert all(_same(a, b) for a, b in zip(got, want))


def test_advance_runs_identical_states_once(new_game, monkeypatch):
    sts = [new_game(42).state for _ in range(3)] + [new_game(335).state]
    sts.append(new_game(42).state)
    seen = []
    run = enemies.advance_frames_many

    def spy(states, frames, plotting=False):
        seen.append(len(states))
        return run(states, frames, plotting)

    monkeypatch.setattr(enemies, "advance_frames_many", spy)
    frames = [200, 200, 201, 200, 200]
    want = [st.clone() for st in sts]
    for st, n in zip(want, frames):
        enemies.advance_frames(st, n)
    batch.advance(sts, frames)
    assert seen == [3]
    assert all(_same(a, b) for a, b in zip(sts, want))


def _replot_advances(monkeypatch, count):
    """``[(state, frames)]``: the first ``count`` advances of a real play of ls42 that
    stopped at a strip replot, each state as it was before the call."""
    from sentinel.game import Game  # pylint: disable=import-outside-toplevel
    from sentinel.phase_player import (
        PhasePlayer,
    )  # pylint: disable=import-outside-toplevel

    priced, found = [0], []
    price = enemies._price_replot
    run = enemies.advance_frames

    def counted(*args):
        priced[0] += 1
        return price(*args)

    def record(state, frames, plotting=False):
        before, priced[0] = state.clone(), 0
        run(state, frames, plotting)
        if priced[0] and len(found) < count:
            found.append((before, frames))

    monkeypatch.setattr(enemies, "_price_replot", counted)
    monkeypatch.setattr(enemies, "advance_frames", record)
    PhasePlayer(Game.typed(42), verbose=False).run(max_actions=4)
    assert len(found) == count
    return found


def test_advance_prices_strip_replots_between_rounds(monkeypatch):
    found = _replot_advances(monkeypatch, 4)
    got = [st.clone() for st, _ in found]
    want = [st.clone() for st, _ in found]
    frames = [n for _, n in found]
    for st, n in zip(want, frames):
        enemies.advance_frames(st, n)
    priced = []
    price = enemies._price_replot
    monkeypatch.setattr(
        enemies, "_price_replot", lambda *a: priced.append(1) or price(*a)
    )
    batch.advance(got, frames)
    assert len(priced) >= len(found)
    assert all(_same(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("threads", (1, 99))
def test_landings_match_single_and_share_duplicates(new_game, monkeypatch, threads):
    monkeypatch.setattr(batch, "width", lambda: threads)
    sts = [new_game(b).state for b in (42, 335, 42)]
    got = batch.landings(sts, GRIDS)
    for st, (rays, tiles) in zip(sts, got):
        one = landtable.landing_pairs(st, st.player, None, GRIDS, landtable.MAX_STEPS)
        assert rays.size
        assert np.array_equal(rays, one[0]) and np.array_equal(tiles, one[1])
    assert got[0] is got[2]
    assert not np.array_equal(got[0][0], got[1][0])


def test_cuda_pays_by_occupancy(monkeypatch):
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    assert not batch.cuda_pays(10**9, 10**3)
    monkeypatch.setattr(kern, "backend", lambda: "cuda")
    monkeypatch.setattr(kern_cuda, "resident_threads", lambda: 1000)
    assert not batch.cuda_pays(100, 9)
    assert batch.cuda_pays(100, 10)
    assert batch.cuda_pays(1000, 1)


@gpu
def test_cuda_batch_launch_matches_cpu_marches(new_game):
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    n = 48 if SIM else 20000
    sts = [new_game(b).state for b in (335, 42, 335)]
    idx = np.unique(np.random.default_rng(1).integers(0, 884736, n))
    vecs = tuple(np.ascontiguousarray(a[idx]) for a in landtable.vectors(GRIDS))
    params = [landtable._march_params(s, s.player, None, 6000) for s in sts]
    mems = [np.frombuffer(s.mem, dtype=np.uint8) for s in sts]
    got = kern_cuda.landing_many(mems, vecs, params)
    for mem, par, (gi, gt) in zip(mems, params, got):
        status, tx, ty, _ = los_jit.march_batch(mem, *vecs, *par)
        clear = np.flatnonzero(status == los_jit.LOS_CLEAR)
        assert np.array_equal(gi, clear)
        assert np.array_equal(gt, tx[clear] * 32 + ty[clear])
    assert np.array_equal(got[0][0], kern_cuda.landing(mems[0], vecs, params[0])[0])


@gpu
@pytest.mark.skipif(SIM, reason="whole-lattice sweeps need a real device")
def test_cuda_batch_landings_match_cpu(new_game, monkeypatch):
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    sts = [new_game(b).state for b in (42, 335, 42)]
    want = batch.landings(sts, GRIDS)
    monkeypatch.setenv("SENTINEL_BACKEND", "cuda")
    monkeypatch.setattr(kern_cuda, "resident_threads", lambda: 0)
    kern_cuda._MEMO.clear()
    got = batch.landings(sts, GRIDS)
    for (gr, gt), (wr, wt) in zip(got, want):
        assert np.array_equal(gr, wr) and np.array_equal(gt, wt)
    assert got[0][0] is got[2][0]


def test_first_by_tile_takes_the_lowest_ray_per_tile():
    dense = np.full((2, 8), -1, dtype=np.int16)
    dense[0, [1, 3, 4]] = [7, 5, 7]
    dense[1, 6] = 1023
    got = batch.first_by_tile(dense)
    assert got.shape == (2, 1024)
    assert (got[0, 7], got[0, 5], got[1, 1023]) == (1, 3, 6)
    assert (got >= 0).sum() == 3


def test_pool_advance_matches_single_states_and_dedupes(new_game):
    sts = [new_game(b).state for b in (42, 335, 42, 0)]
    pool = batch.Pool.from_states(sts)
    frames = [300, 900, 300, 50]
    want = [st.clone() for st in sts]
    for st, n in zip(want, frames):
        enemies.advance_frames(st, n)
    pool.advance(np.arange(4), frames)
    assert all(_same(pool.state(i), w) for i, w in enumerate(want))
    assert np.array_equal(pool.mem[0], pool.mem[2])


def test_pool_advance_prices_replots(monkeypatch):
    found = _replot_advances(monkeypatch, 3)
    pool = batch.Pool.from_states([st for st, _ in found])
    want = [st.clone() for st, _ in found]
    for st, (_, n) in zip(want, found):
        enemies.advance_frames(st, n)
    pool.advance(np.arange(3), [n for _, n in found])
    assert all(_same(pool.state(i), w) for i, w in enumerate(want))


def test_pool_fork_is_independent_and_state_round_trips(new_game):
    st = new_game(42).state
    pool = batch.Pool.from_states([st])
    kid = pool.fork([0, 0])
    assert kid.tolist() == [1, 2] and len(pool) == 3
    pool.advance(kid[:1], 500)
    assert np.array_equal(pool.mem[0], pool.mem[2])
    assert not np.array_equal(pool.mem[0], pool.mem[1])
    assert _same(pool.state(0), st)


def test_pool_landing_is_first_ray_per_tile_of_the_pairs(new_game):
    sts = [new_game(b).state for b in (42, 335, 42)]
    pool = batch.Pool.from_states(sts)
    got = pool.landing(np.arange(3), GRIDS)
    assert got.shape == (3, 1024)
    assert np.array_equal(got[0], got[2])
    for st, row in zip(sts[:2], got):
        rays, tiles = landtable.landing_pairs(
            st, st.player, None, GRIDS, landtable.MAX_STEPS
        )
        assert rays.size
        want = np.full(1024, -1)
        for ray, tile in zip(rays[::-1].tolist(), tiles[::-1].tolist()):
            want[tile] = ray
        assert np.array_equal(row, want)
