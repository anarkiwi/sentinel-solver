"""Batched queries return, per state, what the single-state call returns."""

import numpy as np
import pytest

from sentinel import batch, kern, landtable
from sentinel.tests.test_kern import SIM, gpu

BOARDS = (42, 335, 0)
GRIDS = landtable.lattice(coarse=True)


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")


def test_first_of_maps_each_key_to_its_first_position():
    reps, back = batch.first_of(["a", "b", "a", "c", "b"])
    assert reps == [0, 1, 3]
    assert back == [0, 1, 0, 2, 1]
    assert batch.first_of([]) == ([], [])


class _Enough(BaseException):
    """Stops the play once enough replot advances are recorded."""


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
