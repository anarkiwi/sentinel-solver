"""A recorded search game keeps the RAM after every move and loads back unchanged."""

import numpy as np
import pytest

from driver import trajectory

oracle = pytest.mark.oracle


@oracle
def test_a_recorded_game_keeps_each_moves_ram_and_round_trips(tmp_path):
    trajectory.main(
        ["0", "--preset", "narrow", "--max-actions", "2", "--out", str(tmp_path)]
    )
    rams, trace, won = trajectory.load(tmp_path / "traj_0000.npz")
    assert np.shape(rams) == (len(trace) + 1, 0x10000)
    assert np.result_type(rams) == np.uint8
    assert len(trace) == 2 and not won
    assert not np.array_equal(rams[0], rams[-1])
