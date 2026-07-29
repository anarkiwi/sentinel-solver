"""The recorder captures the enemy CLOCK the distilled fixtures used to omit.

``watch_play._enemy_clock`` decodes exactly the per-enemy facing + rotation step +
cooldowns that ``driver.replay_human._enemy_truth`` reads live, and the ``$1335``
accumulator now in the read span is the byte ``[0, $0CFF]`` alone could not hold.
"""

import pytest

from driver import sentinel_state as gs, watch_play as wp
from driver.replay_human import _enemy_truth
from sentinel import landscape, memmap as mm


@pytest.mark.parametrize("seed,n_enemies", [(0, 1), (66, 2), (821, 7)])
def test_enemy_clock_matches_live_truth_schema(seed, n_enemies):
    """Recorder decode == live-replay decode, field for field, on the ls335 board
    (seed 821, 7 enemies) and the two committed-fixture boards."""
    img = bytearray(landscape.generate(seed).mem)
    st = gs.read_game_state(gs.Py65Source(img))
    end = mm.ROTATION_SPEED_TABLE + wp._ROT_TABLE_LEN
    clock = wp._enemy_clock(st, img, bytes(img[mm.ROTATION_SPEED_TABLE : end]))
    assert len(clock) == n_enemies
    assert clock == _enemy_truth(bytes(img))


def test_read_span_covers_the_cooldown_accumulator():
    """The one mutable enemy-clock byte above the play state ($1335) is inside the
    per-poll read span; the base64 "mem" span still stops at the play page."""
    assert wp._READ_END == mm.COOLDOWN_BRESENHAM == 0x1335
    assert wp._STATE_END == 0x0CFF < wp._READ_END
