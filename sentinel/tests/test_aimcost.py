"""Unit tests for the keyboard-aim pan geometry (no ROM required)."""

from sentinel import aimcost as ac


def test_bearing_cardinals():
    # +x is bearing 0; the 256-unit compass turns anticlockwise-to-atan2.
    assert ac.bearing_to(0, 0, 1, 0) == 0
    assert ac.bearing_to(0, 0, 0, 1) == 64  # +y -> quarter turn
    assert ac.bearing_to(0, 0, -1, 0) == 128
    assert ac.bearing_to(5, 5, 5, 5) is None  # same tile: no bearing


def test_h_steps_shortest_wrap():
    assert ac.h_steps(0, 64) == 8  # 64 units / 8
    assert ac.h_steps(0, 8) == 1
    # shortest way around the circle: 0 -> 248 is -8 units, one step, not 31.
    assert ac.h_steps(0, 248) == 1
    assert ac.h_steps(0, 128) == 16  # antipode: 128/8 either way


def test_v_steps_no_wrap():
    assert ac.v_steps(0, 16) == 4
    assert ac.v_steps(0xE1, 0xED) == 3  # (0xED-0xE1)/4 = 12/4


def test_pan_steps_sums_axes_and_tolerates_none():
    assert ac.pan_steps(0, 0, 64, 16) == 8 + 4
    assert ac.pan_steps(None, 0, 64, 16) == 4  # no h -> only pitch counts
    assert ac.pan_steps(0, None, 64, 16) == 8  # no v -> only bearing counts


def test_h_press_count_uses_uturn_past_half_turn():
    # short swings step directly, no u-turn.
    assert ac.h_press_count(0, 0) == (0, 0)
    assert ac.h_press_count(0, 8) == (0, 1)
    assert ac.h_press_count(0, 64) == (0, 8)  # d=8: direct 8 < u-turn 1+8
    # past the crossover a u-turn + short correction is fewer keystrokes.
    assert ac.h_press_count(0, 72) == (1, 7)  # d=9: u-turn 1+7=8 < 9
    assert ac.h_press_count(0, 120) == (1, 1)  # d=15: 1+1=2 << 15
    assert ac.h_press_count(0, 128) == (1, 0)  # antipode: one keystroke, no correction


def test_h_press_count_never_worse_than_direct():
    for h1 in range(0, 256, 8):
        nu, ns = ac.h_press_count(0, h1)
        assert nu * ac.UTURN_STEP == 0 or nu == 1
        assert nu + ns <= ac.h_steps(0, h1)  # never more keystrokes than stepping


def test_bearing_rounds_prefers_uturn_for_antipode():
    # a 180-degree swing: one u-turn (weight w) beats 16 pan steps (weight s) whenever w < 16s.
    assert ac.bearing_rounds(0, 128, 16, 16) == 16  # one u-turn
    assert ac.bearing_rounds(0, 64, 16, 16) == 8 * 16  # direct, no u-turn


def test_return_pan_dominates():
    """A return-aim swinging the bearing ~180 degrees back (the reabsorb pan) costs
    far more keystrokes than a small re-center -- the asymmetry the flat per-move
    tick constant used to miss."""
    recenter = ac.pan_steps(0x88, 0xED, 0x90, 0xF1)  # build -> synthoid centre
    reabsorb = ac.pan_steps(0x90, 0xF1, 0x08, 0xE5)  # -> look back at prior tile
    assert reabsorb > 5 * recenter
