"""Every DERIVED timing constant recomputed from the ROM primitive its comment
cites.  Each case names the primitive symbols, not the derived literal."""

import pytest

from driver import kbd_aim
from sentinel import actioncost, aimcost, enemies, enemies_jit, memmap as mm
from sentinel import pancost, playerbase, projector

UNIT = 3 * 256.0 / mm.COOLDOWN_BRESENHAM_STEP  # 1-in-3 gate x 205/256 Bresenham


@pytest.mark.parametrize(
    "name,derived,expected",
    [
        # $1310 ADC #$CD Bresenham divider under the $0C50 1-in-3 scan gate.
        ("UNIT_FRAMES", playerbase.UNIT_FRAMES, UNIT),
        # $1813 rotation cooldown reload, in cooldown units.
        (
            "ROT_PERIOD_FRAMES",
            playerbase.ROT_PERIOD_FRAMES,
            enemies.ROTATION_COOLDOWN_RELOAD * UNIT,
        ),
        # $1835 first-target draining cooldown reload.
        (
            "DRAIN_DELAY",
            playerbase.DRAIN_DELAY,
            enemies.DRAINING_COOLDOWN_RELOAD * UNIT,
        ),
        # $1869 post-meanie-create hold.
        (
            "MEANIE_SPAWN_FRAMES",
            playerbase.MEANIE_SPAWN_FRAMES,
            enemies.UPDATE_COOLDOWN_MEANIE_MADE * UNIT,
        ),
        # $171B half-turn at MEANIE_ROTATE_STEP units, $173A rounds per step.
        (
            "MEANIE_ARM_FRAMES",
            playerbase.MEANIE_ARM_FRAMES,
            (128 // enemies.MEANIE_ROTATE_STEP)
            * enemies.UPDATE_COOLDOWN_MEANIE_ROTATE
            * UNIT,
        ),
        # $16F2 FOV width $14 -> +-10 units.
        ("FOV_HALF", playerbase.FOV_HALF, enemies.FOV_SCAN // 2),
        # $11E0 auto-repeat mask: one gated scan skipped per set bit.
        (
            "CURSOR_RAMP",
            playerbase.CURSOR_RAMP,
            float(bin(playerbase.CURSOR_REPEAT_MASK).count("1")),
        ),
        # $E0 eye-height fraction of a tile unit.
        ("ROBOT_EYE", playerbase.ROBOT_EYE, 0xE0 / 256),
        # $1FA4/$86A5 dither loop cycles at the PAL frame.
        ("DITHER_FRAMES", actioncost.DITHER_FRAMES, 977904.0 / projector.FRAME_CYCLES),
        ("FRAME_CYCLES", projector.FRAME_CYCLES, 19656.0),
        # $357D view-less fallback: tune wait + fixed foreground.
        (
            "VIEWPOINT_REPLOT_FRAMES",
            actioncost.VIEWPOINT_REPLOT_FRAMES,
            projector.TUNE_TRANSFER_FRAMES + projector.SETTLE_FIXED_FRAMES,
        ),
        # dither loop + one post-action scene replot.
        (
            "SETTLE[create]",
            actioncost.SETTLE["create"],
            actioncost.FRAME_TICKS
            * (actioncost.DITHER_FRAMES + actioncost.POST_ACTION_REPLOT_FRAMES),
        ),
        (
            "SETTLE[absorb]",
            actioncost.SETTLE["absorb"],
            actioncost.FRAME_TICKS
            * (actioncost.DITHER_FRAMES + actioncost.POST_ACTION_REPLOT_FRAMES),
        ),
        # $1B2F EOR $80 flips 128 units on the +-AZIMUTH_STEP lattice.
        ("UTURN_STEP", aimcost.UTURN_STEP, 128 // aimcost.AZIMUTH_STEP),
        # $10EE 16-step h scroll + $1135 8-step v scroll per notch.
        (
            "_PAN_STALL_FRAMES",
            kbd_aim._PAN_STALL_FRAMES,
            playerbase.H_SCROLL + playerbase.V_SCROLL,
        ),
        # $3912 stores 24 bytes per iteration over 64, $38AD 32 over 40; each strip
        # clear calls its loop twice (odd then even X) at 5 cycles a store, 7 per tail.
        ("_CLEAR_CYCLES_H", pancost._CLEAR_CYCLES_H, 2 * 64 * (24 * 5 + 7)),
        ("_CLEAR_CYCLES_V", pancost._CLEAR_CYCLES_V, 2 * 40 * (32 * 5 + 7)),
        (
            "CLEAR_FRAMES[h]",
            pancost.CLEAR_FRAMES[0],
            pancost._CLEAR_CYCLES_H / projector.FRAME_CYCLES,
        ),
        # The jit twin inlines these as njit-visible globals; they cannot drift.
        ("_COOLDOWN_STICK", enemies_jit._COOLDOWN_STICK, enemies.COOLDOWN_STICK),
        ("_ENERGY_MASK", enemies_jit._ENERGY_MASK, mm.ENERGY_MASK),
    ],
)
def test_derived_constant_matches_primitive(name, derived, expected):
    assert derived == pytest.approx(expected), name


@pytest.mark.parametrize("d", range(aimcost.UTURN_STEP + 1))
def test_uturn_crossover_is_nine_steps(d):
    """h_press_count switches from d direct presses to 1 + (16 - d) at d == 9."""
    nu, ns = aimcost.h_press_count(0, (d * aimcost.AZIMUTH_STEP) & 0xFF)
    if d >= 9:
        assert (nu, ns) == (1, aimcost.UTURN_STEP - d)
        assert nu + ns < d
    else:
        assert (nu, ns) == (0, d)


@pytest.mark.xfail(
    strict=True,
    reason="_PAN_MAX_FRAMES=400 < a full pan 256 h + 208 v = 464 frames",
)
def test_pan_max_covers_full_pan():
    assert kbd_aim._PAN_MAX_FRAMES >= 256 + 208


@pytest.mark.xfail(
    strict=True,
    reason="HOP_FRAMES=700 != 2*SETTLE[create] + SETTLE[transfer] == 459.5",
)
def test_hop_frames_matches_claimed_composition():
    assert playerbase.HOP_FRAMES == pytest.approx(
        2 * actioncost.SETTLE["create"] + actioncost.SETTLE["transfer"]
    )
