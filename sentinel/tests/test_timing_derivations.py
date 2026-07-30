"""Every DERIVED timing constant recomputed from the ROM primitive its comment
cites.  Each case names the primitive symbols, not the derived literal."""

import json
import os

import pytest

from driver import kbd_aim
from sentinel import actioncost, aimcost, enemies, enemies_jit, memmap as mm
from sentinel import pancost, passcost, playerbase, projector
from sentinel.game import Game

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


def _units_turned(landscape):
    """Bearing units each enemy actually sweeps in one REVOLUTION_FRAMES period."""
    st = Game.typed(landscape).state.clone()
    st.mem[mm.PLAYER_NOT_ACTED] = 0
    slots = list(enemies.enemy_slots(st))
    turned = {e: 0 for e in slots}
    was = {e: int(st.obj_h_angle[e]) for e in slots}
    for _ in range(int(playerbase.REVOLUTION_FRAMES)):
        enemies.advance_frame(st)
        for e in slots:
            now = int(st.obj_h_angle[e])
            turned[e] += min((now - was[e]) % 256, (was[e] - now) % 256)
            was[e] = now
    return turned


def test_revolution_steps_cover_a_turn_at_the_rom_rotation_step():
    """14 rotations x the +-20 $1813 step is 280 units, against the 256 a circle needs."""
    for landscape in (0, 42, 110, 335):
        st = Game.typed(landscape).state
        steps = {
            abs(playerbase._signed(st.mem[mm.ROTATION_SPEED_TABLE + e]))
            for e in enemies.enemy_slots(st)
        }
        assert steps and all(s * playerbase.REVOLUTION_STEPS >= 256 for s in steps)


@pytest.mark.xfail(
    strict=True,
    reason="the rotation STALL: a draining or gated enemy skips its $17F9 rotate, so "
    "ls0 slot 0 turns 120 of 280 units per REVOLUTION_FRAMES, ls110 slot 0 turns 80 "
    "and ls335 slots 1/5/6 turn 220/120/180 -- why _verify_starts re-asks the clock",
)
def test_revolution_frames_is_a_full_cone_revolution():
    """One calendar period would sweep every cone past all 256 bearings -- if it did,
    a "never busy" span would mean never rather than not yet."""
    for landscape in (0, 42, 110, 335):
        turned = _units_turned(landscape)
        assert all(t >= 256 for t in turned.values()), (landscape, turned)


_LIVE_PASS_RATE = os.path.join(
    os.path.dirname(__file__), "fixtures", "live_pass_rate.json"
)


def test_irq_cycles_matches_the_live_pass_rate():
    """IRQ_CYCLES is the complement of the counted foreground: on every board the
    modelled idle cadence must land inside the live-measured pass-count bracket."""
    with open(_LIVE_PASS_RATE, encoding="utf-8") as fh:
        boards = json.load(fh)["boards"]
    for digits, rec in boards.items():
        st = Game.typed(int(digits)).state
        counts = [int(k) for k in rec["idle_passes_per_frame"]]
        modelled = passcost.FOREGROUND_CYCLES / passcost.idle_pass_cycles(st.mem)
        assert min(counts) <= modelled <= max(counts), (
            f"ls{digits}: modelled {modelled:.2f} passes/frame outside the live "
            f"bracket {min(counts)}..{max(counts)}"
        )
