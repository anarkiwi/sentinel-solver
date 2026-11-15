"""Derived pricing and sight constants recomputed from the ROM primitives they cite,
and the sight and screen-span cost models against the real 6502."""

import pytest

from driver import kbd_aim
from sentinel import aimcost, memmap as mm, pancost, passcost, projector, relative
from sentinel import settlecost, views
from sentinel.tests import oracle as _oracle_module

FOV_SCAN = 0x14  # $16F2: an enemy's horizontal FOV width during a scan


@pytest.mark.parametrize(
    "name,derived,expected",
    [
        # $11E0 auto-repeat mask: one gated scan skipped per set bit.
        (
            "CURSOR_RAMP",
            views.CURSOR_RAMP,
            float(bin(views.CURSOR_REPEAT_MASK).count("1")),
        ),
        # $E0 eye-height fraction of a tile unit.
        ("ROBOT_EYE", views.ROBOT_EYE, 0xE0 / 256),
        ("FRAME_CYCLES", projector.FRAME_CYCLES, 312.0 * 63),
        # $95E9 settle body: flat/carry frames mixed at the $1310 205/256 Bresenham.
        (
            "SETTLE_BODY",
            settlecost.SETTLE_BODY,
            settlecost.SETTLE_BODY_FLAT
            + (mm.COOLDOWN_BRESENHAM_STEP / 256.0)
            * (settlecost.SETTLE_BODY_CARRY - settlecost.SETTLE_BODY_FLAT),
        ),
        # PAL frame less the four short IRQs, the settle body and the machine steal.
        (
            "SETTLE_FG",
            settlecost.SETTLE_FG,
            passcost.PAL_FRAME_CYCLES
            - passcost.SHORT_IRQ_FRAME
            - settlecost.SETTLE_BODY
            - settlecost.SETTLE_STEAL,
        ),
        # tap_action's own idle+press scans, the count TAP_FRAMES pins.
        ("BRACKET_FRAMES", settlecost.BRACKET_FRAMES, float(views.TAP_FRAMES)),
        # tap bracket + the $AB50-decoded #$28 tune hold.
        (
            "UTURN_FLOOR_FRAMES",
            views.UTURN_FLOOR_FRAMES,
            settlecost.BRACKET_FRAMES + settlecost.TUNE_FRAMES[0x28],
        ),
        # $1B2F EOR $80 flips 128 units on the +-AZIMUTH_STEP lattice.
        ("UTURN_STEP", aimcost.UTURN_STEP, 128 // aimcost.AZIMUTH_STEP),
        # $10EE 16-step h scroll + $1135 8-step v scroll per notch.
        (
            "_PAN_STALL_FRAMES",
            kbd_aim._PAN_STALL_FRAMES,
            views.H_SCROLL + views.V_SCROLL,
        ),
        # $3912 / $38AD strip clears: two loop passes (odd, even X), 5 cycles a store, 7 a tail
        ("_CLEAR_CYCLES_H", pancost._CLEAR_CYCLES_H, 2 * 64 * (24 * 5 + 7)),
        ("_CLEAR_CYCLES_V", pancost._CLEAR_CYCLES_V, 2 * 40 * (32 * 5 + 7)),
        (
            "CLEAR_FRAMES[h]",
            pancost.CLEAR_FRAMES[0],
            pancost._CLEAR_CYCLES_H / settlecost.SETTLE_FG,
        ),
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


def _aim_at(state, observer, target):
    """The observer facing that puts ``target`` dead ahead ($0C57 == $0A)."""
    ra = relative.relative_angles(state, observer, target)
    return (int(state.obj_h_angle[observer]) + ra["c57"] - 0x0A) & 0xFF


@pytest.mark.oracle
@pytest.mark.parametrize("landscape", (0x0042, 0x0335, 0x9795))
def test_the_see_cost_model_matches_the_roms_own_1887(landscape):
    """$1887 priced and answered from state against the real 6502, per call.

    Every occupied slot as target, as the $17B2 robot and as its own type (the $1AB0 one-probe
    call), from every enemy aimed at and away from the player: covers the $1893/$189D/$18CA
    rejects and the partial and full-sight marches, and compares the exposure byte $0014.
    """
    from sentinel.state import State  # pylint: disable=import-outside-toplevel

    oracle = _oracle()
    cpu, mem, mstate = oracle.generate_machine(landscape)
    oracle.prime_enemy_driver(cpu, mem, mstate)
    snap = bytes(mem)
    base = State.from_mem(snap)
    player = snap[mm.PLAYER_OBJECT]
    seen = set()
    for e in range(8):
        if base.obj_flags[e] & 0x80 or base.obj_type[e] not in mm.ENEMY_TYPES:
            continue
        aim = _aim_at(base, e, player)
        for facing in (aim, (aim + 128) & 0xFF):
            for target in range(mm.NUM_SLOTS):
                for etype in (mm.T_ROBOT, int(base.obj_type[target])):
                    mem[:] = bytearray(snap)
                    mem[mm.OBJECTS_H_ANGLE + e] = facing
                    mem[0x6E] = e  # $16FF/$1773: the observer $8401 reads
                    mem[mm.FOV_WIDTH] = FOV_SCAN  # $16F2
                    st = State.from_mem(bytes(mem))
                    see = relative.can_see_object(st, e, target, etype, FOV_SCAN)
                    c0 = cpu.processorCycles
                    mstate["stop"] = False
                    oracle.call(cpu, mem, 0x1887, a=etype, y=target, state=mstate)
                    rom = cpu.processorCycles - c0
                    assert see["cycles"] == rom, (
                        f"ls{landscape:04x} obs {e} tgt {target} type {etype}: "
                        f"{see['cycles']} != {rom}"
                    )
                    assert st.mem[0x0014] == mem[0x0014]
                    seen.add(
                        "empty"
                        if base.obj_flags[target] & 0x80
                        else (
                            "wrong_type"
                            if not see["in_slot"]
                            else (
                                "reject"
                                if not see["in_fov"]
                                else ("full" if see["full"] else "partial")
                            )
                        )
                    )
    assert seen == {"empty", "wrong_type", "reject", "partial", "full"}


@pytest.mark.oracle
@pytest.mark.parametrize("landscape", (0x0042, 0x0335, 0x9795))
def test_the_object_screen_span_is_exact_against_the_roms_own_209b(landscape):
    """$209B priced and decided from state, cycle for cycle, against the real 6502.

    Every occupied slot at every 8th player facing, so all four exits ($209F the
    player, $20D6 off the right, $20EB left of the view, $2103 zero width) and the
    on-screen branch are covered; the whole $1F9F is then checked on the rejects."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel

    cpu, mem, mstate = _oracle().generate_machine(landscape)
    mem[0xFFF0] = 0x60
    player = mem[mm.PLAYER_OBJECT]
    seen = set()
    for slot in list(range(8)) + [player]:
        if mem[mm.OBJECTS_FLAGS + slot] & 0x80 and slot != player:
            continue
        for h_angle in range(0, 256, 8):
            mem[mm.OBJECTS_H_ANGLE + player] = h_angle
            mem[0x0091], mem[0x006E] = slot, player
            rom = _run(cpu, mem, mstate, 0x209B)
            rom_visible = not cpu.p & 0x01
            state = State(bytearray(mem))
            visible, left, columns, span, cycles = relative.object_screen_span(
                state, slot
            )
            assert (cycles, visible) == (rom, rom_visible), (slot, h_angle)
            if visible:
                assert (left, columns, span) == (
                    mem[0x0C62],
                    mem[0x0C69],
                    mem[0x211B],
                )
                seen.add("visible")
                continue
            seen.add("player" if slot == player else "reject")
            rom = _run(cpu, mem, mstate, 0x1F9F)
            state = State(bytearray(mem))
            assert relative.update_object_on_screen_cycles(state, slot) == (
                rom,
                0,
                0,
                0,
            )
    assert seen == {"visible", "reject", "player"}


def _oracle():
    return _oracle_module


def _run(cpu, mem, mstate, addr):
    """One JSR-style call at `addr` on the oracle machine; returns its cycles."""
    mstate["stop"] = False
    c0 = cpu.processorCycles
    _oracle().call(cpu, mem, addr)
    return cpu.processorCycles - c0
