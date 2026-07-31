"""Every DERIVED timing constant recomputed from the ROM primitive its comment
cites.  Each case names the primitive symbols, not the derived literal."""

import json
import os

import pytest

from driver import kbd_aim
from sentinel import actioncost, aimcost, enemies, enemies_jit, memmap as mm
from sentinel import pancost, passcost, playerbase, projector, relative
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


_LIVE_PASS_CYCLES = os.path.join(
    os.path.dirname(__file__), "fixtures", "live_pass_cycles.json"
)


def _live_cycles():
    with open(_LIVE_PASS_CYCLES, encoding="utf-8") as fh:
        return json.load(fh)


def _mode(hist):
    return int(max(hist.items(), key=lambda kv: kv[1])[0])


def test_the_priced_redraw_brackets_every_live_rotation_redraw():
    """Every live $1F9F is a $209B reject, and the modelled rejects bracket them.

    The redraw is bimodal, so no single number is right; what the live capture
    constrains is the off-screen branch, the only one 16 rotations ever took."""
    samples = [
        c
        for board, vals in _live_cycles()["rotation_redraw_1f9f"].items()
        if not board.startswith("_")
        for c in vals
    ]
    assert len(samples) >= 16
    modelled = []
    for digits in ("0042", "0335", "9795"):
        state = Game.typed(int(digits)).state
        for slot in range(mm.NUM_SLOTS):
            if state.obj_flags[slot] & 0x80:
                continue
            for h_angle in range(0, 256, 8):
                state.obj_h_angle[state.mem[mm.PLAYER_OBJECT]] = h_angle
                cycles, columns, _l, _s = relative.update_object_on_screen_cycles(
                    state, slot
                )
                if columns == 0:
                    modelled.append(cycles)
    assert min(modelled) <= min(samples) and max(samples) <= max(modelled)


def test_rotate_is_the_counted_straight_line_plus_its_measured_callees():
    """$1805..$1884 counted off the image, its three fixed callees measured live."""
    parts = _live_cycles()["rotate_parts"]
    straight = 32 + 12  # $1805..$1822 and $187B..$1884, the four JSR opcodes apart
    jsrs = 4 * 6  # $1AF4, $1973, $3470 and the $1881 JSR $1F9F charged separately
    assert passcost.MEANIE_INIT == parts["1973"]
    assert passcost.TUNE == parts["3470"]  # $3470 vectors through $FFF1, off the image
    assert (
        passcost.ROTATE
        == straight + jsrs + parts["1af4"] + parts["1973"] + passcost.TUNE
    )


def test_irq_cycles_is_the_measured_badline_steal_and_handler_time():
    """IRQ_CYCLES is the FIXED cycles a frame denies the play loop: 25 badlines, four
    short raster interrupts and the $9630 body, each measured on the machine."""
    irq = _live_cycles()["irq"]
    assert passcost.BADLINE_STEAL == _mode(irq["badline_steal"])
    assert passcost.BADLINES_PER_FRAME == irq["badlines_per_frame"]
    assert passcost.SHORT_IRQ == _mode(irq["short_wall"])
    steal = passcost.BADLINES_PER_FRAME * passcost.BADLINE_STEAL
    shorts = passcost.SHORT_IRQS_PER_FRAME * passcost.SHORT_IRQ
    assert passcost.IRQ_CYCLES == steal + shorts + passcost.IRQ_BODY
    fg = irq["foreground_cpu_per_frame"]
    cheap = passcost.FOREGROUND_CYCLES - passcost.COOLDOWN_TICK_NO_CARRY
    dear = (
        passcost.FOREGROUND_CYCLES
        - passcost.COOLDOWN_TICK_WALK
        - 24 * passcost.COOLDOWN_TICK_BYTE_DEC
    )
    assert cheap == fg["max"]  # the frame whose $130C does not carry
    assert dear <= fg["min"]  # every cooldown byte decrementing


def test_the_cooldown_tick_prices_every_live_130c_sample():
    """Each live $130C is its counted branch: no carry, gate decrement, or the walk."""
    for cyc, decs, loops, _n in _live_cycles()["irq"]["cooldown_130c"]["samples"]:
        if loops == 0:
            want = (
                passcost.COOLDOWN_TICK_NO_CARRY
                if cyc < passcost.COOLDOWN_TICK_GATE
                else passcost.COOLDOWN_TICK_GATE
            )
        else:
            want = (
                passcost.COOLDOWN_TICK_WALK
                + decs * passcost.COOLDOWN_TICK_BYTE_DEC
                + (loops - decs) * passcost.COOLDOWN_TICK_BYTE_STICK
            )
        assert abs(want - cyc) <= 1, (cyc, decs, loops, want)


def _mode_int(hist):
    """The most common key of a {value: count} histogram, as an int."""
    return int(max(hist.items(), key=lambda kv: kv[1])[0])


def test_the_sub_pass_segments_are_the_measured_split_of_one_pass():
    """A pass is charged in four segments; each is the mode of its live measurement."""
    seg = _live_cycles()["pass_segments"]["9795"]
    assert passcost.PASS_HEAD == _mode_int(seg["head_1289"])
    assert passcost.PASS_TAIL == _mode_int(seg["tail_12a2_less_191f"])
    assert passcost.LOOP_PASS == passcost.PASS_HEAD + passcost.PASS_TAIL
    prnd_cursor = _mode_int(seg["prnd_cursor_16d6"])
    assert passcost.UPDATE_PRND + passcost.UPDATE_CURSOR == prnd_cursor
    assert passcost.UPDATE_TAIL == prnd_cursor
    assert (
        passcost.UPDATE_TAIL_WRAP == passcost.UPDATE_PRND + passcost.UPDATE_CURSOR_WRAP
    )


def test_the_frame_boundary_lands_inside_the_pass_it_is_charged_against():
    """The raster IRQ interrupts a pass mid-body far more often than at its edge, so
    the loop cannot spend a pass atomically."""
    census = _live_cycles()["frame_boundary_segment"]
    for board in ("9795", "0335", "0042"):
        counts = census[board]
        total = sum(counts.values())
        assert total >= 1000
        at_edge = (counts.get("head_1289", 0) + counts.get("dispatch_16b5", 0)) / total
        assert at_edge < 0.10, (board, at_edge)
    frac = {
        b: census[b].get("consider_16e6", 0) / sum(census[b].values())
        for b in ("9795", "0335", "0042")
    }
    assert frac["9795"] > frac["0335"] > frac["0042"]
    assert frac["0042"] < 0.05


def test_the_split_loop_charges_every_segment_exactly_once():
    """Resuming mid pass must neither re-charge a segment nor skip one: over 400 frames
    the three segments are entered the same number of times, bar the one in flight."""
    state = Game.typed(335).state
    state.mem[mm.PLAYER_NOT_ACTED] = 0x00
    real = {
        n: getattr(enemies, n)
        for n in ("dispatch_cycles", "update_body", "update_cursor")
    }
    counts = dict.fromkeys(real, 0)

    def wrap(name):
        def call(*args):
            out = real[name](*args)
            # update_body is re-entered once per suspension; count only completions
            if name != "update_body" or out[1]:
                counts[name] += 1
            return out

        return call

    for name in real:
        setattr(enemies, name, wrap(name))
    try:
        enemies.advance_frames_python(state, 400)
    finally:
        for name, fn in real.items():
            setattr(enemies, name, fn)
    assert counts["dispatch_cycles"] > 400
    assert counts["dispatch_cycles"] - counts["update_body"] in (0, 1)
    assert counts["update_body"] - counts["update_cursor"] in (0, 1)
    assert state.pass_phase in (
        enemies.PHASE_HEAD,
        enemies.PHASE_BODY,
        enemies.PHASE_CURSOR,
    )


@pytest.mark.oracle
def test_the_status_bar_cost_is_a_function_of_the_energy_byte():
    """$9508 pads to fixed columns, so passcost recomputes it from the energy alone."""
    from sentinel.tests import oracle  # pylint: disable=import-outside-toplevel

    cpu, mem, state = oracle.generate_machine(0x0042)
    oracle.prime_enemy_driver(cpu, mem, state)
    with open(oracle.IMG, "rb") as fh:
        mem[0x9508] = fh.read()[0x9508]  # un-stub plot_status_bar
    for energy in range(40):
        mem[mm.PLAYER_ENERGY] = energy
        state["stop"] = False
        c0 = cpu.processorCycles
        oracle.call(cpu, mem, 0x9508, state=state)
        assert cpu.processorCycles - c0 == passcost.status_bar_cycles(energy), energy


@pytest.mark.oracle
def test_the_drain_cost_model_matches_the_roms_own_1a08(monkeypatch):
    """$1A08 priced per target kind against the real 6502: player, robot, tree, boulder.

    $3470 is stubbed by the oracle (it vectors off the image); $9508 is not."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel
    from sentinel.tests import oracle  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(passcost, "TUNE", 6)
    cpu, mem, state = oracle.generate_machine(0x0335)
    oracle.prime_enemy_driver(cpu, mem, state)
    with open(oracle.IMG, "rb") as fh:
        mem[0x9508] = fh.read()[0x9508]
    base = State.from_mem(bytes(mem))
    kinds = {}
    for slot in range(mm.NUM_SLOTS):
        if base.obj_flags[slot] & 0x80 or slot == mem[mm.PLAYER_OBJECT]:
            continue
        kinds.setdefault(int(base.obj_type[slot]), slot)
    cases = [(mem[mm.PLAYER_OBJECT], 0x40)] + [
        (slot, flags)
        for otype, slot in sorted(kinds.items())
        for flags in ((0x00, 0x41) if otype == mm.T_TREE else (None,))
    ]
    for slot, flags in cases:
        snap = bytes(mem)
        if flags is not None and slot != mem[mm.PLAYER_OBJECT]:
            mem[mm.OBJECTS_FLAGS + slot] = flags
        mem[mm.TARGETED_OBJECT_SLOT] = slot
        st = State.from_mem(bytes(mem))
        _drained, model = enemies._reduce_object_energy(st, slot, 0)
        state["stop"] = False
        c0 = cpu.processorCycles
        oracle.call(cpu, mem, 0x1A08, state=state)
        assert cpu.processorCycles - c0 == model, (slot, flags)
        mem[:] = snap
    assert len(cases) >= 4


@pytest.mark.oracle
@pytest.mark.parametrize("landscape", (0x0042, 0x0335, 0x9795))
def test_the_body_cost_model_matches_the_roms_own_16e6_cycle_count(
    landscape, monkeypatch
):
    """$16E6 priced by passcost against the real 6502, every round, exactly.

    Gated, marching, rotating, held-target and draining rounds alike; a slot the play
    loop would not dispatch ($16BB/$16CC) has no body to compare.  The ROM falls through
    into $16D6, so its call costs UPDATE_TAIL more; $1F9F/$3470 are stubbed with RTS."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel
    from sentinel.tests import oracle  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(
        relative, "update_object_on_screen_cycles", lambda state, target: (6, 0, 0, 0)
    )
    monkeypatch.setattr(passcost, "ROTATE", passcost.ROTATE - 323 + 6)
    monkeypatch.setattr(passcost, "MEANIE_ROTATE", passcost.MEANIE_ROTATE - 323 + 6)
    cpu, mem, state = oracle.generate_machine(landscape)
    oracle.prime_enemy_driver(cpu, mem, state)
    mem[mm.PLAYER_NOT_ACTED] = 0
    checked = 0
    for _ in range(400):
        state["stop"] = False
        oracle.call(cpu, mem, oracle.TICK_COOLDOWNS, state=state)
        x = mem[mm.CURSOR]
        st = State.from_mem(bytes(mem))
        dispatched = st.obj_type[x] in mm.ENEMY_TYPES and not st.obj_flags[x] & 0x80
        model = enemies.UNBOUNDED - enemies.update_body(st, enemies.UNBOUNDED)[0]
        c0 = cpu.processorCycles
        state["stop"] = False
        oracle.call(cpu, mem, 0x16E6, x=x, state=state)
        tail = passcost.UPDATE_TAIL_WRAP if x == 0 else passcost.UPDATE_TAIL
        rom = cpu.processorCycles - c0 - tail
        mem[mm.CURSOR] = (mem[mm.CURSOR] - 1) & 7
        if dispatched:
            assert model == rom, f"ls{landscape:04x} slot {x}: {model} != {rom}"
            checked += 1
    assert checked > 50


def _aim_at(state, observer, target):
    """The observer facing that puts ``target`` dead ahead ($0C57 == $0A)."""
    ra = relative.relative_angles(state, observer, target)
    return (int(state.obj_h_angle[observer]) + ra["c57"] - 0x0A) & 0xFF


@pytest.mark.oracle
@pytest.mark.parametrize("landscape", (0x0042, 0x0335, 0x9795))
def test_the_see_cost_model_matches_the_roms_own_1887(landscape):
    """$1887 priced and answered from state against the real 6502, per call.

    Every occupied slot as target, as the robot the $17B2 scan asks for and as its own
    type (the $1AB0 tree/boulder call, the only $1887 whose $18DA takes the one-probe
    branch), from every enemy aimed at the player and turned away from it -- so the
    $1893/$189D rejects, the $18CA FOV reject, the partial and the full-sight marches
    are all covered.  The exposure byte $0014 the answer rides on is compared too."""
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
                    mem[mm.FOV_WIDTH] = enemies.FOV_SCAN  # $16F2
                    st = State.from_mem(bytes(mem))
                    see = relative.can_see_object(
                        st, e, target, etype, enemies.FOV_SCAN
                    )
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
def test_the_tile_scan_cost_model_matches_the_roms_own_1ab0(landscape):
    """$1AB0 priced and decided from state against the real 6502, call for call.

    A freshly generated board stacks nothing, so its $1AB0 only ever walks the empty
    ($1AB5), rejected ($1AC0) and wrong-top ($1AE1) exits -- the drain half of the loop
    ($1AC2 tile fetch, the $1AE3 JSR $1887 and the $1AEA hit) needs an object standing
    on a stack, so the sweep restacks every top-of-tile tree/boulder as one.  A direct
    call is first checked against $1AB0 as the play round itself reaches it."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel

    oracle = _oracle()
    cpu, mem, mstate = oracle.generate_machine(landscape)
    oracle.prime_enemy_driver(cpu, mem, mstate)
    mem[mm.PLAYER_NOT_ACTED] = 0

    def rom_call(image, enemy):
        c, m, ms = oracle.wrap_image(image)
        m[0x6E] = enemy
        c0 = c.processorCycles
        ms["stop"] = False
        oracle.call(c, m, 0x1AB0, state=ms)
        return c.processorCycles - c0, c.p & 1, m[mm.TARGETED_OBJECT_SLOT]

    def model_call(image, enemy):
        st = State.from_mem(bytes(image))
        budget, _index, tb = enemies._find_drainable_boulder_or_tree(
            st, enemy, enemies.UNBOUNDED, mm.NUM_SLOTS - 1
        )
        return passcost.TILE_SCAN_ENTRY + enemies.UNBOUNDED - budget, tb

    # the direct call is the same call the round makes: same image, same cycles
    inplay = 0
    for _ in range(300):
        if inplay >= 2:
            break
        mstate["stop"] = False
        oracle.call(cpu, mem, oracle.TICK_COOLDOWNS, state=mstate)
        for hit in _trace_calls(cpu, mem, mstate, oracle, 0x1AB0):
            assert rom_call(hit[1], hit[2])[0] == hit[0]
            inplay += 1
    assert inplay >= 2

    snap = bytes(mem)
    base = State.from_mem(snap)
    tops = [
        s
        for s in range(mm.NUM_SLOTS)
        if not base.obj_flags[s] & 0x80
        and base.obj_type[s] in (mm.T_TREE, mm.T_BOULDER)
        and enemies._tile_top(base, s) == s
    ]
    assert tops
    outcomes = set()
    for otype in (mm.T_TREE, mm.T_BOULDER):
        for e in range(8):
            if base.obj_flags[e] & 0x80 or base.obj_type[e] not in mm.ENEMY_TYPES:
                continue
            aim = _aim_at(base, e, tops[0])
            for facing in (aim, (aim + 96) & 0xFF, (aim + 160) & 0xFF):
                image = bytearray(snap)
                for s in tops:
                    image[mm.OBJECTS_FLAGS + s] = 0x40  # $1AB9: standing on a stack
                    image[mm.OBJECTS_TYPE + s] = otype
                image[mm.OBJECTS_H_ANGLE + e] = facing
                image[0x6E] = e
                image[mm.FOV_WIDTH] = enemies.FOV_SCAN
                model, tb = model_call(image, e)
                rom, carry, slot = rom_call(bytes(image), e)
                assert model == rom, f"ls{landscape:04x} enemy {e} type {otype}"
                assert (tb >= 0) == (carry == 0)  # $1AED CLC hit / $1AF2 SEC exhausted
                if tb >= 0:
                    assert tb == slot
                outcomes.add(tb >= 0)
    assert outcomes == {True, False}


def _trace_calls(cpu, mem, mstate, oracle, target):
    """[(cycles, image at entry, $6E)] for each ``target`` call one $16B5 round makes."""
    ret, hits, inside = 0xFFF0, [], None
    mem[ret] = 0x60
    cpu.a = cpu.x = cpu.y = 0
    sp = cpu.sp
    mem[0x0100 + sp] = (ret - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (ret - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = oracle.UPDATE_ENEMIES
    mstate["stop"] = False
    while cpu.pc != ret and not mstate["stop"]:
        if inside is None and cpu.pc == target:
            inside = (cpu.sp, cpu.processorCycles, bytes(mem), mem[0x6E])
        elif inside is not None and cpu.sp > inside[0]:
            hits.append((cpu.processorCycles - inside[1], inside[2], inside[3]))
            inside = None
        cpu.step()
    return hits


@pytest.mark.oracle
@pytest.mark.xfail(
    strict=True,
    reason="the body applies a segment's CORE writes at the segment's start: $16ED "
    "lands 15 cycles early, $1809/$1813 65/72, $1825/$1835 66/89 -- open_items 8",
)
@pytest.mark.parametrize("landscape", (0x0042, 0x0335, 0x9795))
def test_the_body_commits_its_core_writes_at_the_roms_own_cycle(landscape):
    """The `$16ED` reload must not appear in the sim before the ROM has paid for it.

    `$16E6 LDA $0C30,X` 4 + `CMP #2` 2 + `BCS` not taken 2 + `$16ED LDA #4` 2 +
    `$16EF STA $0C30,X` 5: the machine's `update_cd` becomes 4 at cycle 15 of the body,
    so a model spending fewer than 15 must still read the old value."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel

    oracle = _oracle()
    cpu, mem, mstate = oracle.generate_machine(landscape)
    oracle.prime_enemy_driver(cpu, mem, mstate)
    checked = 0
    for e in range(8):
        if mem[mm.OBJECTS_FLAGS + e] & 0x80:
            continue
        if mem[mm.OBJECTS_TYPE + e] not in mm.ENEMY_TYPES:
            continue
        image = bytearray(mem)
        image[mm.ENEMIES_UPDATE_COOLDOWN + e] = 0
        image[mm.CURSOR] = e
        c, m, ms = oracle.wrap_image(bytes(image))
        c.a, c.x, c.y = 0, e, 0
        c.pc = 0x16E6
        c0, at = c.processorCycles, None
        while at is None and not ms["stop"]:
            c.step()
            if m[mm.ENEMIES_UPDATE_COOLDOWN + e] == enemies.UPDATE_COOLDOWN_SCAN:
                at = c.processorCycles - c0
        assert at == 15
        for budget in range(1, at):
            st = State.from_mem(bytes(image))
            enemies.update_body(st, budget)
            assert (
                st.mem[mm.ENEMIES_UPDATE_COOLDOWN + e] == 0
            ), f"ls{landscape:04x} enemy {e}: $16ED committed on {budget} cycles"
        checked += 1
    assert checked


@pytest.mark.oracle
@pytest.mark.parametrize("landscape", (0x0042, 0x0335, 0x9795))
def test_the_body_split_is_exact_wherever_the_frame_ends(landscape, monkeypatch):
    """Suspending $16E6 anywhere costs and does exactly what running it whole does.

    The body is charged in units that are not one instruction wide, so a frame boundary
    inside one leaves the model mid-unit; every resume point must still spend the same
    total and leave the same CORE bytes, including the ones where the resume re-runs an
    $1887 whose cycles are already paid ($0C56/$0CDD/$0C76 rotate on every call)."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(
        relative, "update_object_on_screen_cycles", lambda state, target: (6, 0, 0, 0)
    )
    fields = (
        (mm.OBJECTS_FLAGS, mm.NUM_SLOTS),
        (mm.OBJECTS_TYPE, mm.NUM_SLOTS),
        (mm.OBJECTS_H_ANGLE, mm.NUM_SLOTS),
        (mm.ENEMIES_UPDATE_COOLDOWN, 8),
        (mm.ENEMIES_DRAINING_COOLDOWN, 8),
        (mm.ENEMIES_TARGETED_OBJECT, 8),
        (mm.ENEMIES_ROTATION_COOLDOWN, 8),
        (mm.ENEMIES_MEANIE_OBJECT, 8),
    )

    def core(st):
        return tuple(bytes(st.mem[a : a + n]) for a, n in fields)

    def run(image, enemy, first):
        st = State.from_mem(bytes(image))
        budget, stage, index, partial = enemies._consider_enemy_state(
            st, enemy, first, enemies.BODY_ENTRY, 0, -1
        )
        spent = first - budget
        while stage != enemies.BODY_DONE:
            grant = budget + enemies.UNBOUNDED
            budget, stage, index, partial = enemies._consider_enemy_state(
                st, enemy, grant, stage, index, partial
            )
            spent += grant - budget
        return spent, core(st)

    mem = bytearray(_oracle().generate(landscape))
    base = State.from_mem(bytes(mem))
    player = mem[mm.PLAYER_OBJECT]
    checked = 0
    for e in range(8):
        if base.obj_flags[e] & 0x80 or base.obj_type[e] not in mm.ENEMY_TYPES:
            continue
        aim = _aim_at(base, e, player)
        for off in (0, 5, 96):
            image = bytearray(mem)
            image[mm.OBJECTS_H_ANGLE + e] = (aim + off) & 0xFF
            image[mm.ENEMIES_UPDATE_COOLDOWN + e] = 0
            image[mm.CURSOR] = e
            whole = run(image, e, enemies.UNBOUNDED)
            for first in range(1, whole[0] + 1, max(1, whole[0] // 40)):
                assert (
                    run(image, e, first) == whole
                ), f"ls{landscape:04x} enemy {e} split at {first}"
                checked += 1
    assert checked > 100


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
    from sentinel.tests import oracle  # pylint: disable=import-outside-toplevel

    return oracle


def _run(cpu, mem, mstate, addr):
    """One JSR-style call at `addr` on the oracle machine; returns its cycles."""
    mstate["stop"] = False
    c0 = cpu.processorCycles
    _oracle().call(cpu, mem, addr)
    return cpu.processorCycles - c0


@pytest.mark.oracle
@pytest.mark.parametrize("landscape", (0x0042, 0x0335, 0x9795))
def test_the_strip_replot_backend_is_the_roms_own_1f9f(landscape, monkeypatch):
    """RENDER_COST_BACKEND=py65 prices $1FA4..$1F9E by running the real $1F9F.

    The proxy prices the same camera through the same $29C7 window and adds the $2211
    strip clear, the $9730 buffer flush and $1F9F's own line, all off the uncapped
    $211B span; the residual left is the $2625 area-fill proxy's."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel

    mem = bytearray(_oracle().generate(landscape))
    state = State(mem)
    player = mem[mm.PLAYER_OBJECT]
    checked = 0
    for h_angle in range(0, 256, 16):
        state.obj_h_angle[player] = h_angle
        for slot in range(8):
            if state.obj_flags[slot] & 0x80:
                continue
            visible, left, cols, span, _cyc = relative.object_screen_span(state, slot)
            if not visible:
                continue
            monkeypatch.setenv("RENDER_COST_BACKEND", "py65")
            exact = projector.strip_replot_frames(state, slot, left, cols, span)
            monkeypatch.setenv("RENDER_COST_BACKEND", "proxy")
            proxy = projector.strip_replot_frames(state, slot, left, cols, span)
            assert exact > 1.0  # the replot is always many frames, never a redraw
            assert 0.98 <= proxy / exact <= 1.00
            checked += 1
            break
        if checked >= 2:
            break
    assert checked >= 1


def _rom_replot_line(state, target):
    """The ROM's own $1FA4..$1F9E on ``target``, less every $1FFC JSR $2625 under it."""
    oracle = _oracle()
    cpu, mem, mstate = oracle.machine_from_image(state.mem)
    plot = [0]
    inside = [None, 0]

    def trace(pc):
        if inside[0] is None:
            if pc == oracle.PLOT_WORLD:
                inside[0], inside[1] = cpu.sp, cpu.processorCycles
        elif cpu.sp > inside[0]:
            plot[0] += cpu.processorCycles - inside[1]
            inside[0] = None

    frames = oracle.update_object_cost(
        cpu, mem, mstate, target, trace=trace, from_pc=oracle.REPLOT_LINE
    )
    return round(frames * oracle.FRAME_CYCLES) - plot[0], mem


@pytest.mark.oracle
@pytest.mark.parametrize("landscape", (0x0042, 0x0335, 0x9795))
def test_the_strip_replot_line_is_the_roms_own_1fa4(landscape):
    """$1FA4..$1F9E priced from state, cycle for cycle, less the $2625 chunks.

    The $2211 clear and the $207E $9730 flush both run over the uncapped $211B span,
    the $29C7 window and the $1FC2 camera shift once per <=20-column chunk."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel

    mem = bytearray(_oracle().generate(landscape))
    state = State(mem)
    player = mem[mm.PLAYER_OBJECT]
    checked = 0
    for h_angle in range(0, 256, 8):
        state.obj_h_angle[player] = h_angle
        for slot in range(8):
            if state.obj_flags[slot] & 0x80:
                continue
            visible, left, cols, span, _cyc = relative.object_screen_span(state, slot)
            if not visible:
                continue
            rom, rom_mem = _rom_replot_line(state, slot)
            assert (rom_mem[0x211B], rom_mem[0x211C]) == (span, left)
            chunks = projector.replot_chunks(left, span)
            assert chunks[0][1] == cols
            model = projector.strip_line_cycles(
                span, len(chunks), state.mem[projector.SCREEN_SCROLL]
            )
            assert model == rom, (h_angle, slot, span, model, rom)
            checked += 1
            break
        if checked >= 2:
            break
    assert checked >= 2


def test_the_strip_buffer_window_is_the_roms_own_29c7():
    """$29C7 halves the $0C69 column count into $0007, folds it into $0012 and rotates
    the halving's carry into $0028, so an odd strip drops the left column's near half.

    A 40-column A is the whole play screen, which is $2993 mode 0's own triple."""
    assert projector.strip_window(40) == projector.BUF_WINDOW[projector.PLAY_MODE]
    for columns in range(1, 21):
        left, right, frac = projector.strip_window(columns)
        assert (left, right) == (columns >> 1, ((columns >> 1) >> 1) ^ 0x80)
        assert frac == (0x80 if columns & 1 else 0x00)
        assert projector._window_columns((left, right, frac)) == columns
