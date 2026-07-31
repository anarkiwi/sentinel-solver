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


_LIVE_PASS_CYCLES = os.path.join(
    os.path.dirname(__file__), "fixtures", "live_pass_cycles.json"
)


def _live_cycles():
    with open(_LIVE_PASS_CYCLES, encoding="utf-8") as fh:
        return json.load(fh)


def _mode(hist):
    return int(max(hist.items(), key=lambda kv: kv[1])[0])


def test_rotate_redraw_matches_the_live_object_redraw():
    """ROTATE_REDRAW is the mean $1F9F a rotation forces, over every live rotation."""
    samples = [
        c
        for board, vals in _live_cycles()["rotation_redraw_1f9f"].items()
        if not board.startswith("_")
        for c in vals
    ]
    assert len(samples) >= 16
    assert min(samples) <= passcost.ROTATE_REDRAW <= max(samples)
    assert abs(passcost.ROTATE_REDRAW - sum(samples) / len(samples)) <= 1.0


def test_rotate_is_the_counted_straight_line_plus_its_measured_callees():
    """$1805..$1884 counted off the image, its three fixed callees measured live."""
    parts = _live_cycles()["rotate_parts"]
    straight = 32 + 12  # $1805..$1822 and $187B..$1884, the four JSR opcodes apart
    jsrs = 4 * 6  # $1AF4, $1973, $3470 and the $1881 JSR $1F9F charged separately
    assert passcost.MEANIE_INIT == parts["1973"]
    assert passcost.TUNE_ROTATE == parts["3470"]  # counted below, corroborated live
    assert (
        passcost.ROTATE
        == straight + jsrs + parts["1af4"] + parts["1973"] + passcost.TUNE_ROTATE
    )


def test_irq_cycles_is_the_measured_badline_steal_and_handler_time():
    """IRQ_CYCLES is the FIXED cycles a frame denies the play loop: 25 badlines, four
    short raster interrupts and the $9630 body, each measured on the machine."""
    irq = _live_cycles()["irq"]
    assert passcost.BADLINE_STEAL == _mode(irq["badline_steal"])
    assert passcost.BADLINES_PER_FRAME == irq["badlines_per_frame"]
    assert passcost.SHORT_IRQ == _mode(irq["short_wall"])
    steal = passcost.BADLINES_PER_FRAME * passcost.BADLINE_STEAL
    shorts = (
        passcost.SHORT_IRQS_PER_FRAME * passcost.SHORT_IRQ + passcost.SHORT_IRQ_WRAP
    )
    assert passcost.IRQ_CYCLES == steal + shorts + passcost.IRQ_BODY
    fg = irq["foreground_cpu_per_frame"]
    cheap = passcost.FOREGROUND_CYCLES - passcost.COOLDOWN_TICK_NO_CARRY
    dear = (
        passcost.FOREGROUND_CYCLES
        - passcost.COOLDOWN_TICK_WALK
        - 24 * passcost.COOLDOWN_TICK_BYTE_DEC
    )
    # The cheapest frame is the one whose $130C does not carry.  The fixture's own
    # short_wall histogram is 3:1 over 119:120 -- one split entry a frame wraps its
    # index and costs SHORT_IRQ_WRAP more -- so its foreground max is that 1 high.
    assert cheap == fg["max"] - passcost.SHORT_IRQ_WRAP
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
@pytest.mark.parametrize(
    "tune,want",
    ((0, "TUNE_ROTATE"), (1, "TUNE_ROTATE"), (5, "TUNE_DRAIN")),
)
def test_the_tune_cost_is_counted_off_the_image(tune, want):
    """$3470 ends in JMP $FFF1, which the image carries as JMP $8D81 in the game's RAM.

    So the tune is countable after all: index 0 (a rotation) and 1 (a meanie's turn)
    cost TUNE_ROTATE, and index 5 -- the tune a drain starts at $1A1F -- costs more."""
    from sentinel.tests import oracle  # pylint: disable=import-outside-toplevel

    cpu, mem, state = oracle.generate_machine(0x0042)
    oracle.prime_enemy_driver(cpu, mem, state)
    with open(oracle.IMG, "rb") as fh:
        img = fh.read()
    mem[0x3470] = img[0x3470]  # un-stub start_tune
    for addr in range(0xE000, 0x10000):  # the vector page the ROM banks in
        mem[addr] = img[addr]
    state["stop"] = False
    c0 = cpu.processorCycles
    oracle.call(cpu, mem, 0x3470, a=tune, state=state)
    assert cpu.processorCycles - c0 == getattr(passcost, want)


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
def test_the_drain_cost_model_matches_the_roms_own_1a08():
    """$1A08 priced per target kind against the real 6502: player, robot, tree, boulder.

    $9508 and $3470 are un-stubbed for it: both are countable on the image."""
    from sentinel.state import State  # pylint: disable=import-outside-toplevel
    from sentinel.tests import oracle  # pylint: disable=import-outside-toplevel

    cpu, mem, state = oracle.generate_machine(0x0335)
    oracle.prime_enemy_driver(cpu, mem, state)
    with open(oracle.IMG, "rb") as fh:
        img = fh.read()
    mem[0x9508], mem[0x3470] = img[0x9508], img[0x3470]
    for addr in range(0xE000, 0x10000):
        mem[addr] = img[addr]
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
def test_the_sub_pass_position_is_read_back_off_the_stack_frame():
    """Interrupt a real $16B5 anywhere and the model can name where the ROM is.

    Pushes the $95E9 frame the raster IRQ would push, asks
    ``enemies.resume_from_stack`` for the resume point, and checks it against the ROM's
    own PC -- including the scan index, which survives a call in $0C58."""
    from sentinel.tests import oracle  # pylint: disable=import-outside-toplevel

    cpu, mem, state = oracle.generate_machine(0x9795)
    oracle.prime_enemy_driver(cpu, mem, state)
    mem[mm.PLAYER_NOT_ACTED] = 0
    direct = called = prnd = 0
    for stop in range(1, 6000, 11):
        state["stop"] = False
        oracle.call(cpu, mem, oracle.TICK_COOLDOWNS, state=state)
        state["stop"] = False
        oracle.call(cpu, mem, 0x16B5, state=state, maxins=stop)
        pc, sp = cpu.pc, (cpu.sp - 6) & 0xFF  # the six bytes the interrupt pushes
        frame = (cpu.y, cpu.x, cpu.a, cpu.p, pc & 0xFF, pc >> 8)
        for i, val in enumerate(frame):
            mem[0x0100 + ((sp + 1 + i) & 0xFF)] = val
        phase, stage, index, _p, _c = enemies.resume_from_stack(
            mem, sp, mem[0x100:0x200]
        )
        if 0x17B2 <= pc < 0x17CD:  # in the scan loop itself: Y is the live index
            assert (phase, stage, index) == (
                enemies.PHASE_BODY,
                enemies.BODY_SCAN,
                cpu.y,
            )
            direct += 1
        elif stage == enemies.BODY_SCAN:  # under $1887: the index is the saved $0C58
            assert phase == enemies.PHASE_BODY and index == mem[enemies.SAVED_Y]
            called += 1
        elif 0x31CA <= pc < 0x31F0:  # the prnd: the body is over, its result unstored
            assert (phase, stage) == (enemies.PHASE_CURSOR, enemies.BODY_ENTRY)
            prnd += 1
        state["stop"] = False
        oracle.call(cpu, mem, 0x16B5, state=state)  # let the round finish
        mem[mm.CURSOR] = (mem[mm.CURSOR] - 1) & 7
    assert direct and called and prnd


@pytest.mark.oracle
def test_the_resume_offset_is_the_cycles_the_machine_already_spent():
    """Each ladder entry is what the ROM itself spends from that segment's first byte.

    The resume credits the model those cycles, so a seed taken mid-segment starts where
    the machine is rather than at the segment's head."""
    from sentinel.tests import oracle  # pylint: disable=import-outside-toplevel

    cpu, mem, state = oracle.generate_machine(0x9795)
    oracle.prime_enemy_driver(cpu, mem, state)
    mem[mm.PLAYER_NOT_ACTED] = 0
    mem[0x0CDC] = mem[0x0CE4] = 0  # in the play loop, on the $1294 BPL path

    def spends(start, addr):
        state["stop"] = False
        c0 = cpu.processorCycles
        oracle.call(cpu, mem, start, state=state, stop_pc=addr)
        return cpu.processorCycles - c0

    slots = {}
    for slot in range(mm.NUM_SLOTS):
        slots.setdefault(mem[mm.OBJECTS_TYPE + slot], slot)
    checked = 0
    for otype, ladder in enemies._DISPATCH_SPENT.items():  # pylint: disable=W0212
        if otype not in slots:
            continue
        mem[mm.CURSOR] = slots[otype]
        head = dict(enemies._HEAD_SPENT)  # pylint: disable=protected-access
        if otype == mm.T_SENTRY:  # the $16C0 BEQ skips the second type compare
            head = {a: c for a, c in head.items() if a not in (0x16C2, 0x16C4)}
        for addr, want in sorted({**head, **ladder}.items()):
            assert spends(0x1289, addr) == want, (hex(otype), hex(addr))
            checked += 1
    for cursor, wrap in ((5, False), (0, True)):  # the $16DB BPL both ways
        for addr, want in sorted(
            enemies._CURSOR_SPENT[wrap].items()
        ):  # pylint: disable=W0212
            mem[mm.CURSOR] = cursor
            assert spends(0x16D9, addr) == want, (cursor, hex(addr))
            checked += 1
    mem[0x0C4E] = mem[0x0C63] = mem[0x0C64] = 0  # the tail's own three gates, open
    exposure = passcost.exposure_cycles(mem)
    for addr, want in sorted(enemies._TAIL_SPENT.items()):  # pylint: disable=W0212
        want += enemies._TAIL_SOUND.get(addr, 0)  # pylint: disable=protected-access
        want += exposure if addr > 0x12BB else 0
        assert spends(0x12A2, addr) == want, hex(addr)
        checked += 1
    for addr in (0x31CA, 0x31CD, 0x31CF, 0x31D9, 0x31E8):
        assert spends(0x31CA, addr) == enemies._prnd_spent(addr, 8), hex(addr)
        checked += 1
    assert checked > 25


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

    monkeypatch.setattr(passcost, "ROTATE_REDRAW", 6)
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
