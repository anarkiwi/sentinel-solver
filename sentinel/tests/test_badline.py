"""What one badline costs, derived from the BA window and checked against the machine.

The fixture is ``driver.badline``'s own report on three boards: every instruction's
unstolen cost recovered from cpuhistory, so each sample is an exact steal.
"""

import collections
import json
import os

from sentinel import badline, enemies, memmap as mm, passcost, writeruns, writeweight
from sentinel.game import Game

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "live_badline.json")


def _live():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["boards"]


def _complete_frames(board):
    """Per-frame totals of the frames that carry all 25 badlines."""
    floor = passcost.BADLINES_PER_FRAME * badline.MIN_STEAL
    return {int(k): v for k, v in board["per_frame_total"].items() if int(k) >= floor}


def _machine_steal(name):
    """The mean steal a frame of ``name``'s captures pays."""
    totals = _complete_frames(_live()[name])
    return sum(k * v for k, v in totals.items()) / sum(totals.values())


def _model_steal(typed, frames):
    """The mean steal the model charges itself over ``frames`` of board ``typed``.

    The frame is charged the ceiling up front and each window refunds what it lands on,
    so the steal is that ceiling less the refund the frame's own clock accrued.
    """
    clocks, real = [], badline.frame_clock

    def spy(armed=True):
        clock = real(armed)
        if armed:
            clocks.append(clock)
        return clock

    game = Game.typed(typed)
    game.state.mem[mm.PLAYER_NOT_ACTED] = 0x00  # $12E1: the player has acted
    badline.frame_clock = spy
    try:
        for _ in range(frames):
            enemies.advance_frame_python(game.state)
    finally:
        badline.frame_clock = real
    return badline.FRAME_STEAL_CEILING - sum(int(c[3]) for c in clocks) / frames


def test_the_badline_window_is_the_25_lines_of_a_pal_frame():
    """Raster 51..243 step 8, one window a line, 504 cycles apart."""
    windows = badline.window_positions()
    assert len(windows) == passcost.BADLINES_PER_FRAME
    assert windows[0] // badline.LINE_CYCLES == badline.BADLINE_FIRST_LINE
    assert windows[-1] // badline.LINE_CYCLES == 243
    step = badline.BADLINE_LINE_STEP * badline.LINE_CYCLES
    assert all(b - a == step for a, b in zip(windows, windows[1:]))
    assert all(w % badline.LINE_CYCLES == badline.BADLINE_WINDOW_CYCLE for w in windows)


def test_a_write_run_is_never_three_cycles_so_the_steal_never_reaches_40():
    """Every instruction opens with an opcode fetch, so the longest write run is the
    two of an NMOS read-modify-write or of a JSR/push pair -- which puts a hard floor of
    BADLINE_STEAL - 2 under one badline, and 40 out of reach."""
    for op, cycles in badline.WRITE_CYCLES.items():
        assert len(cycles) <= 2, hex(op)
        assert cycles == tuple(range(cycles[0], cycles[0] + len(cycles))), hex(op)
    assert badline.MIN_STEAL == passcost.BADLINE_STEAL - 2


def test_every_live_badline_steal_is_the_derived_one():
    """4824 live badlines over four captures, each exactly 43 less its window writes."""
    windows = badline.window_positions()
    counted = 0
    for name, board in _live().items():
        for position, op, steal, count in board["samples"]:
            assert badline.steal(op, position, windows) == steal, (name, position, op)
            counted += count
        assert board["window_line_cycle"] == badline.BADLINE_WINDOW_CYCLE
    assert counted == sum(b["badlines"] for b in _live().values())
    assert counted > 4000


def test_the_live_steal_never_leaves_the_derived_bounds():
    """No sample is under BADLINE_STEAL - 2 or over BADLINE_STEAL; 44 is an artefact of
    charging an opcode's own taken branch or crossed page to the VIC."""
    for name, board in _live().items():
        seen = {int(k) for k in board["steal"]}
        assert seen <= set(range(badline.MIN_STEAL, passcost.BADLINE_STEAL + 1)), name
        assert max(seen) == passcost.BADLINE_STEAL, name


_MODEL_FRAMES = 600  # the model's own opening, the stretch the captures span
_STEAL_TOLERANCE = 1.0  # the machine's mean over 20 captures carries about 0.5


def test_the_models_own_frame_steal_is_the_one_the_machine_pays_on_each_board():
    """Every board's model steal against the machine's own, over a comparable stretch
    of the same board -- named by the typed number the capture entered, so the fixture's
    "0335" is ``Game.typed(335)`` and not ``0x335``.  The machine's mean must come from
    captures spread over the run: 15 consecutive frames carry an error of about 0.6."""
    for name, typed in (("0042", 42), ("0335", 335), ("9795", 9795)):
        machine, model = _machine_steal(name), _model_steal(typed, _MODEL_FRAMES)
        assert abs(model - machine) < _STEAL_TOLERANCE, (name, model, machine)


def test_the_frame_steal_is_state_dependent_so_no_constant_is_right():
    """The frame's own total moves board to board AND frame to frame, so the budget
    charges it off the model's own term stream rather than carrying a constant."""
    floor = passcost.BADLINES_PER_FRAME * badline.MIN_STEAL
    ceiling = passcost.BADLINES_PER_FRAME * passcost.BADLINE_STEAL
    means = {}
    for name, board in _live().items():
        totals = _complete_frames(board)
        assert len(totals) > 1, f"{name}: the frame total is a single value"
        assert floor <= min(totals) and max(totals) <= ceiling, name
        means[name] = sum(k * v for k, v in totals.items()) / sum(totals.values())
    assert max(means.values()) - min(means.values()) > 1, means
    assert not hasattr(passcost, "BADLINE_FRAME")


_BRANCHES = frozenset(("BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ"))


def _entries(field, offset=0):
    """``(entry cycles, mnemonic, its own cost) -> count`` over every ``field``."""
    out = collections.Counter()
    for board in _live().values():
        for key, count in board[field].items():
            gap, mnemonic, cost = key.split()
            out[(int(gap) - offset, mnemonic, int(cost))] += count
    return out


def test_the_9630_anchor_is_instruction_aligned_and_its_spread_is_that_instruction():
    """The marker's 4-6 cycle spread is the tail of the instruction the raster IRQ
    interrupted, not blur: its frame position is that instruction's own end plus the
    entry and MARKER_OFFSET, on every frame of every capture."""
    for name, board in _live().items():
        positions = sorted(int(k) for k in board["anchor_9630"])
        assert len(positions) > 1 and positions[-1] - positions[0] <= 6, name
        boundaries = sorted(int(k) for k in board["irq_boundary"])
        assert boundaries[0] // badline.LINE_CYCLES == badline.RASTER_IRQ_LINE, name
        placed = {
            badline.marker_position(b, entry)
            for b in boundaries
            for branch in (0, 3, 4)
            for entry in badline.entry_cycles(branch)
        }
        assert set(positions) <= placed, (name, positions, sorted(placed))


def test_the_interrupt_sequence_is_short_only_by_a_taken_branchs_own_cycles():
    """The $9589 chain fires on rasters 53/93/133/173/213 and every live entry is the
    6510's sequence past the boundary it caught: 7, or -- caught by a TAKEN branch's
    two-cycle poll -- 6 off its 3 and 5 off the 4 of a page crossing, never else."""
    lines = set()
    for board in _live().values():
        lines |= {int(k) for k in board["irq_entry_line"]}
    assert lines == {53, 93, 133, 173, badline.RASTER_IRQ_LINE}
    seen = _entries("irq_entry_gap") + _entries("marker_gap", badline.MARKER_OFFSET)
    for cycles, mnemonic, cost in seen:
        taken = cost if mnemonic in _BRANCHES and cost > 2 else 0
        assert cycles in badline.entry_cycles(taken), (cycles, mnemonic, cost)
    assert {5, 6, badline.IRQ_ENTRY} == {cycles for cycles, _, _ in seen}


def test_every_badline_window_lands_in_a_run_the_cost_model_anchors():
    """A BA window can only be placed if the model knows which run is executing.  Over
    6424 live windows -- 1600 of them inside one $1887 march -- every one falls after
    a $XXXX the cost model itself counts from."""
    windows = sum(b["windows"] for b in _live().values())
    covered = sum(b["anchor_covered"] for b in _live().values())
    assert windows > 6000 and covered == windows
    march = _live()["9795_march"]
    assert march["window_routine"]["$1CBB march"] > march["windows"] // 2


_BRANCHED_ANCHORS = frozenset((0x0D05, 0x16BB, 0x193A, 0x1CCC))


def test_the_offset_into_an_anchored_run_determines_the_steal():
    """2718 distinct (anchor, offset) keys over five captures, 27 of them carrying two
    steals -- and every one of those is a run whose branch the charging term itself
    decides: $0D03's per-bit shift-add, $16B5's dispatch (the $16D6 JSR $31CA it either
    takes or not), $191F's targeting walk, $1CBB's per-component negate."""
    keys = sum(b["anchor_keys"] for b in _live().values())
    two = sum(b["ambiguous_keys"] for b in _live().values())
    ambiguous = {k: v for b in _live().values() for k, v in b["ambiguous"].items()}
    assert keys > 2000 and two * 50 < keys
    assert all(len(v) == 2 for v in ambiguous.values()), ambiguous
    anchors = {int(k.split("+")[0][1:], 16) for k in ambiguous}
    assert anchors <= _BRANCHED_ANCHORS, ambiguous


def test_the_frame_steal_is_the_law_applied_to_the_frames_instruction_stream():
    """``frame_steal`` over the live stream reproduces every complete frame's own
    measured total, march frames included."""
    for name, board in _live().items():
        assert board["complete_frames"] > 0, name
        assert board["frame_steal_agrees"] == board["complete_frames"], name


def test_the_frame_clock_places_every_window_and_split_irq_at_its_own_raster():
    """The clock is pinned at the raster IRQ, so the frame opens with the four badlines
    the $9630 body contains and then 7056 cycles of foreground with none at all."""
    events = badline.frame_events()
    assert len(events) == passcost.BADLINES_PER_FRAME + len(badline.SHORT_IRQ_LINES)
    assert sum(1 for _, short in events if short) == len(badline.SHORT_IRQ_LINES)
    assert list(events) == sorted(events)
    origin = badline.FRAME_ORIGIN
    first = [p for p, short in events if not short][:4]
    assert first == [
        (219 + 8 * i) * badline.LINE_CYCLES + 11 - origin for i in range(4)
    ]
    gap = [p for p, short in events if not short][4] - first[-1]
    assert gap == (51 + 312 - 243) * badline.LINE_CYCLES


def test_charge_refunds_the_write_run_the_window_lands_on():
    """The frame pays BADLINE_STEAL for all 25 up front, so a window over $31CA's own
    ROL abs pair comes back two cycles and one over a read comes back none."""
    clk = badline.frame_clock()
    window = int(badline.EVENT_POS[0])
    assert badline.run_at(0x31CA, window) == 2  # the LFSR's dummy-plus-real write
    assert badline.charge(clk, 0x31CA, window + 1) == window - 1
    assert clk[3] == 2
    clk = badline.frame_clock()
    assert badline.charge(clk, 0x0000, window + 1) == window + 1  # an unmapped run
    assert clk[3] == 0


def test_an_unarmed_clock_charges_nothing_so_the_isolated_round_is_unchanged():
    """``step`` and ``update_enemies`` price a routine, not a frame, so their clock
    places no window and every term costs exactly what passcost says."""
    clk = badline.frame_clock(False)
    assert badline.charge(clk, 0x31CA, passcost.PRND) == passcost.PRND
    assert clk[3] == 0


MARCH_CYCLES = (
    274578  # the one $1887 the ls9795 gate turns on: ~18 frames of foreground
)


def _lap(neg=0):
    """``(cycles, write weight)`` of one $1CE8..$1D18 flat march sub-step."""
    return (
        sum(getattr(passcost, n) for n in writeweight.FLAT_LAP)
        + neg * passcost.ADD_VECTOR_NEG,
        sum(writeweight.WEIGHT.get(n, 0) for n in writeweight.FLAT_LAP)
        + neg * writeweight.WEIGHT["ADD_VECTOR_NEG"],
    )


def test_a_marching_term_refunds_the_write_weight_of_its_own_laps():
    """A term the static map cannot reach is priced by its own write profile instead:
    each of the 25 windows a frame gets back the ``weight / cycles`` the loop's
    instructions drive, in every one of the frames the term spans."""
    lap, weight = _lap()
    laps = MARCH_CYCLES // lap
    clk = badline.frame_clock()
    spent = badline.charge_run(clk, laps * lap, laps * weight)
    frames = laps * lap / passcost.PAL_FRAME_CYCLES
    assert frames > 13  # the gate's own march outlives fourteen frames
    assert spent == laps * lap - clk[3]
    per_window = weight / lap
    per_frame = clk[3] / frames
    assert abs(per_frame - passcost.BADLINES_PER_FRAME * per_window) < 0.2, per_frame
    assert clk[2] < badline.N_EVENTS  # the event list wrapped rather than running out
    assert clk[1] == badline.EVENT_POS[clk[2]]


def test_the_marched_refund_is_the_per_frame_steal_the_machine_pays():
    """Inside a march the machine's frame keeps 3.48 of the 1075-cycle ceiling back.
    The march's own laps carry between 30/306 and 39/318 of a cycle per window --
    2.4 to 3.1 a frame -- so the weight covers most of that and none of it is fitted.
    """
    totals = _complete_frames(_live()["9795_march"])
    machine = passcost.BADLINES_PER_FRAME * passcost.BADLINE_STEAL - sum(
        k * v for k, v in totals.items()
    ) / sum(totals.values())
    lo, hi = (passcost.BADLINES_PER_FRAME * w / c for c, w in (_lap(0), _lap(3)))
    assert lo < hi < machine < hi * 1.5, (lo, hi, machine)


def test_a_term_outliving_the_frame_pays_the_ceiling_with_nothing_to_refund_it():
    """An anchor's map covers only the run walked from it, so the gate's 274578-cycle
    $1887 pays the full steal at all 25 windows against a 49-cycle map, and the walk from
    $95E9 RTIs out of the split chain before the first window the $9630 body contains.
    Which is why both are charged by their own write weight instead.
    """
    assert writeruns.LENGTH_AT[0x1887] < int(badline.EVENT_POS[0])
    clk = badline.frame_clock()
    assert badline.charge(clk, 0x1887, MARCH_CYCLES) == MARCH_CYCLES
    assert clk[2] == badline.N_EVENTS and clk[3] == 0  # every event, no refund at all
    clk = badline.frame_clock()
    badline.charge(clk, 0x95E9, passcost.IRQ_BODY)
    assert clk[2] == 4 and clk[3] == 0  # the body's own four windows, unrefunded
    clk = badline.frame_clock()
    badline.charge_run(clk, passcost.IRQ_BODY, writeweight.WEIGHT["IRQ_BODY"])
    assert clk[2] == 4 and clk[4] > 0  # ... which the body's own weight now gives back
    totals = _complete_frames(_live()["9795_march"])
    mean = sum(k * v for k, v in totals.items()) / sum(totals.values())
    ceiling = passcost.BADLINES_PER_FRAME * passcost.BADLINE_STEAL
    assert 3.0 < ceiling - mean < 4.0, mean
