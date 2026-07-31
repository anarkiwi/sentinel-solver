"""What one badline costs, derived from the BA window and checked against the machine.

The fixture is ``driver.badline``'s own report on three boards: every instruction's
unstolen cost recovered from cpuhistory, so each sample is an exact steal.
"""

import collections
import json
import os

from sentinel import badline, passcost

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "live_badline.json")


def _live():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["boards"]


def _complete_frames(board):
    """Per-frame totals of the frames that carry all 25 badlines."""
    floor = passcost.BADLINES_PER_FRAME * badline.MIN_STEAL
    return {int(k): v for k, v in board["per_frame_total"].items() if int(k) >= floor}


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


def test_the_frame_steal_is_state_dependent_so_no_constant_is_right():
    """The frame's own total moves board to board AND frame to frame, and BADLINE_FRAME
    -- the one fitted term left in the budget -- sits inside the range on every board.
    """
    floor = passcost.BADLINES_PER_FRAME * badline.MIN_STEAL
    ceiling = passcost.BADLINES_PER_FRAME * passcost.BADLINE_STEAL
    means = {}
    for name, board in _live().items():
        totals = _complete_frames(board)
        assert len(totals) > 1, f"{name}: the frame total is a single value"
        assert floor <= min(totals) and max(totals) <= ceiling, name
        means[name] = sum(k * v for k, v in totals.items()) / sum(totals.values())
    assert min(means.values()) < passcost.BADLINE_FRAME < max(means.values()), means


_BRANCHES = frozenset(("BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ"))


def _entries(field, offset=0):
    """``(entry cycles, mnemonic) -> count`` over every capture's ``field``."""
    out = collections.Counter()
    for board in _live().values():
        for key, count in board[field].items():
            gap, mnemonic = key.split()
            out[(int(gap) - offset, mnemonic)] += count
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
        placed = {badline.marker_position(b, br) for b in boundaries for br in (0, 1)}
        assert set(positions) <= placed, (name, positions, sorted(placed))


def test_the_interrupt_sequence_is_a_cycle_shorter_only_off_a_branch():
    """The $9589 chain fires on rasters 53/93/133/173/213 and every entry is the
    6510's own interrupt sequence past the boundary it caught -- 7 cycles, or 6 when
    the instruction was a branch, whose IRQ poll comes a cycle earlier."""
    lines = set()
    for board in _live().values():
        lines |= {int(k) for k in board["irq_entry_line"]}
    assert lines == {53, 93, 133, 173, badline.RASTER_IRQ_LINE}
    seen = _entries("irq_entry_gap") + _entries("marker_gap", badline.MARKER_OFFSET)
    assert {cycles for cycles, _ in seen} == {
        badline.IRQ_ENTRY,
        badline.IRQ_ENTRY_BRANCH,
    }
    short = {m for cycles, m in seen if cycles == badline.IRQ_ENTRY_BRANCH}
    assert short and short <= _BRANCHES, short


def test_every_badline_window_lands_in_a_run_the_cost_model_anchors():
    """A BA window can only be placed if the model knows which run is executing.  Over
    6424 live windows -- 1600 of them inside one $1887 march -- every one falls after
    a $XXXX the cost model itself counts from."""
    windows = sum(b["windows"] for b in _live().values())
    covered = sum(b["anchor_covered"] for b in _live().values())
    assert windows > 6000 and covered == windows
    march = _live()["9795_march"]
    assert march["window_routine"]["$1CBB march"] > march["windows"] // 2


def test_the_offset_into_an_anchored_run_determines_the_steal():
    """1081 distinct (anchor, offset) keys over five captures; 4 carry two steals, and
    each of those is a branch the cost model already charges ($0D03's per-bit shift-add,
    $1CBB's per-component negate, $191F's targeting walk)."""
    keys = sum(b["anchor_keys"] for b in _live().values())
    ambiguous = {k: v for b in _live().values() for k, v in b["ambiguous"].items()}
    assert keys > 1000
    assert len(ambiguous) <= 4, ambiguous
    assert set(ambiguous) <= {"$0D05+57", "$1CCC+33", "$193A+10"}, ambiguous


def test_the_frame_steal_is_the_law_applied_to_the_frames_instruction_stream():
    """``frame_steal`` over the live stream reproduces every complete frame's own
    measured total, march frames included."""
    for name, board in _live().items():
        assert board["complete_frames"] > 0, name
        assert board["frame_steal_agrees"] == board["complete_frames"], name
