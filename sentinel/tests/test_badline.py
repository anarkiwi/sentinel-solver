"""What one badline costs, derived from the BA window and checked against the machine.

The fixture is ``driver.badline``'s own report on three boards: every instruction's
unstolen cost recovered from cpuhistory, so each sample is an exact steal.
"""

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


def test_the_9630_anchor_is_not_instruction_aligned():
    """The raster IRQ is taken at an instruction boundary, so the frame marker's own
    position moves; a budget model therefore cannot place a badline inside an
    instruction even where it knows the routine."""
    for name, board in _live().items():
        positions = sorted(int(k) for k in board["anchor_9630"])
        assert len(positions) > 1, name
        assert positions[-1] - positions[0] <= 6, (name, positions)
