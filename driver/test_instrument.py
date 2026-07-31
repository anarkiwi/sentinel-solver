"""The instrument's $95E9 stack decode and the replot debt it turns into."""

import sys

sys.path.insert(0, "/".join(__file__.split("/")[:-2]))

from sentinel import projector  # noqa: E402
from driver import instrument  # noqa: E402

STRIP_COLUMNS = 3


def _frame(sp, pc, chain):
    """A $9630 stack page: the $95E9 frame at ``sp`` over ``chain``'s JSR returns."""
    page = bytearray(0x100)
    page[(sp + 5) & 0xFF] = pc & 0xFF
    page[(sp + 6) & 0xFF] = pc >> 8
    for i, ret in enumerate(chain):
        page[(sp + 7 + 2 * i) & 0xFF] = (ret - 1) & 0xFF
        page[(sp + 8 + 2 * i) & 0xFF] = (ret - 1) >> 8
    return page


def test_stack_frames_reads_the_pc_and_the_return_chain():
    page = _frame(0xF0, 0x26EC, [0x1FFF, 0x1884])
    assert instrument.stack_frames(page, 0xF0)[:3] == [0x26EC, 0x1FFF, 0x1884]


def test_no_replot_on_the_stack_owes_nothing(new_state):
    frames = instrument.stack_frames(_frame(0xF0, 0x1AC5, [0x17E8, 0x12A2]), 0xF0)
    assert instrument.replot_debt(new_state(9795), frames) == 0


def _strip_state(new_state):
    """A board with the $1FC2/$1FE5 camera and window a strip replot plots into."""
    st = new_state(9795)
    st.mem[projector.CAMERA_OBJECT] = st.player
    left, right, frac = projector.strip_window(STRIP_COLUMNS)
    st.mem[projector.BUF_LEFT], st.mem[projector.BUF_RIGHT] = left, right
    st.mem[projector.BUF_FRAC] = frac
    return st


def test_the_row_loop_head_has_already_plotted_the_row_0026_names(new_state):
    """$26EC is before $26EF DEC $26, so $0026's own $295D is spent and $0026-1 is next."""
    st = _strip_state(new_state)
    st.mem[projector.PLOT_ROW] = 0x10
    head = instrument.stack_frames(_frame(0xF0, 0x26EC, [0x1FFF, 0x1884]), 0xF0)
    scan = instrument.stack_frames(_frame(0xF0, 0x37F2, [0x2701, 0x1FFF]), 0xF0)
    assert instrument.replot_debt(st, head) < instrument.replot_debt(st, scan)


def test_a_column_inside_295d_owes_less_than_the_scan_that_precedes_it(new_state):
    """The $2760 return says $0025 is live, so the row is part spent."""
    st = _strip_state(new_state)
    st.mem[projector.PLOT_ROW] = 0x10
    st.mem[projector.PLOT_COLUMN] = 0xFF  # not a plotted column: the whole row is spent
    scan = instrument.stack_frames(_frame(0xF0, 0x37F2, [0x2701, 0x1FFF]), 0xF0)
    plot = instrument.stack_frames(_frame(0xF0, 0x2377, [0x2760, 0x1FFF]), 0xF0)
    assert 0 < instrument.replot_debt(st, plot) < instrument.replot_debt(st, scan)
