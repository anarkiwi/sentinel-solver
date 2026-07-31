"""Resuming an in-flight $1FFC JSR $2625 from plot_world's own progress zero page.

$0026 is the row the walk has reached and $0025 the column $295D is plotting, so a
halt inside the replot owes ``total - elapsed`` and not, as before, either of them.
"""

import pytest

from sentinel import projector

BOARD = 9795
STRIP_COLUMNS = 3  # the $0C69 the ROM gives a rotating ls9795 enemy


@pytest.fixture(name="mid_replot")
def _mid_replot(new_state):
    """A board set up the way $1FC2/$1FE5 leave it entering the strip replot."""
    st = new_state(BOARD)
    obs = st.player
    st.mem[projector.CAMERA_OBJECT] = obs
    left, right = projector.strip_window(STRIP_COLUMNS)
    st.mem[projector.BUF_LEFT] = left
    st.mem[projector.BUF_RIGHT] = right
    return st


def _pass_cycles(state):
    """The whole $2625 the fixture is set up for, priced by :func:`render_cost`."""
    obs = state.mem[projector.CAMERA_OBJECT]
    view = {
        "h_angle": state.obj_h_angle[obs],
        "v_angle": state.obj_v_angle[obs],
    }
    window = (state.mem[projector.BUF_LEFT], state.mem[projector.BUF_RIGHT])
    return projector.render_cost(state, view, obs, window=window) * (
        projector.FRAME_CYCLES
    )


def test_an_untouched_plot_world_owes_the_whole_pass(mid_replot):
    """$0026 still $1F: $26EF has not run, so nothing of the pass is spent."""
    owed = projector.replot_owed(mid_replot, 0xFF, 0, False)
    assert owed == pytest.approx(_pass_cycles(mid_replot), rel=1e-9)


def test_the_observer_row_owes_only_its_own_tile(mid_replot):
    """Walked past every plotted row, only $27CE's checkerboard is left."""
    owed = projector.replot_owed(mid_replot, -1, 0, False)
    assert 0 < owed < 0.02 * _pass_cycles(mid_replot)


def test_the_owed_cycles_fall_as_the_row_walks_in(mid_replot):
    """$26EF walks $0026 down, so each row it reaches owes strictly less."""
    owed = [projector.replot_owed(mid_replot, r, 0, False) for r in range(0x1E, -1, -1)]
    assert owed == sorted(owed, reverse=True)
    assert owed[0] > owed[-1]


def test_a_column_inside_a_row_owes_less_than_that_rows_start(mid_replot):
    """$0025 splits the row $295D is walking at the tile it has reached."""
    total = _pass_cycles(mid_replot)
    for row in range(0x1E, 0, -1):
        at_row = projector.replot_owed(mid_replot, row, 0, False)
        columns = sorted(
            {projector.replot_owed(mid_replot, row, c, True) for c in range(0x20)}
        )
        if len(columns) > 1:  # a row with more than one plotted tile
            assert columns[-1] < at_row
            assert columns[0] > 0
            assert columns[-1] < total
            return
    pytest.fail("no plotted row carried more than one tile")
