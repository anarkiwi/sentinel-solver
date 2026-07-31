"""Resuming an in-flight $1FFC JSR $2625 from plot_world's own progress zero page.

$0026 is the row the walk has reached and $0025 the column $295D is plotting, so a halt
inside the replot owes ``total - elapsed``, plus the part of $1F9F's own line that has
not run: the $2211 clear is spent, the chunks left, the $9730 flush and the exit are not.
"""

import pytest

from sentinel import passcost, projector

BOARD = 9795
STRIP_COLUMNS = 3  # the $0C69 the ROM gives a rotating ls9795 enemy
STRIP_LEFT = 0x11  # its $0C62, an odd column so $001F carries the half


def _entering_replot(state, columns, span):
    """A board set up the way $209B/$1FC2/$1FE5 leave it entering a strip's $2625."""
    obs = state.player
    state.mem[projector.CAMERA_OBJECT] = obs
    left, right, frac = projector.strip_window(columns)
    state.mem[projector.BUF_LEFT] = left
    state.mem[projector.BUF_RIGHT] = right
    state.mem[projector.BUF_FRAC] = frac
    state.mem[projector.STRIP_LEFT] = STRIP_LEFT
    state.mem[projector.STRIP_COLUMNS] = columns
    state.mem[projector.STRIP_REMAINING] = span
    state.mem[projector.STRIP_SPAN] = span
    state.mem[projector.CAMERA_SAVED] = state.obj_h_angle[obs]
    return state


@pytest.fixture(name="mid_replot")
def _mid_replot(new_state):
    """One chunk wide: $0C6A == $0C69, so no chunk follows this $2625."""
    return _entering_replot(new_state(BOARD), STRIP_COLUMNS, STRIP_COLUMNS)


def _pass_cycles(state):
    """The whole $2625 the fixture is set up for, priced by :func:`render_cost`."""
    obs = state.mem[projector.CAMERA_OBJECT]
    view = {
        "h_angle": state.obj_h_angle[obs],
        "v_angle": state.obj_v_angle[obs],
        "ref_lo": state.mem[projector.CAMERA_REF_LO],
    }
    window = (
        state.mem[projector.BUF_LEFT],
        state.mem[projector.BUF_RIGHT],
        state.mem[projector.BUF_FRAC],
    )
    return projector.render_cost(state, view, obs, window=window) * (
        projector.FRAME_CYCLES
    )


def _line_owed(state):
    """$1FFF..$1F9E: the camera restore, the $9730 flush and the exit, always owed."""
    return passcost.REDRAW_CHUNK_TAIL + projector.strip_flush_cycles(
        state.mem[projector.STRIP_SPAN], state.mem[projector.SCREEN_SCROLL]
    )


def test_an_untouched_plot_world_owes_the_whole_pass(mid_replot):
    """$0026 still $1F: $26EF has not run, so nothing of the pass is spent."""
    owed = projector.replot_owed(mid_replot, 0xFF, 0, False)
    want = _pass_cycles(mid_replot) + _line_owed(mid_replot)
    assert owed == pytest.approx(want, rel=1e-9)


def test_the_strip_clear_is_spent_not_owed(mid_replot):
    """$1FBA JSR $2211 runs before the chunk loop, so a halt in $2625 owes none of it."""
    span = mid_replot.mem[projector.STRIP_SPAN]
    whole = projector.strip_line_cycles(
        span, 1, mid_replot.mem[projector.SCREEN_SCROLL]
    )
    spent = whole - _line_owed(mid_replot)
    assert spent == passcost.REDRAW_CLEAR_CALL + projector.clear_strip_cycles(span) + (
        passcost.REDRAW_CHUNK_HEAD + passcost.BUF_WINDOW_CALL
    )


def test_a_halt_in_the_first_chunk_still_owes_the_second(new_state, mid_replot):
    """$0C6A over $0C69: $201E re-enters $1FC2, so the chunk left is owed as well."""
    one = projector.replot_owed(mid_replot, 0xFF, 0, False)
    wide = _entering_replot(new_state(BOARD), 0x14, 0x14 + STRIP_COLUMNS)
    two = projector.replot_owed(wide, 0xFF, 0, False)
    extra = passcost.REDRAW_CHUNK_MORE + passcost.REDRAW_CHUNK_RESUME
    assert two - one > extra + projector.CHUNK_CYCLES  # and the second chunk's $2625


def test_the_observer_row_owes_only_its_own_tile(mid_replot):
    """Walked past every plotted row, only $27CE's checkerboard and the line are left."""
    owed = projector.replot_owed(mid_replot, -1, 0, False) - _line_owed(mid_replot)
    assert 0 < owed < 0.02 * _pass_cycles(mid_replot)


def test_the_owed_cycles_fall_as_the_row_walks_in(mid_replot):
    """$26EF walks $0026 down, so each row it reaches owes strictly less."""
    owed = [projector.replot_owed(mid_replot, r, 0, False) for r in range(0x1E, -1, -1)]
    assert owed == sorted(owed, reverse=True)
    assert owed[0] > owed[-1]


def test_a_column_inside_a_row_owes_less_than_that_rows_start(mid_replot):
    """$0025 splits the row $295D is walking at the tile it has reached."""
    total = _pass_cycles(mid_replot) + _line_owed(mid_replot)
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


def _rotating(new_state):
    """ls9795 armed, wound to just before its one on-screen $1F9F."""
    from sentinel import enemies, memmap as mm  # pylint: disable=C0415

    state = new_state(BOARD)
    state.mem[mm.PLAYER_NOT_ACTED] = 0x00
    enemies.advance_frames(state, 128)
    return state


def _to_the_replot(state, frames=40):
    """Step until the model charges a strip replot; the frame it did, or None."""
    from sentinel import enemies  # pylint: disable=import-outside-toplevel

    for f in range(frames):
        enemies.advance_frame(state)
        if state.camera_shift:
            return f
    return None


def test_the_camera_shift_is_live_for_the_frames_the_replot_stalls(new_state):
    """$1FC2 leaves $09C0,X shifted until $2003/$2008, i.e. for the whole stall.

    The model charges the replot in one lump, so the shift has to persist over every
    frame that lump spends and come off in the frame the pass resumes in."""
    from sentinel import enemies  # pylint: disable=import-outside-toplevel

    state = _rotating(new_state)
    own = int(state.obj_h_angle[state.player])
    seen = []
    for _ in range(40):
        enemies.advance_frame(state)
        seen.append(int(state.obj_h_angle[state.player]))
    shifted = [a for a in seen if a != own]
    assert shifted, "ls9795 reached no on-screen $1F9F in this span"
    assert len(set(shifted)) == 1  # one strip, one $0C62, one shift
    assert seen[-1] == own and state.camera_shift == 0  # $2003/$2008 put it back
    assert shifted == seen[seen.index(shifted[0]) :][: len(shifted)]  # contiguous
    assert len(shifted) > 10  # a replot is many frames of stall, not a redraw


def test_the_shift_is_0c62_halved_over_the_211a_the_rom_saved(new_state):
    """$1FD5 saves the bearing, $1FD8 adds $0C62/2 and $1FD0 keeps the odd half."""
    from sentinel import enemies  # pylint: disable=import-outside-toplevel

    state = _rotating(new_state)
    for _ in range(40):
        enemies.advance_frame(state)
        if projector.held_strip(state):
            break
    left = state.camera_shift
    assert left and projector.held_strip(state) == left  # the column it names back
    assert (
        int(state.obj_h_angle[state.player])
        == (state.mem[projector.CAMERA_SAVED] + (left >> 1)) & 0xFF
    )
    assert state.mem[projector.CAMERA_REF_LO] == (0x80 if left & 1 else 0x00)


def test_the_shift_waits_for_the_2211_clear_the_replot_runs_first(new_state):
    """$1FBA JSR $2211 runs before $1FC2, so the clear's own cycles carry no shift.

    ls9795's strip starts at column 37, so the clear is 8021 cycles the model must
    spend before $09C0,X moves -- more than the frame the replot is charged in had
    left, which is why the camera is still the player's own at that frame's end."""
    from sentinel import enemies  # pylint: disable=import-outside-toplevel

    state = _rotating(new_state)
    assert _to_the_replot(state) is not None  # the frame $1F9F was charged in
    assert state.camera_clear < 0  # the clear, less the whole replot it belongs to
    assert state.cycle_residual < state.camera_clear  # ... still inside that clear
    assert projector.held_strip(state) == 0  # so the camera is the player's own
    for _ in range(40):  # and the rule holds over the whole stall
        enemies.advance_frame(state)
        if not state.camera_shift:
            break
        held = state.cycle_residual >= state.camera_clear
        assert (projector.held_strip(state) != 0) == held
    assert projector.held_strip(state) == 0  # $2003/$2008 on the resuming frame
