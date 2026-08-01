"""The numba enemy clock is byte-identical to the pure-Python reference.

Both halves advance the SAME board and the FULL 64 KB image is compared after every
chunk, so a divergent cooldown, facing, PRNG byte, tree or hyperspace fails at once.
"""

import pytest

from sentinel import enemies, memmap as mm
from sentinel.game import Game

pytestmark = pytest.mark.skipif(not enemies._HAVE_JIT, reason="numba not available")

LANDSCAPES = [0, 42, 335]
CHUNK = 25
CHUNKS = 16  # 400 frames per board


def _armed(landscape):
    """A board with the enemy clock running (the ROM freezes it until the first act)."""
    game = Game.typed(landscape)
    game.state.mem[mm.PLAYER_NOT_ACTED] = 0x00
    return game.state


def _first_diff(a, b):
    """(address, python byte, jit byte) of the first divergence, else None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    return None


def _resume(state):
    """The whole sub-pass resume point, which the image itself does not carry."""
    return (
        state.cycle_residual,
        state.pass_phase,
        state.body_stage,
        state.body_index,
        state.body_partial,
        state.camera_shift,
        state.camera_clear,
        state.steal_residue,
        state.clock_overhang,
        state.entry_b,
        state.carry_step,
    )


@pytest.mark.parametrize("landscape", LANDSCAPES)
def test_jit_matches_python_full_image(landscape):
    ref = _armed(landscape)
    jit = ref.clone()
    for chunk in range(CHUNKS):
        enemies.advance_frames_python(ref, CHUNK)
        enemies.advance_frames(jit, CHUNK)
        diff = _first_diff(ref.mem, jit.mem)
        assert diff is None, f"ls{landscape} frame {(chunk + 1) * CHUNK}: {diff}"
        assert _resume(ref) == _resume(jit), f"ls{landscape} chunk {chunk}"


@pytest.mark.parametrize("landscape", LANDSCAPES)
def test_jit_matches_python_while_plotting(landscape):
    """A plotting span advances only the cooldown clock -- also byte-identical."""
    ref = _armed(landscape)
    jit = ref.clone()
    for _ in range(CHUNKS):
        enemies.advance_frames_python(ref, CHUNK, plotting=True)
        enemies.advance_frames(jit, CHUNK, plotting=True)
    assert _first_diff(ref.mem, jit.mem) is None


@pytest.mark.parametrize("landscape", LANDSCAPES)
def test_single_frame_dispatch_matches(landscape):
    """:func:`enemies.advance_frame`, the per-frame entry the driver races against the
    ROM, tracks the reference frame for frame."""
    ref = _armed(landscape)
    jit = ref.clone()
    for frame in range(120):
        enemies.advance_frame_python(ref)
        enemies.advance_frame(jit)
        assert _first_diff(ref.mem, jit.mem) is None, f"ls{landscape} frame {frame}"
        assert _resume(ref) == _resume(jit), f"ls{landscape} frame {frame}"


@pytest.mark.parametrize("landscape", LANDSCAPES)
def test_jit_matches_python_across_an_on_screen_redraw(landscape):
    """$1F9F with a screen span costs a $1FFC replot the jit twin cannot price itself:
    it stops and hands the object back, so the two must still land on the same clock."""
    from sentinel import relative  # pylint: disable=import-outside-toplevel

    state = _armed(landscape)
    player = state.mem[mm.PLAYER_OBJECT]
    for h_angle in range(256):
        state.obj_h_angle[player] = h_angle
        if any(
            not state.obj_flags[s] & 0x80
            and state.obj_type[s] in mm.ENEMY_TYPES
            and relative.object_screen_span(state, s)[0]
            for s in range(8)
        ):
            break
    else:
        pytest.skip("no facing puts an enemy on screen")
    ref, jit = state.clone(), state.clone()
    enemies.advance_frames_python(ref, 3000)
    enemies.advance_frames(jit, 3000)
    assert _first_diff(ref.mem, jit.mem) is None
    assert _resume(ref) == _resume(jit)
