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


@pytest.mark.parametrize("landscape", LANDSCAPES)
def test_jit_matches_python_full_image(landscape):
    ref = _armed(landscape)
    jit = ref.clone()
    for chunk in range(CHUNKS):
        enemies.advance_frames_python(ref, CHUNK)
        enemies.advance_frames(jit, CHUNK)
        diff = _first_diff(ref.mem, jit.mem)
        assert diff is None, f"ls{landscape} frame {(chunk + 1) * CHUNK}: {diff}"


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
