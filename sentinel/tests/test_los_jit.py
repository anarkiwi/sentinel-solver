"""The numba fast-march (:func:`sentinel.los._march_jit`) is bit-for-bit identical
to the reference pure-Python march (:func:`sentinel.los._march_python`).

The numba path now resolves object tiles (the recursive object-stack walk
$1E3F) entirely in numba, so these tests deliberately place rich object stacks --
lone boulders, boulder+synthoid stacks, trees, the platform tile, deep
multi-boulder stacks -- in the ray's path and sweep the full aim lattice, at
ground and elevated eye height, asserting the two marches never diverge.

If numba is absent the JIT path is never taken, so these tests skip.
"""

import pytest

from sentinel import los, actions, memmap as mm
from sentinel.game import Game
from sentinel.los import (
    _march_python,
    _march_jit,
    prepare_vector_from_player_sights,
    PITCH_BAND,
    SIGHTS_CX,
    SIGHTS_CY,
)

pytestmark = pytest.mark.skipif(not los._HAVE_JIT, reason="numba not available")

SEEDS = [0, 1, 7, 42, 66, 999, 9999]


def _sweep_mismatches(state, slot, eye_z, max_steps):
    """Every (h, v) lattice aim where the two marches disagree on (tx, ty, los).

    Returns ``(bad, n)`` -- the list of disagreements and the comparison count."""
    do_los = state.mem[0x0C6E] & 0x7F
    bad = []
    n = 0
    for h in range(0, 256, 8):
        for v in PITCH_BAND:
            v1 = prepare_vector_from_player_sights(
                state, h, v, SIGHTS_CX, SIGHTS_CY, slot
            )
            r1 = _march_python(
                v1, state, slot, do_los_checks=do_los, eye_z=eye_z, max_steps=max_steps
            )
            v2 = prepare_vector_from_player_sights(
                state, h, v, SIGHTS_CX, SIGHTS_CY, slot
            )
            r2 = _march_jit(
                v2, state, slot, do_los_checks=do_los, eye_z=eye_z, max_steps=max_steps
            )
            n += 1
            if r1 != r2:
                bad.append((h, v, r1, r2))
    return bad, n


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_steps", [2000, 20000])
def test_jit_matches_python_bare(seed, max_steps):
    g = Game.new(seed)
    bad, _ = _sweep_mismatches(g.state, g.state.player, None, max_steps)
    assert not bad, f"seed {seed} ms {max_steps}: {bad[:3]}"


@pytest.mark.parametrize("seed", SEEDS)
def test_jit_matches_python_with_objects(seed):
    """Objects placed around the player force the object-tile path (the ray lands
    on object tiles resolved in numba)."""
    g = Game.new(seed)
    st = g.state
    px, py = st.player_xy()
    for dx, dy, typ in [
        (1, 0, mm.T_BOULDER),
        (2, 0, mm.T_BOULDER),
        (0, 1, mm.T_ROBOT),
        (1, 1, mm.T_TREE),
        (2, 2, mm.T_BOULDER),
        (-1, 0, mm.T_BOULDER),
    ]:
        x, y = px + dx, py + dy
        if 0 <= x < mm.N and 0 <= y < mm.N:
            try:
                actions.create(st, typ, (x, y))
            except Exception:
                pass
    bad, _ = _sweep_mismatches(st, st.player, None, 20000)
    assert not bad, f"seed {seed} with objects: {bad[:3]}"


@pytest.mark.parametrize("seed", SEEDS)
def test_jit_matches_python_centre(seed):
    """The return_centre path (final px_sub/py_sub) matches too."""
    g = Game.new(seed)
    st, slot = g.state, g.state.player
    for h in range(0, 256, 8):
        for v in PITCH_BAND:
            los._HAVE_JIT = False
            try:
                a = los.aim_target(
                    st,
                    h,
                    v,
                    SIGHTS_CX,
                    SIGHTS_CY,
                    slot,
                    max_steps=2000,
                    return_centre=True,
                )
            finally:
                los._HAVE_JIT = True
            b = los.aim_target(
                st, h, v, SIGHTS_CX, SIGHTS_CY, slot, max_steps=2000, return_centre=True
            )
            assert a == b, f"seed {seed} aim h={h} v={v}: py={a} jit={b}"


# ---------------------------------------------------------------------------
# rich object-stack scenarios
# ---------------------------------------------------------------------------
RING8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _ring(typ, r):
    return [(dx * r, dy * r, typ) for dx, dy in RING8]


def _place(st, px, py, specs):
    """create each (dx, dy, type) around (px, py); creates silently no-op if the
    tile is off-board or the stack can't take it."""
    for dx, dy, typ in specs:
        x, y = px + dx, py + dy
        if 0 <= x < mm.N and 0 <= y < mm.N:
            actions.create(st, typ, (x, y))


def _scen_boulders(st, px, py):
    """Lone boulders in concentric rings."""
    _place(
        st,
        px,
        py,
        _ring(mm.T_BOULDER, 1) + _ring(mm.T_BOULDER, 2) + _ring(mm.T_BOULDER, 3),
    )


def _scen_stacks(st, px, py):
    """Boulder + synthoid stacks, plus trees and a platform tile."""
    for dx, dy in RING8:
        x, y = px + dx, py + dy
        if 0 <= x < mm.N and 0 <= y < mm.N:
            actions.create(st, mm.T_BOULDER, (x, y))
            actions.create(st, mm.T_ROBOT, (x, y))  # stacks on the boulder
    _place(
        st,
        px,
        py,
        [
            (2, 0, mm.T_TREE),
            (0, 2, mm.T_TREE),
            (2, 2, mm.T_PLATFORM),
            (3, 0, mm.T_TREE),
        ],
    )


def _scen_deep(st, px, py):
    """Deep multi-boulder stacks with a synthoid on top."""
    for dx, dy in [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (-1, 0), (0, -1)]:
        x, y = px + dx, py + dy
        if 0 <= x < mm.N and 0 <= y < mm.N:
            actions.create(st, mm.T_BOULDER, (x, y))
            actions.create(st, mm.T_BOULDER, (x, y))
            actions.create(st, mm.T_BOULDER, (x, y))
            actions.create(st, mm.T_ROBOT, (x, y))


def _scen_elevated(st, px, py):
    """Objects all around, viewed from an elevated eye (as after a transfer up)."""
    _place(
        st,
        px,
        py,
        _ring(mm.T_BOULDER, 1) + _ring(mm.T_TREE, 2) + _ring(mm.T_BOULDER, 3),
    )
    return (st.obj_z_height[st.player] + 2) & 0xFF


SCENARIOS = [_scen_boulders, _scen_stacks, _scen_deep, _scen_elevated]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scen", SCENARIOS)
@pytest.mark.parametrize("max_steps", [2000, 20000])
def test_jit_matches_python_object_stacks(seed, scen, max_steps):
    """Rich object stacks in the ray path: the numba object-stack resolution stays
    bit-for-bit identical to the pure-Python oracle across the full aim lattice."""
    g = Game.new(seed)
    st = g.state
    st.energy = 9999  # guarantee every create lands, regardless of energy
    px, py = st.player_xy()
    eye_z = scen(st, px, py)
    bad, n = _sweep_mismatches(st, st.player, eye_z, max_steps)
    assert n == len(range(0, 256, 8)) * len(PITCH_BAND)
    assert not bad, f"seed {seed} {scen.__name__} ms {max_steps}: {bad[:5]}"
