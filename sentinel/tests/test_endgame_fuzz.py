"""Endgame fuzz: states a KNOWN number of moves from a win, closed by the planner.

A win is a hyperspace FROM the platform tile ($0CDE bit6, ``actions.won``), so the
last three moves are fixed: ``robot`` on the platform, ``transfer`` into it,
``hyperspace``.  Each case is built at a distance d in {1,2,3} along that tail.
"""

import math
import random

import pytest

from sentinel import actions, enemies, memmap as mm, terrain
from sentinel.astar_player import AStarPlayer
from sentinel.game import Game
from sentinel.playerbase import ROBOT_EYE

_LANDSCAPES = (0, 7, 42, 110, 200, 335, 1024, 2024)  # typed codes (Game.typed)
_MAX_PEDESTAL = 6  # boulders a synthetic perch may stack
_NEAR, _MAX_RANGE = 2, 8  # chebyshev band the perch is drawn from
_STANCES = 4  # perches kept per board: the "which stance" fuzz axis
_PROBES = 60  # candidate tiles tried while looking for those perches
_EYE_MARGIN = 0.6  # eye clearance over the platform's top face
_NODE_BUDGET = 400  # the endgame is one macro deep; a wide search means a bug
_TIME_BUDGET = 30.0
_SAFE_WINDOW = 4000.0  # frames a surviving sentry must be off draining anything
_NEED = {1: 3, 2: 3, 3: 6}  # energy floor: 3 per hyperspace, +3 for the robot

_BASES = {}


def _strip_enemies(st, keep=None):
    """Absorb every enemy, the Sentinel LAST: ``try_to_absorb_object`` $1B8E opens on
    objects_flags[0], so an absorbed Sentinel locks out every later absorb."""
    for slot in enemies.enemy_slots(st):
        if slot in (actions.SENTINEL_SLOT, keep):
            continue
        actions.absorb(st, slot)
    actions.absorb(st, actions.SENTINEL_SLOT)


def _perch(st, tile, height):
    """Stack ``height`` boulders on ``tile``, top it with a robot and move the player
    into it -- the pedestal a human climbs, built from the model's own primitives
    (line of sight is the caller's business, and the caller checks it)."""
    for _ in range(height):
        if actions.create(st, mm.T_BOULDER, tile) is None:
            return False
    slot = actions.create(st, mm.T_ROBOT, tile)
    if slot is None:
        return False
    old = st.player
    actions.transfer(st, slot)
    if old != slot:
        actions.remove_object(st, old)
    return True


def _quiet(player, ptile):
    """Whether a surviving sentry is far enough off aim that neither the player's body
    nor the platform can be drained inside the whole endgame."""
    if player._frozen():
        return False  # a frozen clock hides the sentry rather than clearing it
    return min(player._player_window(), player._gaze_window(ptile)) >= _SAFE_WINDOW


def _perches(landscape, keep_sentry):
    """Up to ``_STANCES`` boards for ``landscape`` with the player perched where the
    platform tile is genuinely landable (the search's ``_landable`` AND the executor's
    ``_view_for``), every enemy gone bar an optional far-off sentry."""
    key = (landscape, keep_sentry)
    if key in _BASES:
        return _BASES[key]
    board = Game.typed(landscape).state
    sentry = None
    if keep_sentry:
        sentry = next(
            (s for s in enemies.enemy_slots(board) if s != actions.SENTINEL_SLOT), None
        )
    _strip_enemies(board, keep=sentry)
    board.energy = mm.ENERGY_MASK  # build freely; the real energy is set per case
    ptile = board.platform_xy
    top = board.obj_z_height[actions.PLATFORM_SLOT] + 1  # $1F47: one unit high
    tiles = [
        (x, y)
        for x in range(mm.N)
        for y in range(mm.N)
        if _NEAR <= max(abs(x - ptile[0]), abs(y - ptile[1])) <= _MAX_RANGE
    ]
    random.Random(landscape).shuffle(tiles)
    out = []
    for tile in tiles[:_PROBES]:
        ground = terrain.tile_byte(board, *tile) >> 4
        height = max(0, math.ceil((top + _EYE_MARGIN - ROBOT_EYE - ground) * 2))
        if height > _MAX_PEDESTAL:
            continue
        st = board.clone()
        if not _perch(st, tile, height):
            continue
        player = AStarPlayer(Game(st), node_budget=1)
        if player._view_for(ptile) is None or not player._landable(st, ptile):
            continue
        if sentry is not None and not _quiet(player, ptile):
            continue
        out.append(st)
        if len(out) >= _STANCES:
            break
    _BASES[key] = out
    return out


def _endgame(landscape, distance, rng):
    """A board exactly ``distance`` legal moves from a win, or None if this landscape
    offers no perch.  The moves spent building it ARE the tail's leading moves, so
    what remains is the tail's remaining length."""
    perches = _perches(landscape, bool(rng.getrandbits(1))) or _perches(
        landscape, False
    )
    if not perches:
        return None
    st = rng.choice(perches).clone()
    ptile = st.platform_xy
    if distance <= 2:
        slot = actions.create(st, mm.T_ROBOT, ptile)
        assert slot is not None, "the platform tile refused a robot"
        if distance == 1:
            assert actions.transfer(st, slot)
            assert actions.on_platform(st)
    st.energy = min(_NEED[distance] + rng.randrange(8), mm.ENERGY_MASK)
    return st


@pytest.mark.parametrize("distance", (1, 2, 3))
@pytest.mark.parametrize("landscape", _LANDSCAPES)
def test_planner_closes_a_known_distance_endgame(landscape, distance):
    """``d`` moves from a win, the planner must win in at most ``2d`` actions -- and
    must ACT: an empty trace is a planning freeze, which "lost" alone cannot tell
    apart from a legitimate loss (the d=2 bug produced exactly that)."""
    rng = random.Random(landscape * 4 + distance)
    st = _endgame(landscape, distance, rng)
    if st is None:
        pytest.skip(f"ls{landscape}: no synthetic perch sees the platform")
    player = AStarPlayer(Game(st), node_budget=_NODE_BUDGET, time_budget=_TIME_BUDGET)
    won = player.run(max_actions=3 * distance + 2)
    verbs = [rec[1] for rec in player.trace]
    assert verbs, f"ls{landscape} d={distance}: planner froze, zero actions"
    assert won, f"ls{landscape} d={distance}: lost with {verbs}"
    assert len(verbs) <= 2 * distance, f"ls{landscape} d={distance}: {verbs}"
    assert verbs[-1] == "hyperspace"  # the only move that sets $0CDE bit6


def test_endgame_distances_are_exact_by_construction():
    """d=1 stands on the platform, d=2 has a robot there with the player off it, d=3
    has neither -- and none of the three is already won."""
    rng = random.Random(11)
    shapes = ((1, True, True), (2, False, True), (3, False, False))
    for distance, on_plat, has_robot in shapes:
        st = _endgame(_LANDSCAPES[0], distance, rng)
        assert st is not None
        assert actions.on_platform(st) is on_plat
        top = terrain.top_object(st, *st.platform_xy)
        assert (st.obj_type[top] == mm.T_ROBOT) is has_robot
        assert st.energy >= _NEED[distance]
        assert not actions.won(st)


def test_a_robot_on_the_platform_is_transferred_into_not_rebuilt():
    """The d=2 defect at its source: ``_c_endgame`` must yield a successor when the
    platform tile already carries a robot -- ``create`` there is refused by $1F38, and
    the endgame is the only child a Sentinel-free board has."""
    st = _endgame(_LANDSCAPES[0], 2, random.Random(3))
    assert st is not None and st.is_empty(actions.SENTINEL_SLOT)
    player = AStarPlayer(Game(st), node_budget=_NODE_BUDGET, time_budget=_TIME_BUDGET)
    plan = player._search()
    assert plan, "no plan two moves from the win"
    assert [step.verb for step in plan] == ["transfer", "hyperspace"]
    assert plan[0].tile == st.platform_xy
