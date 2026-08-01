"""Abandoned material pins the cone that can see it.

``_consider_enemy_state`` reaches its rotate branch ($17F9) only when nothing was
drainable -- a held target returns at $178C, a fully-visible robot at $17B2, a
boulder/stacked tree at $17E0 -- so leftovers in view stop an enemy turning.
"""

import pytest

from sentinel import actions, badline, enemies, memmap as mm, terrain
from sentinel.game import Game
from sentinel.playerbase import UNIT_FRAMES

DRAIN_STEP_FRAMES = enemies.UPDATE_COOLDOWN_DRAIN * UNIT_FRAMES


def _bait_board(otype=mm.T_BOULDER):
    """(game, enemy, tile) with ``otype`` somewhere the enemy can fully see it."""
    for y in range(32):
        for x in range(32):
            game = Game.typed(0)
            st = game.state
            st.mem[mm.PLAYER_NOT_ACTED] = 0  # the world runs
            enemy = enemies.enemy_slots(st)[0]
            if terrain.tile_byte(st, x, y) >= mm.OBJECT_TILE:
                continue
            slot = actions.create(st, otype, (x, y))
            if slot is None:
                continue
            st.obj_flags[st.player] = 0x80  # body off the board: only the bait remains
            scan = enemies._find_drainable_boulder_or_tree(
                st,
                enemy,
                enemies.UNBOUNDED,
                mm.NUM_SLOTS - 1,
                badline.frame_clock(False),
            )
            if scan[2] == slot:
                return game, enemy, (x, y)
    return None, None, None


def _facings(st, enemy, frames, step=25):
    seen = []
    for _ in range(0, frames, step):
        seen.append(int(st.obj_h_angle[enemy]))
        enemies.advance_frames(st, step)
    return seen


def test_a_lone_ground_tree_is_not_drainable():
    """$1AB0 marks candidates from boulders and STACKED objects only."""
    game = Game.typed(0)
    st = game.state
    for y in range(32):
        for x in range(32):
            if terrain.tile_byte(st, x, y) >= mm.OBJECT_TILE:
                continue
            slot = actions.create(st, mm.T_TREE, (x, y))
            if slot is None:
                continue
            assert st.obj_flags[slot] < 0x40  # on the ground, not stacked
            actions.remove_object(st, slot)
            return
    pytest.skip("no flat tile")


def test_a_boulder_in_view_stops_the_cone_turning():
    """The pin: while the bait stands, the enemy never reaches $17F9."""
    game, enemy, _tile = _bait_board()
    if game is None:
        pytest.skip("no tile the enemy can see a boulder on")
    st = game.state
    before = int(st.obj_h_angle[enemy])
    seen = _facings(st, enemy, int(DRAIN_STEP_FRAMES))
    assert set(seen) == {before}, "the cone turned while it had something to drain"


def _chain(st, tile, samples=60, step=25):
    """The tile's topmost type over time, deduplicated."""
    out = []
    for _ in range(samples):
        top = terrain.top_object(st, *tile)
        now = None if top is None or st.is_empty(top) else int(st.obj_type[top])
        if not out or out[-1] != now:
            out.append(now)
        if now is None:
            break
        enemies.advance_frames(st, step)
    return out


def test_a_lone_boulder_pins_for_exactly_one_step():
    """It drains to a tree and stalls there: its own product is a LONE ground tree,
    which $1AB0 will not mark, so the pin is one step and then the cone is free."""
    game, enemy, tile = _bait_board()
    if game is None:
        pytest.skip("no tile the enemy can see a boulder on")
    st = game.state
    start = int(st.obj_h_angle[enemy])
    assert _chain(st, tile) == [mm.T_BOULDER, mm.T_TREE]
    turned = False
    for _ in range(60):
        enemies.advance_frames(st, 25)
        if int(st.obj_h_angle[enemy]) != start:
            turned = True
            break
    assert turned, "the cone stayed frozen after the bait stopped being drainable"
