"""The phase player's rules, pinned on the cheapest board that exercises them."""

from sentinel import actions, enemies, memmap as mm
from sentinel.freeplayer import BUILD_FRAMES, FreePlayer
from sentinel.game import Game
from sentinel.playerbase import _Views


def _player(code=0):
    return FreePlayer(Game.typed(code), verbose=False)


def test_an_unseen_body_needs_no_drain_simulation():
    """Geometry answers it: no sightline, no drain, whatever the cones do."""
    player = _player()
    st = player.st
    for slot in enemies.enemy_slots(st):
        st.obj_flags[slot] = 0x80  # no enemies at all -> nothing can see us
    assert not player._drained_over(BUILD_FRAMES["robot"])
    assert not player._under_fire()


def test_waiting_is_refused_while_under_fire():
    """Idling is free only when unexposed; under a cone every quantum is a drain."""
    player = _player()
    player._under_fire = lambda: True
    player._drained_over = lambda _frames: True
    assert not player._wait_for_gap(BUILD_FRAMES["robot"])


def test_the_sentinel_is_not_taken_without_the_endgame_banked():
    """$1B91 refuses every absorb once slot 0 empties, so the finish must be paid for
    before the Sentinel dies, not after."""
    player = _player()
    st = player.st
    for slot in enemies.enemy_slots(st):
        if st.obj_type[slot] != mm.T_SENTINEL:
            st.obj_flags[slot] = 0x80
    st.energy = 0  # cannot fund robot + hyperspace
    player._refuel = lambda *a, **k: False
    assert not player._strike(object())


def test_a_landing_that_kills_us_is_not_a_candidate():
    """The destination's cone decides a hop, not the one we are leaving."""
    player = _player()
    player._landing_holds = lambda _tile, _k: False
    assert player._best_climb(_Views(player.st), affordable=True) is None


def test_ls0_is_won():
    """End to end on the smallest board: the phases hand over and the game completes."""
    player = _player(0)
    won = player.run(max_actions=120)
    assert won and actions.won(player.st)
    assert not enemies.enemy_slots(player.st)
    assert not actions.player_dead(player.st)
