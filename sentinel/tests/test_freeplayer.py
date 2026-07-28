"""The phase player's rules, pinned on the cheapest board that exercises them."""

from sentinel import actions, enemies, freeplayer, memmap as mm
from sentinel.freeplayer import FreePlayer
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
    assert not player._drained_over(600.0)
    assert not player._under_fire()


def test_waiting_is_refused_while_under_fire():
    """Idling is free only when unexposed; under a cone every quantum is a drain."""
    player = _player()
    player._under_fire = lambda: True
    player._drained_over = lambda _frames: True
    assert not player._wait_for_gap(600.0)


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


class _Twin:
    """A fork stand-in that reaches its own state without touching the board."""

    def __init__(self, state):
        self.st = state
        self.built = None

    def _build_and_mount(self, _views, tile, k):
        self.built = (tile, k)
        return True

    def run(self, max_actions=0):
        del max_actions


def _tie(tile):
    return ((0.1667, 5), tile, 0, 3, 0.5)


def test_a_tie_is_settled_by_the_outcome_not_the_order(monkeypatch):
    """Equal scores hold no information, and the winning hop's payoff is 40 actions
    out, so the tie is broken by playing each candidate to the END."""
    player = _player()
    tied = [_tie((1, 1)), _tie((2, 2)), _tie((3, 3))]
    monkeypatch.setattr(player, "_climb_candidates", lambda *_a: tied)
    twins = [_Twin(player.st.clone()) for _ in tied]
    winners = {id(twins[1].st), id(twins[2].st)}
    order = []

    def fork():
        order.append(tied[len(order)][1])
        return twins[len(order) - 1]

    monkeypatch.setattr(player, "_fork", fork)
    monkeypatch.setattr(freeplayer.actions, "won", lambda st: id(st) in winners)
    assert player._best_climb(_Views(player.st), affordable=True)[0] == (2, 2)
    assert order == [(1, 1), (2, 2)]  # stops at the first winner
    assert twins[1].built == ((2, 2), 0)


def test_a_rollout_never_settles_a_tie(monkeypatch):
    """A fork plays the fixed ladder: settling inside one would nest without bound."""
    player = FreePlayer(Game.typed(0), rollout=True)
    monkeypatch.setattr(player, "_climb_candidates", lambda *_a: [_tie((1, 1))] * 2)

    def boom():
        raise AssertionError("a rollout forked")

    monkeypatch.setattr(player, "_fork", boom)
    assert player._best_climb(_Views(player.st), affordable=True)[0] == (1, 1)


def test_ls0_is_won():
    """End to end on the smallest board: the phases hand over and the game completes."""
    player = _player(0)
    won = player.run(max_actions=120)
    assert won and actions.won(player.st)
    assert not enemies.enemy_slots(player.st)
    assert not actions.player_dead(player.st)
