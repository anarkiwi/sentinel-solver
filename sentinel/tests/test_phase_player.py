"""The phase player's rules, pinned on the cheapest board that exercises them."""

import pytest

from sentinel import actions, enemies, phase_player, terrain, memmap as mm
from sentinel.phase_player import PhasePlayer
from sentinel.game import Game
from sentinel.playerbase import BOULDER_H, ROBOT_EYE, _Views

STACK_TILE = (4, 4)  # ls42: bare, and every create on it is granted


def _player(code=0):
    return PhasePlayer(Game.typed(code), verbose=False)


class _Barren:
    """A stance whose landable sets are both empty: no verb has a target."""

    @staticmethod
    def band():
        return {}

    @staticmethod
    def primary():
        return {}


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
    monkeypatch.setattr(phase_player.actions, "won", lambda st: id(st) in winners)
    assert player._best_climb(_Views(player.st), affordable=True)[0] == (2, 2)
    assert order == [(1, 1), (2, 2)]  # stops at the first winner
    assert twins[1].built == ((2, 2), 0)


def test_a_rollout_never_settles_a_tie(monkeypatch):
    """A fork plays the fixed ladder: settling inside one would nest without bound."""
    player = PhasePlayer(Game.typed(0), rollout=True)
    monkeypatch.setattr(player, "_climb_candidates", lambda *_a: [_tie((1, 1))] * 2)

    def boom():
        raise AssertionError("a rollout forked")

    monkeypatch.setattr(player, "_fork", boom)
    assert player._best_climb(_Views(player.st), affordable=True)[0] == (1, 1)


def test_a_barren_stance_is_left_by_hyperspace():
    """Nothing landable means every generator is empty, and no wait can change that:
    $216A is the one move that shifts the body without a sightline."""
    player = _player()
    before = player.st.energy
    assert player._relocate(_Barren())
    assert player.trace[-1][1] == "hyperspace"
    assert player.st.energy == before - mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]


def test_a_jump_that_would_underflow_is_not_taken():
    """$2170 kills below the toll, so the purse IS the whole affordability test."""
    player = _player()
    player.st.energy = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT] - 1
    assert not player._relocate(_Barren())
    assert not player.trace


def test_a_stance_with_a_move_is_not_barren():
    """Relocation is the empty-generator case only; a refused move is a wait."""
    player = _player()
    views = _Views(player.st)
    assert player._climb_candidates(views, True)
    assert not player._barren(views)
    assert not player._relocate(views)
    assert not player.trace


@pytest.mark.xfail(
    strict=True,
    reason="_stance_base returns a FOOT for a bare tile and an already-robot eye for a "
    "stacked one, and _climb_candidates adds ROBOT_EYE to both, so on any tile that "
    "already carries a boulder the predicted eye is 0.875 high: open item 14. Making "
    "_stance_base return the eye a robot built now would have, and dropping ROBOT_EYE "
    "from its callers, makes this pass -- and loses ls373.",
)
def test_the_predicted_climb_eye_is_the_eye_a_robot_built_there_gets():
    """Pinned by construction on ls42: stack `already` boulders, predict what the
    ranker predicts, build it and compare."""
    start = _player(42).st
    start.energy = 60  # the geometry is the subject, not the purse
    assert terrain.top_object(start, *STACK_TILE) is None
    for already in range(3):
        base = start.clone()
        for _ in range(already):
            assert actions.create(base, mm.T_BOULDER, STACK_TILE) is not None
        for k in (0, 1):
            trial = base.clone()
            predicted = PhasePlayer(Game(trial))._stance_base(STACK_TILE)
            predicted += BOULDER_H * k + ROBOT_EYE  # the _climb_candidates convention
            for _ in range(k):
                assert actions.create(trial, mm.T_BOULDER, STACK_TILE) is not None
            slot = actions.create(trial, mm.T_ROBOT, STACK_TILE)
            assert slot is not None
            assert trial.eye_z(slot) == predicted


def test_harvest_reports_the_absorb_it_took_not_the_purse_it_wanted(monkeypatch):
    """One +1 tree in view is a move, and _arbitrate drops any option whose policy
    says False -- so reporting the GOAL threw away the only action on the board."""
    player = _player()
    monkeypatch.setattr(player, "_reclaim_targets", lambda *_a, **_k: [(1, (1, 1))])

    def take(_verb, _tile, _views, cost=0, band=False):
        del cost, band
        player.st.energy += 1
        return True

    monkeypatch.setattr(player, "_do", take)
    before = player.st.energy
    views = _Views(player.st)
    assert player._harvest(views)
    assert player.st.energy == before + 1  # one tree, two wanted
    assert not player._refuel(views, player.st.energy + 2)  # _refuel's contract intact


def test_a_fork_carries_this_class_not_the_base():
    """Naming PhasePlayer would evaluate every arbitrated option under the BASE
    policies, so a subclass's overrides would never be searched."""

    class _Sub(PhasePlayer):
        pass

    class _Live(PhasePlayer):
        """A player whose constructor is not the simulator's pins the twin class."""

        def _twin_class(self):
            return PhasePlayer

    assert isinstance(_Sub(Game.typed(0))._fork(), _Sub)
    twin = _Live(Game.typed(0))._fork()
    assert twin.__class__ is PhasePlayer and twin.rollout  # not a _Live twin


def test_ls0_is_won():
    """End to end on the smallest board: the phases hand over and the game completes."""
    player = _player(0)
    won = player.run(max_actions=120)
    assert won and actions.won(player.st)
    assert not enemies.enemy_slots(player.st)
    assert not actions.player_dead(player.st)
