"""The reactive greedy player (sentinel.player) beats landscape 0."""

from sentinel import actions, memmap as mm
from sentinel.game import Game
from sentinel.player import Player


def test_player_wins_landscape_0():
    """The player wins ls0: hyperspace from the platform tile, alive, solvent."""
    game = Game.new(0)
    player = Player(game)
    assert player.run(max_actions=100)
    assert actions.won(game.state)
    assert not actions.player_dead(game.state)
    assert game.energy >= mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
    verbs = [verb for _, verb, *_ in player.trace]
    assert verbs[-1] == "hyperspace"
    assert game.state.is_empty(actions.SENTINEL_SLOT)


def test_player_wins_landscape_0042():
    """Seed 66 (typed 0042, two enemies, down-look-only start hollow): the band
    climb fallback and frozen-world model get the player out and to the win."""
    game = Game.new(66)
    player = Player(game)
    assert player.run(max_actions=250)
    assert actions.won(game.state)
    assert not actions.player_dead(game.state)


def _placement_breaches(seed, max_actions=250):
    """Run the player; return (won, breaches) from its built-in audit.  A breach
    is any create/transfer left in an enemy's live scan cone POST-SETTLE, judged
    by the ROM's own relative.can_see_object (independent of the planner)."""
    game = Game.new(seed)
    player = Player(game, audit=True)
    won = player.run(max_actions=max_actions)
    return won, player.breaches


def test_player_placement_invariant():
    """No create or transfer leaves the player's body in an enemy's live cone on
    the winning boards."""
    for seed in (0, 66):
        won, breaches = _placement_breaches(seed)
        assert won, f"seed {seed} did not win"
        assert not breaches, f"seed {seed}: objects left in a live cone: {breaches}"
