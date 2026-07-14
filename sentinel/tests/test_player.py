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


def test_player_never_transfers_into_gaze():
    """Every transfer in the winning trace landed outside an enemy's live cone
    (the player re-checks the gaze window before each one)."""
    game = Game.new(0)
    player = Player(game)
    gazes = []
    orig = Player._fire

    def spy(self, verb, tile, view):
        if verb == "transfer":
            gazes.append(self._gaze_window(tile))
        return orig(self, verb, tile, view)

    Player._fire = spy
    try:
        assert player.run(max_actions=100)
    finally:
        Player._fire = orig
    assert gazes and all(w > 0 for w in gazes)
