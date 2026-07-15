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


def test_player_placement_invariant():
    """No create or transfer ever leaves an object inside an enemy's LIVE scan
    cone, judged by the ROM's own visibility test on the ACTUAL object right
    after the action fires (not the player's own predicate, which a phantom
    quirk once blinded to transfer destinations)."""
    from sentinel import enemies, relative
    from sentinel.terrain import top_object

    for seed in (0, 66):
        game = Game.new(seed)
        player = Player(game)
        bad = []
        orig = Player._fire

        def spy(self, verb, tile, view):
            ok = orig(self, verb, tile, view)
            if ok and verb in ("boulder", "robot", "transfer"):
                st = self.st
                top = top_object(st, *tile)
                seen = []
                for e in enemies.enemy_slots(st):
                    see = relative.can_see_object(
                        st, e, top, st.obj_type[top], enemies.FOV_SCAN
                    )
                    if not see["exposure"]:
                        continue
                    if verb == "transfer" and not (
                        see["full"] or self._tree_near(tuple(tile))
                    ):
                        continue  # harmless partial glimpse: undrainable, no meanie
                    seen.append(e)
                if seen:
                    bad.append((verb, tuple(tile), seen))
            return ok

        Player._fire = spy
        try:
            assert player.run(max_actions=250), f"seed {seed} did not win"
        finally:
            Player._fire = orig
        assert not bad, f"seed {seed}: objects left in a live cone: {bad}"
