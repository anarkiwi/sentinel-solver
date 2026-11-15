"""A board built the way the game enters it, as one bit-exact state, no emulator.

>>> from sentinel.game import Game
>>> Game.typed(42).player_xy()        # the landscape number a player keys in
(13, 29)
"""

from sentinel import landscape, memmap as mm


class Game:
    """A live game on one :class:`~sentinel.state.State`."""

    __slots__ = ("state",)

    def __init__(self, state):
        self.state = state

    # -- construction --------------------------------------------------------
    @classmethod
    def typed(cls, landscape_number):
        """Landscape ``landscape_number`` -- the number a player KEYS IN, the only id
        a landscape has.  Every tool and test names a board this way.
        """
        return cls.new(landscape.seed_for(landscape_number))

    @classmethod
    def new(cls, prnd_seed):
        """Generate a board from the ROM's raw PRNG seed value (see
        :func:`sentinel.landscape.seed_for`) in the at-entry state: enemies frozen
        until the player's first action ($0CE5 bit7, skipped at $3682; cleared by the
        action dispatch $12E1).  Callers naming a landscape want :meth:`typed`."""
        game = cls(landscape.generate(prnd_seed))
        game.state.mem[mm.CURSOR] = 7
        game.state.mem[mm.COOLDOWN_GATE] = 0
        game.state.mem[mm.PLAYER_NOT_ACTED] = 0x80
        return game

    def clone(self):
        """A deep, independent copy (search branches without side effects)."""
        return Game(self.state.clone())

    # -- board queries -------------------------------------------------------
    @property
    def energy(self):
        return self.state.energy

    def player_xy(self):
        return self.state.player_xy()

    def platform_xy(self):
        return self.state.platform_xy

    def enemy_slots(self):
        return self.state.enemy_slots()
