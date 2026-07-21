"""A small facade tying the simulator together: build a board, read it, take the
player actions and advance the enemies -- all on one bit-exact state, with no
emulator.  Line-of-sight questions go to :mod:`sentinel.threat`/:mod:`sentinel.los`.

    >>> from sentinel.game import Game
    >>> g = Game.new(42)
    >>> g.player_xy(), g.energy
    ((14, 27), 10)
    >>> g.step_enemies()          # advance the world one round
    >>> won = g.won()

Every method delegates to the package modules (:mod:`sentinel.landscape`,
:mod:`sentinel.actions`, :mod:`sentinel.enemies`); ``Game`` just holds the
:class:`~sentinel.state.State` and offers them under one object.
"""

from sentinel import landscape, actions, enemies, memmap as mm


class Game:
    """A live game on one :class:`~sentinel.state.State`."""

    __slots__ = ("state",)

    def __init__(self, state):
        self.state = state

    # -- construction --------------------------------------------------------
    @classmethod
    def typed(cls, landscape_number):
        """The board a player gets by TYPING ``landscape_number`` -- the canonical id.

        ``Game.typed(335)`` is the ls335 everyone means; ``Game.new(335)`` is the raw
        seed 335, a different board (:func:`sentinel.landscape.seed_for`).
        """
        return cls.new(landscape.seed_for(landscape_number))

    @classmethod
    def new(cls, landscape_number):
        """Generate the board for ``landscape_number`` from scratch, in the ROM's
        at-entry state: enemies frozen until the player's first action ($0CE5
        bit7, skipped at $3682; cleared by the action dispatch $12E1)."""
        game = cls(landscape.generate(landscape_number))
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

    def objects(self):
        """(slot, type, x, y) for every occupied slot."""
        st = self.state
        return [
            (s, st.obj_type[s], st.obj_x[s], st.obj_y[s]) for s in st.occupied_slots()
        ]

    def enemy_slots(self):
        return enemies.enemy_slots(self.state)

    # -- player actions ------------------------------------------------------
    def can_create(self, otype, tile):
        return actions.can_create(self.state, otype, tile)

    def create(self, otype, tile):
        return actions.create(self.state, otype, tile)

    def can_absorb(self, slot):
        return actions.can_absorb(self.state, slot)

    def absorb(self, slot):
        return actions.absorb(self.state, slot)

    def transfer(self, slot):
        return actions.transfer(self.state, slot)

    def win(self, tile=None):
        return actions.win(self.state, tile)

    def won(self):
        return actions.won(self.state)

    # -- enemy dynamics ------------------------------------------------------
    def step_enemies(self):
        """Advance the world by one game round (cooldowns + one enemy update)."""
        enemies.step(self.state)

    def meanie_threat(self, enemy):
        return enemies.meanie_threat(self.state, enemy)
