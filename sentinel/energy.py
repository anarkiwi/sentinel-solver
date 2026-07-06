"""The energy economy (gain_or_lose_energy_from_object $2136, table $214F,
set_player_energy $2148).

Every object type has a fixed energy value; absorbing adds it, creating spends
it.  The player meter is 6 bits: it wraps mod 64 on a gain and a create fails
(no change) if it would underflow.
"""

from sentinel import memmap as mm


def value(otype):
    """The energy value of an object type (table $214F)."""
    return mm.ENERGY_IN_OBJECTS[otype]


def gain(state, otype):
    """Add an object's energy to the player (carry-clear path of $2136); the
    meter wraps mod 64 (set_player_energy $2148 AND #$3F)."""
    state.energy = (state.energy + value(otype)) & mm.ENERGY_MASK


def lose(state, otype):
    """Spend an object's energy (carry-set path of $2136).  Returns False without
    changing the meter if the player has too little (the ROM's "out of energy"
    carry-set exit at $2143); otherwise debits and returns True."""
    cost = value(otype)
    if state.energy < cost:
        return False
    state.energy = (state.energy - cost) & mm.ENERGY_MASK
    return True
