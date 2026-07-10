"""The landscape PRNG -- a 40-bit LFSR (prnd $31CA) seeded from the landscape
number (seed_prnd_from_landscape_number $33ED).

Every landscape (0..9999) is deterministic from its number: the terrain heights,
the enemy/tree placement and the per-enemy rotation directions are all drawn from
this one stream, so it must be byte-faithful.

prnd $31CA (per call, 8 shuffles of the 5-byte state at $0C7B-$0C7F):

    feedback = bit3(state[2]) XOR bit0(state[4])
    rotate the whole 40-bit value left by one bit, inserting `feedback` at
    bit0 of state[0]; the bit shifted out of the top (bit7 of state[4]) is
    discarded.

The call returns state[4] ($0C7F).
"""

from sentinel.memmap import PRND_STATE


class Prng:
    """The 40-bit LFSR.  Holds the 5-byte state as a plain list (state[0] is the
    low byte $0C7B ... state[4] is $0C7F, the returned byte)."""

    __slots__ = ("s",)

    def __init__(self, state=None):
        self.s = [0, 0, 0, 0, 0] if state is None else [b & 0xFF for b in state]

    def _shuffle(self):
        s = self.s
        # feedback = bit0 of ((state[2] >> 3) ^ state[4])  ($31D1-$31DA)
        carry = ((s[2] >> 3) ^ s[4]) & 1
        # ROL the 40-bit value left through carry, feedback entering at bit0.
        for i in range(5):
            v = (s[i] << 1) | carry
            carry = v >> 8
            s[i] = v & 0xFF
        # the final carry out (old bit7 of state[4]) is discarded.

    def next(self):
        """One prnd call: 8 shuffles, return state[4]."""
        for _ in range(8):
            self._shuffle()
        return self.s[4]

    @classmethod
    def seeded(cls, landscape):
        """seed_prnd_from_landscape_number $33ED: state[0]=lo, state[1]=hi of the
        landscape number, state[2..4]=0 (the $0C00 bank is fresh zeroed RAM, which
        is what makes landscapes reproducible across machines)."""
        return cls([landscape & 0xFF, (landscape >> 8) & 0xFF, 0, 0, 0])

    def load(self, mem):
        """Read the 5-byte state from a memory image at $0C7B."""
        self.s = [mem[PRND_STATE + i] for i in range(5)]
        return self

    def store(self, mem):
        """Write the 5-byte state back into a memory image at $0C7B."""
        for i in range(5):
            mem[PRND_STATE + i] = self.s[i]
        return self


def seed_state(landscape):
    """The 5-byte LFSR state for a landscape number, as a list."""
    return Prng.seeded(landscape).s
