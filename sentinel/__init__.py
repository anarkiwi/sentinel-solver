"""Standalone, bit-exact simulator of The Sentinel (Geoff Crammond, 1986, C64).

A pure-Python forward model of the whole game -- terrain, line-of-sight, the
player actions (absorb / create / transfer / win), the energy economy, the
enemy rotation/drain/meanie dynamics, and a from-scratch landscape generator.
It needs no emulator to run; the 6502 code is used only in the test suite as a
parity oracle.

The single mutable state is a 64 KB ``bytearray`` laid out exactly like the
game's RAM (see :mod:`sentinel.memmap`), so every mechanic reads and writes the
same addresses the original does and the line-of-sight stays bit-exact.
"""

from sentinel.prng import Prng, seed_state

__all__ = ["Prng", "seed_state"]
