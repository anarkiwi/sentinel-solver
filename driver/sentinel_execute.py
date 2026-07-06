#!/usr/bin/env python3
"""Live-execution constants and a thin Executor used by the plan runner.

Provides the verified key mapping for CREATE/ABSORB/TRANSFER/HYPERSPACE actions
and an Executor exposing raw memory reads (`rd`) and a decoded live GameState
(`state`) over a BinMon connection.
"""

from driver import sentinel_state as gs
from sentinel import memmap as mm
from sentinel.memmap import T_BOULDER, T_ROBOT, T_TREE

# action keys (vice_driver.keys names), decoded from the game's key-number table
# $138D + action-code table $139C and confirmed live:
#   A = absorb ($20), Q = transfer ($21), R = create robot ($00),
#   T = create tree ($02), B = create boulder ($03), H = hyperspace ($22).
K_ABSORB = "A"
K_TRANSFER = "Q"
K_CREATE_ROBOT = "R"
K_CREATE_TREE = "T"
K_CREATE_BOULDER = "B"
K_HYPERSPACE = "H"

CREATE_KEY = {
    T_ROBOT: K_CREATE_ROBOT,
    T_TREE: K_CREATE_TREE,
    T_BOULDER: K_CREATE_BOULDER,
}


def otype_cost(otype):
    """The ROM energy a create of ``otype`` spends / an absorb of it refunds
    (energy_in_objects $214F, via sentinel.memmap)."""
    return mm.ENERGY_IN_OBJECTS.get(otype, 3)


def verify(verb, otype, tile, before, after, objs0, objs1, slot0, slot1, e0, e1):
    """Arbitrate whether a fired action really did what the plan step intended, from
    the live memory delta: the EXACT on-tile object-count change AND the EXACT energy
    delta, flagging any other global object-count change (wrong-tile landing, meanie
    spawn, held-key extra creates) as a divergence. Returns ``(ok, message)``."""
    dtot = len(after.objects) - len(before.objects)
    if verb == "create":
        if objs1 != objs0 + 1:
            return (
                False,
                f"create wrong-tile/none on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}",
            )
        if dtot != 1:
            return (
                False,
                f"create changed global object count by {dtot} (meanie/extra?); E {e0}->{e1}",
            )
        exp = (e0 - otype_cost(otype)) & 0x3F
        if e1 != exp:
            return (
                False,
                f"create energy {e0}->{e1} != expected {exp} (cost {otype_cost(otype)})",
            )
        return True, f"object created on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}"
    if verb == "transfer":
        moved = (slot1 != slot0) or (
            after.player and (after.player.x, after.player.y) == tile
        )
        if moved:
            return (
                True,
                f"slot {slot0}->{slot1}, now ({after.player.x},{after.player.y})",
            )
        return False, f"transfer did not move player (slot {slot0}->{slot1})"
    if verb == "absorb":
        if objs1 != objs0 - 1:
            return (
                False,
                f"absorb wrong-tile/none on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}",
            )
        if dtot != -1:
            return False, f"absorb changed global object count by {dtot}; E {e0}->{e1}"
        exp = (e0 + otype_cost(otype)) & 0x3F
        if e1 != exp:
            return (
                False,
                f"absorb energy {e0}->{e1} != expected {exp} (refund {otype_cost(otype)})",
            )
        return True, f"object absorbed on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}"
    return False, "?"


class Executor:
    def __init__(self, bm, log):
        self.bm = bm
        self.log = log

    def rd(self, a):
        return self.bm.mem_get(a, a)[0]

    def state(self):
        return gs.read_game_state(gs.ViceSource(self.bm))

    def platform(self):
        """The Sentinel's platform tile (x, y)."""
        return (self.rd(mm.PLATFORM_X), self.rd(mm.PLATFORM_Y))

    def landscape_done(self):
        """The raw landscape-complete byte ($0CDE); bit6 is the win flag."""
        return self.rd(mm.LANDSCAPE_COMPLETE)

    def won(self):
        """Whether the landscape is complete ($0CDE bit6 set)."""
        return bool(self.landscape_done() & 0x40)
