#!/usr/bin/env python3
"""Live-execution constants and a thin Executor used by the plan runner.

Provides the verified key mapping for CREATE/ABSORB/TRANSFER/HYPERSPACE actions
and an Executor exposing raw memory reads (`rd`) and a decoded live GameState
(`state`) over a BinMon connection.
"""

from driver import sentinel_state as gs
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


class Executor:
    def __init__(self, bm, log):
        self.bm = bm
        self.log = log

    def rd(self, a):
        return self.bm.mem_get(a, a)[0]

    def state(self):
        return gs.read_game_state(gs.ViceSource(self.bm))
