"""Gaze timeline oracle: precomputed idle-enemy facings over a horizon, exposed
as safe-window queries used as an admissible heuristic / pruning filter.

This is a HEURISTIC device, not the correctness transition. It forward-sims a
PASSIVE clone of the state (the real :func:`sentinel.enemies.step`, so it captures
pre-existing-scenery drain/discharge events) and records each idle enemy's
horizontal facing per round. A tile is "seen" at round ``t`` when some enemy's
static terrain line-of-sight reaches it AND the tile's bearing falls inside that
enemy's scan cone at round ``t``. Once the player actually acts and perturbs the
enemies, the true transition governs and this timeline is only a pruning bound.

The cone gate and terrain factor mirror :func:`sentinel.relative.can_see_object`
under :data:`sentinel.enemies.FOV_SCAN` exactly, so at ``t == 0`` the oracle agrees
with :func:`sentinel.threat.ticks_until_seen` on every tile:

  * terrain LOS is the base-probe reach of ``can_see_object`` with the facing gate
    dropped (``FOV_FULL``), cached per (enemy, tile) since idle-enemy terrain is
    static;
  * the per-round cone is the ROM's FOV gate ``((bearing - facing + FOV_HALF) &
    0xFF) < FOV_SCAN`` where ``bearing`` is the bit-exact horizontal angle
    (``relative_angles``'s ``angle_hi``) -- NOT the analytic ``aimcost.bearing_to``,
    whose atan2 rounding disagrees with the ROM trig on ~113 ls0 tiles.
"""

import numpy as np

from sentinel import memmap as mm, relative, enemies, terrain

FOV_SCAN = enemies.FOV_SCAN  # 0x14: enemy horizontal scan FOV width
FOV_HALF = FOV_SCAN // 2  # 10
FOV_FULL = 0x100  # fov_width that drops the facing gate (terrain LOS only)

assert FOV_SCAN == 0x14, "unexpected enemies.FOV_SCAN"


def _place_phantom(state, tile, slot):
    """Place a phantom T_ROBOT at `tile` in `slot`, mirroring actions.create's
    bare-terrain / stack-on-boulder placement. No energy spent, no PRNG drawn.
    Returns True, or False if the tile can't host a robot. Inlined from
    threat._place_phantom."""
    x, y = tile
    b = terrain.tile_byte(state, x, y)
    if b >= mm.OBJECT_TILE:  # stacking on an existing object
        below = b & 0x3F
        if state.obj_type[below] != mm.T_BOULDER:
            return False
        t = state.obj_z_frac[below] + 0x80  # boulders are half a unit high
        zf = t & 0xFF
        z = (state.obj_z_height[below] + (t >> 8)) & 0xFF
        flags = 0x40 | below
    else:  # bare terrain
        zf = 0xE0
        z = (b >> 4) & 0xFF
        flags = 0x00
    state.obj_type[slot] = mm.T_ROBOT
    state.obj_flags[slot] = flags
    state.obj_x[slot] = x
    state.obj_y[slot] = y
    state.obj_z_height[slot] = z
    state.obj_z_frac[slot] = zf
    terrain.set_tile_byte(state, x, y, mm.OBJECT_TILE | slot)
    return True


def _restore_tile(state, tile, old_tile_byte, slot):
    """Undo _place_phantom: restore the tile byte and mark `slot` empty."""
    terrain.set_tile_byte(state, tile[0], tile[1], old_tile_byte)
    state.obj_flags[slot] |= 0x80


class GazeTimeline:
    """Precomputed idle-enemy facings over `horizon` rounds with safe-window
    queries. Constructed once per world state; queries are O(horizon) and cached
    per tile."""

    def __init__(self, state, horizon=4000):
        self.horizon = int(horizon)
        self._enemies = enemies.enemy_slots(state)
        n = len(self._enemies)
        # Forward-sim a passive clone, recording each enemy's facing per round.
        clone = state.clone()
        facings = np.zeros((n, self.horizon), dtype=np.uint8)
        for t in range(self.horizon):
            for i, e in enumerate(self._enemies):
                facings[i, t] = clone.obj_h_angle[e]
            enemies.step(clone)
        self._facings = facings
        # A clone at t=0 for the static terrain-LOS / bearing probes.
        self._terr_state = state.clone()
        free = self._terr_state.free_slots()
        self._terr_slot = free[-1] if free else None
        self._tile_cache = {}  # (x, y) -> (placeable, terr[n]bool, bearing[n]uint8)
        self._seen_cache = {}  # (x, y) -> seen timeline (bool, len horizon)

    def _tile_info(self, x, y):
        """(placeable, terr, bearing) for `(x, y)`: per-enemy static terrain LOS
        and the ROM horizontal bearing enemy->tile. Cached (terrain is static)."""
        key = (x, y)
        cached = self._tile_cache.get(key)
        if cached is not None:
            return cached
        n = len(self._enemies)
        terr = np.zeros(n, dtype=bool)
        bearing = np.zeros(n, dtype=np.uint8)
        slot = self._terr_slot
        placeable = False
        if slot is not None:
            st = self._terr_state
            old = terrain.tile_byte(st, x, y)
            if _place_phantom(st, (x, y), slot):
                placeable = True
                for i, e in enumerate(self._enemies):
                    res = relative.can_see_object(st, e, slot, mm.T_ROBOT, FOV_FULL)
                    terr[i] = res["full"]
                    bearing[i] = relative.relative_angles(st, e, slot)["angle_hi"]
                _restore_tile(st, (x, y), old, slot)
        info = (placeable, terr, bearing)
        self._tile_cache[key] = info
        return info

    def _seen_timeline(self, x, y):
        """Boolean array over [0, horizon): True where some enemy sees `(x, y)`."""
        key = (x, y)
        cached = self._seen_cache.get(key)
        if cached is not None:
            return cached
        seen = np.zeros(self.horizon, dtype=bool)
        placeable, terr, bearing = self._tile_info(x, y)
        if placeable:
            for i in range(len(self._enemies)):
                if not terr[i]:
                    continue
                delta = (int(bearing[i]) - self._facings[i].astype(np.int16)) & 0xFF
                seen |= ((delta + FOV_HALF) & 0xFF) < FOV_SCAN
        self._seen_cache[key] = seen
        return seen

    def seen_at(self, x, y, t) -> bool:
        """Whether some enemy sees `(x, y)` at round `t` (terrain LOS AND the
        enemy's scan cone at `t`)."""
        if t < 0 or t >= self.horizon:
            return False
        return bool(self._seen_timeline(x, y)[t])

    def safe_windows(self, x, y):
        """Maximal [t0, t1] inclusive intervals over [0, horizon) with no enemy
        sight of `(x, y)` (the complement of the seen intervals)."""
        seen = self._seen_timeline(x, y)
        out = []
        start = None
        for t in range(self.horizon):
            if not seen[t]:
                if start is None:
                    start = t
            elif start is not None:
                out.append((start, t - 1))
                start = None
        if start is not None:
            out.append((start, self.horizon - 1))
        return out

    def is_safe(self, x, y, t0, t1) -> bool:
        """Whether [t0, t1] lies wholly inside one safe window."""
        if t0 < 0 or t1 >= self.horizon or t1 < t0:
            return False
        return not bool(self._seen_timeline(x, y)[t0 : t1 + 1].any())

    def ticks_until_seen(self, x, y, t_from) -> int:
        """First round `t >= t_from` at which `(x, y)` is seen; `horizon` if none
        within the horizon."""
        if t_from >= self.horizon:
            return self.horizon
        seen = self._seen_timeline(x, y)
        hits = np.nonzero(seen[max(0, t_from) :])[0]
        if len(hits) == 0:
            return self.horizon
        return max(0, t_from) + int(hits[0])
