"""Planner-facing enemy exposure and forecasting helpers.

Built on the bit-exact object visibility of :mod:`sentinel.relative` and the
round advance of :mod:`sentinel.enemies`, this module answers the geometric
questions a strategy search asks about standing/building on a tile:

  * :func:`is_exposed` / :func:`exposed_tiles` -- could ANY enemy at ANY rotation
    ever see a robot on this tile (terrain LOS, facing gate dropped)?
  * :func:`gaze_distance` -- how far off each enemy's CURRENT facing a tile sits.
  * :func:`ticks_until_seen` -- rounds until some enemy first sees the tile within
    its ACTUAL rotating field of view.
  * :func:`meanie_safe` -- whether standing here arms no meanie-spawn.
  * :func:`drain_over_window` -- energy the player loses while the world advances.

A visibility query places a phantom robot on the queried tile -- mirroring the
placement math of :func:`sentinel.actions.create` without spending energy or
drawing the PRNG -- runs the object-relative visibility, then restores the tile.
Every query operates on a clone or places-then-restores, so the caller's state is
never mutated.
"""

import math

from sentinel import memmap as mm, relative, enemies, terrain

FOV_FULL = 0x100  # fov_width that drops the facing gate (in_fov always True)
ROBOT_EYE = 0.875  # a robot's top above its foot tile ($E0 fraction)


def _free_slot(state):
    """The highest empty object slot, or None."""
    free = state.free_slots()
    return free[-1] if free else None


def _place_phantom(state, tile, slot):
    """Place a phantom T_ROBOT at `tile` in `slot`, mirroring actions.create's
    bare-terrain / stack-on-boulder placement (z_height, z_frac, obj_flags, tile
    byte, obj_type). No energy is spent and no PRNG is drawn. Returns True, or
    False if the tile can't be stood on (a non-boulder object is on top)."""
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
    """Undo :func:`_place_phantom`: restore the tile byte and mark `slot` empty."""
    terrain.set_tile_byte(state, tile[0], tile[1], old_tile_byte)
    state.obj_flags[slot] |= 0x80


def is_exposed(state, x, y, object_top=ROBOT_EYE):
    """Whether ANY enemy could ever see a robot on tile (x, y) with the facing gate
    dropped (terrain LOS only). False if the tile can't host a phantom robot."""
    clone = state.clone()
    slot = _free_slot(clone)
    if slot is None:
        return False
    old = terrain.tile_byte(clone, x, y)
    if not _place_phantom(clone, (x, y), slot):
        return False
    seen = any(
        relative.can_see_object(clone, e, slot, mm.T_ROBOT, FOV_FULL)["full"]
        for e in enemies.enemy_slots(clone)
    )
    _restore_tile(clone, (x, y), old, slot)
    return seen


def exposed_tiles(state, tiles, object_top=ROBOT_EYE):
    """The subset of `tiles` that are :func:`is_exposed`. Clones once and reuses a
    single free slot across the whole batch (the planner hot path)."""
    clone = state.clone()
    ens = enemies.enemy_slots(clone)
    result = set()
    if not ens:
        return result
    slot = _free_slot(clone)
    if slot is None:
        return result
    for x, y in tiles:
        old = terrain.tile_byte(clone, x, y)
        if not _place_phantom(clone, (x, y), slot):
            continue
        if any(
            relative.can_see_object(clone, e, slot, mm.T_ROBOT, FOV_FULL)["full"]
            for e in ens
        ):
            result.add((x, y))
        _restore_tile(clone, (x, y), old, slot)
    return result


def player_sees_tile(state, tile, observer_slot, eye_z=None):
    """True iff the observer at `observer_slot` can see `tile` (x, y) -- the ROM's
    direct observer->object geometric line of sight (relative.can_see_object with the
    facing gate dropped), the mirror of :func:`is_exposed`'s enemy->tile test. For an
    occupied tile (platform/Sentinel/boulder/...) the real object in its slot is tested;
    for bare terrain a phantom T_ROBOT is placed on the tile and tested. `eye_z`
    overrides the observer's standing height. Runs on a clone; the caller's state is
    never mutated."""
    clone = state.clone()
    if eye_z is not None:
        clone.obj_z_height[observer_slot] = int(eye_z) & 0xFF
    x, y = tile
    b = terrain.tile_byte(clone, x, y)
    if b >= mm.OBJECT_TILE:  # occupied -> test the real object in its slot
        slot = b & 0x3F
        typ = clone.obj_type[slot]
        return (
            relative.can_see_object(clone, observer_slot, slot, typ, FOV_FULL)[
                "exposure"
            ]
            > 0
        )
    slot = _free_slot(clone)  # bare terrain -> phantom-place a robot and test it
    if slot is None or not _place_phantom(clone, (x, y), slot):
        return False
    return (
        relative.can_see_object(clone, observer_slot, slot, mm.T_ROBOT, FOV_FULL)[
            "exposure"
        ]
        > 0
    )


def player_visible_footholds(state, observer_slot, eye_z, bare_only=True):
    """Every tile the observer at `observer_slot` (from `eye_z`) can see, as a set --
    the observer->tile geometric line of sight (the mirror of :func:`exposed_tiles`,
    and the batched form of :func:`player_sees_tile`). `bare_only` restricts to
    build-legal bare-terrain footholds (object-topped tiles excluded). Clones ONCE and
    reuses a single free slot across the whole board (the planner hot path), so it is
    far cheaper than a per-tile :func:`player_sees_tile` (one 64 KB clone, not 1024)."""
    clone = state.clone()
    if eye_z is not None:
        clone.obj_z_height[observer_slot] = int(eye_z) & 0xFF
    slot = _free_slot(clone)
    out = set()
    if slot is None:
        return out
    for x in range(mm.N):
        for y in range(mm.N):
            b = terrain.tile_byte(clone, x, y)
            if bare_only and b >= mm.OBJECT_TILE:
                continue
            if not _place_phantom(clone, (x, y), slot):
                continue
            if (
                relative.can_see_object(
                    clone, observer_slot, slot, mm.T_ROBOT, FOV_FULL
                )["exposure"]
                > 0
            ):
                out.add((x, y))
            _restore_tile(clone, (x, y), b, slot)
    return out


def gaze_distance(state, tiles):
    """For each tile, the minimum angular distance (0..128) from any enemy's
    CURRENT facing to the bearing toward that tile. 128 when there is no enemy.
    Larger == further out of an enemy's instantaneous line of sight right now."""
    tiles = list(tiles)
    best = {t: 128 for t in tiles}
    for s in range(mm.NUM_SLOTS):
        if state.obj_flags[s] & 0x80:
            continue
        if state.obj_type[s] not in mm.ENEMY_TYPES:
            continue
        ex, ey = state.obj_x[s], state.obj_y[s]
        gaze = state.obj_h_angle[s]
        for t in tiles:
            dx, dy = t[0] - ex, t[1] - ey
            if dx == 0 and dy == 0:
                continue
            bearing = int(round(math.atan2(dy, dx) * 128.0 / math.pi)) & 0xFF
            diff = (bearing - gaze) & 0xFF
            ang_diff = min(diff, 256 - diff)
            if ang_diff < best[t]:
                best[t] = ang_diff
    return best


def ticks_until_seen(state, x, y, horizon=256, object_top=ROBOT_EYE):
    """Rounds until some enemy first sees (x, y) within its ACTUAL rotating field
    of view; `horizon` if never within the horizon. 0 == seen now.

    The phantom is kept OUT of the enemy-rotation state -- placed only for the
    per-tick visibility test then restored -- so enemy targeting/rotation is not
    perturbed by the query target."""
    clone = state.clone()
    slot = _free_slot(clone)
    if slot is None:
        return horizon
    for t in range(horizon):
        old = terrain.tile_byte(clone, x, y)
        if _place_phantom(clone, (x, y), slot):
            seen = any(
                relative.can_see_object(clone, e, slot, mm.T_ROBOT, enemies.FOV_SCAN)[
                    "full"
                ]
                for e in enemies.enemy_slots(clone)
            )
            _restore_tile(clone, (x, y), old, slot)
            if seen:
                return t
        enemies.step(clone)
    return horizon


def meanie_safe(state, tile):
    """True iff standing at `tile` carries NO meanie-spawn risk.

    Ported from the ROM's meanie predicate: for some enemy that sees the player
    PARTIALLY at `tile` (A), and some tree within 10 tiles in both axes (B) that
    the enemy fully sees (C) and that can itself see the player (D), the enemy can
    arm a meanie -> unsafe. All tests run on a clone."""
    clone = state.clone()
    ens = enemies.enemy_slots(clone)
    if not ens:
        return True
    trees = [
        s
        for s in range(mm.NUM_SLOTS)
        if not clone.is_empty(s) and clone.obj_type[s] == mm.T_TREE
    ]
    if not trees:
        return True
    slot = _free_slot(clone)
    if slot is None:
        return True
    x, y = tile
    for e in ens:
        # (A) the enemy sees the player partially at `tile`.
        old = terrain.tile_byte(clone, x, y)
        if not _place_phantom(clone, (x, y), slot):
            return True
        res = relative.can_see_object(clone, e, slot, mm.T_ROBOT, FOV_FULL)
        partial = bool(
            res["in_fov"]
            and res["probes"]
            and res["probes"][0]
            and not res["probes"][1]
        )
        _restore_tile(clone, (x, y), old, slot)
        if not partial:
            continue
        for tr in trees:
            # (B) tree within 10 tiles of `tile` in both axes.
            if abs(clone.obj_x[tr] - x) >= 10 or abs(clone.obj_y[tr] - y) >= 10:
                continue
            # (C) the enemy fully sees the tree.
            if not relative.can_see_object(clone, e, tr, mm.T_TREE, FOV_FULL)["full"]:
                continue
            # (D) the tree can see the player.
            old2 = terrain.tile_byte(clone, x, y)
            if not _place_phantom(clone, (x, y), slot):
                continue
            sees_player = relative.can_see_object(
                clone, tr, slot, mm.T_ROBOT, FOV_FULL
            )["full"]
            _restore_tile(clone, (x, y), old2, slot)
            if sees_player:
                return False
    return True


def drain_over_window(state, ticks):
    """Energy the actual player loses while the world advances `ticks` rounds from
    `state`. The player object is already at its position in `state`."""
    clone = state.clone()
    e0 = clone.energy
    for _ in range(ticks):
        enemies.step(clone)
    return max(0, e0 - clone.energy)
