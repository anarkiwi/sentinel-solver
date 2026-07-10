"""Per-action world-advance cost, in ``enemies.step`` units (one step == one ROM
``update_enemy_cooldowns``/$1317 tick, the quantity that gates enemy drain and
rotation).  Shared by the simulated runner's world advance and the planner's
forecast so both price an action the same way the *game* advances the enemies
while that action executes.

Every term is the GAME-INTRINSIC cost derived from the ROM's own code (see the
re-sentinel C64 disasm), NOT a per-action-type fitted floor.  The earlier model
carried SETTLE floors (create 290 / absorb 190 / transfer 300) and STACK_CREATE
285; a ROM read (2026-07-07) showed those are ~3-7x the game-intrinsic cost and
the excess is a DRIVER artifact -- the live driver's aim ring-search
(``kbd_aim.fine_to_tile`` probes ~24 cursor positions, each a keystroke that
advances the game) plus post-fire read-back idle.  That excess is not a property
of the game, so it is eliminated (by aiming the live driver directly at the
plan's cursor), not fitted here.

Frame -> tick conversion.  A video frame ticks the cooldowns once (raster IRQ
$9663 -> $130C); the $1335 Bresenham divider ($130C: ``+= $CD`` == 205/256) means
the real $1317 decrement -- the unit ``enemies.step`` models -- runs ``205/256``
of the frames.  So ``ticks == FRAME_TICKS * frames`` with ``FRAME_TICKS = 0.80``.

Per-phase frame counts, each cited to the ROM:

  * CREATE / ABSORB: the object dither animation loop ($1FA4 / $86A5) runs
    ``977904`` CPU cycles == ``DITHER_FRAMES`` frames at the ``19656``-cycle PAL
    frame, then ``plot_world`` ($2625) re-plots the scene once (~REDRAW_FRAMES).
    The stacked-vs-bare create path is byte-identical (< 1 frame difference), so
    there is NO stack surcharge.

  * TRANSFER (hyperspace): the tune wait ($AB69) is ``TUNE_FRAMES`` frames, framed
    by two ``plot_world`` redraws ($35C3).

  * AIM is priced by the caller from the keyboard-scroll cadence (a +-8 bearing
    notch animates a 16-step horizontal scroll $10EE, a +-4 pitch notch an 8-step
    vertical scroll $1135, each followed by one ``plot_world``); see
    ``solver.cost.aim_rounds``.

  * REDRAW: ``plot_world`` is a single blocking pass whose cost scales with the
    summed polygon EDGES of the objects in view; the raster IRQ keeps ticking
    cooldowns while it runs.  The per-edge term (STEPS_PER_EDGE) was validated
    against the py65 renderer and is a minor, scene-scaling correction on top of
    the base terrain redraw.

Caveat: DITHER_FRAMES and REDRAW_FRAMES are py65 foreground cycle-counts (no
raster-IRQ steal), so they are ~5-15% lower bounds; TUNE_FRAMES and the pan scroll
counts are exact static loop bounds.  The env overrides below let a VICE-measured
frame count refine a ROM number -- they are ROM measurements, not outcome fits.
"""

import os

from sentinel import memmap as mm

# --- frame -> tick conversion (exact, $130C/$1335 Bresenham 205/256) -----------
FRAME_TICKS = float(os.environ.get("FRAME_TICKS", "0.80"))

# --- ROM-cited per-phase frame counts ------------------------------------------
# Object dither animation loop ($1FA4 create / $86A5 absorb): 977904 cycles at the
# 19656-cycle PAL frame.
DITHER_FRAMES = float(os.environ.get("DITHER_FRAMES", str(977904.0 / 19656.0)))
# Hyperspace tune wait ($AB69), a static 96-frame countdown.
TUNE_FRAMES = float(os.environ.get("TUNE_FRAMES", "96"))
# One blocking plot_world ($2625) terrain-dominant redraw pass (py65 ~5 frames).
REDRAW_FRAMES = float(os.environ.get("REDRAW_FRAMES", "5"))

# Redraw ticks per rasterised edge (frames/edge from the py65 plot_world
# measurement, * FRAME_TICKS): a minor scene-scaling correction. Validated.
STEPS_PER_EDGE = float(os.environ.get("STEPS_PER_EDGE", "0.02"))

# Half-width of the on-screen field of view in compass units: the ROM reloads the
# enemy/screen FOV width to $14 == 20 units each scan ($16F2), i.e. +-10 units.
FOV_HALF = 10

# --- game-intrinsic per-verb settle (ticks), derived from the frame counts ------
# create/absorb: dither loop + one scene replot.
# transfer:      tune wait + two scene replots.
# No fitted floors, no driver read-back idle, no stack surcharge.
SETTLE = {
    "absorb": FRAME_TICKS * (DITHER_FRAMES + REDRAW_FRAMES),
    "create": FRAME_TICKS * (DITHER_FRAMES + REDRAW_FRAMES),
    "transfer": FRAME_TICKS * (TUNE_FRAMES + 2 * REDRAW_FRAMES),
}

# The ROM stacked-create path is byte-identical to the bare-create path (the live
# +285 was driver aim-search idle, not the game). Kept as a symbol for callers but
# ROM-zero. Env-overridable only to reintroduce a measured surcharge if one is ever
# found in VICE.
STACK_CREATE = float(os.environ.get("STACK_CREATE", "0"))

# Per-type polygon EDGE counts rasterised by plot_object ($8533/$8579), read from
# the ROM model tables ($9CA0 vertices / $9CAB polygons / $A1A0 shape): a redraw's
# cost tracks the sum of these over the objects in view.
EDGES = {
    mm.T_ROBOT: 96,
    mm.T_SENTRY: 88,
    mm.T_TREE: 52,
    mm.T_BOULDER: 32,
    mm.T_MEANIE: 81,
    mm.T_SENTINEL: 124,
    6: 40,  # pedestal / platform
}


def _bearing_to(ex, ey, tx, ty):
    """Compass bearing (0..255) from tile (ex,ey) toward (tx,ty); None if same."""
    import math

    if ex == tx and ey == ty:
        return None
    return int(round(math.atan2(ty - ey, tx - ex) / (2 * math.pi) * 256)) & 0xFF


def visible_edges(mem, view):
    """Sum of the polygon EDGE counts of the objects that fall inside the field of
    view aimed by `view` -- the scene-complexity that drives the redraw cost.  A
    coarse frustum test (bearing within FOV_HALF of the view heading) mirrors the
    ROM's on-screen inclusion without a full projection."""
    if not view or view.get("h_angle") is None:
        return 0
    vh = view["h_angle"] & 0xFF
    ps = mem[mm.PLAYER_OBJECT]
    ex, ey = mem[mm.OBJECTS_X + ps], mem[mm.OBJECTS_Y + ps]
    total = 0
    for slot in range(mm.NUM_SLOTS):
        if slot == ps or (mem[mm.OBJECTS_FLAGS + slot] & 0x80):
            continue
        b = _bearing_to(ex, ey, mem[mm.OBJECTS_X + slot], mem[mm.OBJECTS_Y + slot])
        if b is None:
            continue
        if abs(((b - vh) + 128) % 256 - 128) <= FOV_HALF:
            total += EDGES.get(mem[mm.OBJECTS_TYPE + slot], 40)
    return total


def action_rounds(mem, verb, view, stacked=False):
    """Enemy-round (``enemies.step``) cost an action costs AFTER the aim pan: the
    game-intrinsic per-verb settle, plus the scene redraw term.  STACK_CREATE is
    ROM-zero (the stacked-create path is byte-identical) but still added when set,
    for callers that want to model a VICE-measured surcharge.  The aim itself is
    priced separately by the caller (``solver.cost.aim_rounds``)."""
    settle = SETTLE.get(verb, SETTLE["absorb"])
    if verb == "create" and stacked:
        settle += STACK_CREATE
    return settle + STEPS_PER_EDGE * visible_edges(mem, view)


def is_stacked(mem, tile):
    """True if `tile` already holds a live object (so a create there STACKS)."""
    b = mem[mm.TILES_TABLE + mm.tidx(tile[0], tile[1])]
    return b >= mm.OBJECT_TILE
