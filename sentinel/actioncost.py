"""Per-action world-advance cost, in ``enemies.step`` units (one step == one ROM
``update_enemy_cooldowns``/$1317 tick, the quantity that gates enemy drain and
rotation).  Shared by the simulated runner's world advance and the planner's
forecast so both price an action the same way the live driver's game actually
advances the enemies while that action executes.

Every term is grounded in the ROM's own cadence (see re-sentinel disasm), not a
per-action-type fitted constant:

  * A game frame ticks the cooldowns once (raster IRQ $9663 -> $130C); the $1335
    Bresenham divider ($130C: ``+= $CD``) means the real $1317 decrement -- the
    unit ``enemies.step`` models -- runs ``205/256`` of the frames.  So
    ``steps == round(0.80 * frames)`` and every constant below is quoted directly
    in steps.

  * AIM is priced by the caller from the keyboard-scroll cadence
    (``climb_search._pan_rounds``: a +-8 bearing notch animates a 16-step
    horizontal scroll $10EE, a +-4 pitch notch an 8-step vertical scroll $1135).

  * REDRAW: after any viewpoint change the ROM re-plots the scene in a single
    blocking ``plot_world`` ($2625) pass whose cost is O(sum of the polygon EDGES
    of the objects in view); the raster IRQ keeps ticking cooldowns while it runs.
    Measured against the real renderer (py65) this is small -- a whole 19-object
    view adds < 2 frames -- so it is a minor, scene-scaling correction, but it is
    the term that makes a dense view cost more than a bare one.

  * SETTLE: the fixed floor an action spends after firing before the live driver
    reads back the result.  Its ROM-mechanism parts are the object dither
    animation (25 steps, $1FA4) for create/absorb and the hyperspace tune wait
    (96 frames, $35D5) for transfer, plus the driver's post-fire verify poll while
    the game idles.  These floors dominate the per-action cost (the aim and redraw
    terms are secondary); the split is documented per verb below.

All constants are env-overridable so a diverging live run can be recalibrated from
its own telemetry without a code change.
"""

import os

from sentinel import memmap as mm

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

# Redraw steps per rasterised edge (frames/edge from the py65 plot_world
# measurement, * 0.80 frame->step): small, so the scene term is a minor correction.
STEPS_PER_EDGE = float(os.environ.get("STEPS_PER_EDGE", "0.02"))

# Half-width of the on-screen field of view in compass units: the ROM reloads the
# enemy/screen FOV width to $14 == 20 units each scan ($16F2), i.e. +-10 units.
FOV_HALF = 10

# Per-verb SETTLE floor (steps) an action costs after firing, before the live
# driver confirms it.  Decomposition (steps ~= 0.80 * frames):
#   verify poll (driver idles the game reading back ~2 full states)  ~110
#   + create/absorb: object dither animation (25 frames $1FA4)        ~20
#   + create:        taller/new-object full replot on top of dither   ~90
#   + transfer:      hyperspace tune wait (96 frames $35D5) + full     ~190
#                    viewpoint redraw ($35C3 plot_world x2)
SETTLE = {
    "absorb": float(os.environ.get("SETTLE_ABSORB", "190")),
    "create": float(os.environ.get("SETTLE_CREATE", "290")),
    "transfer": float(os.environ.get("SETTLE_TRANSFER", "300")),
}

# Extra rounds a create costs when it STACKS on an existing object (a synthoid built on
# a boulder, the ls0 foothold pattern). The new object sits a tile-height higher, so the
# ROM re-plots a taller stack -> more frames. Measured live: a synthoid-on-boulder create
# costs ~285-300 more than the bare boulder at the same tile (L10->L11 254->557,
# L14->L15 270->555), yet the flat SETTLE priced them equally -- the dominant source of
# the sim's ~793-round cumulative UNDER-count that hid the far-corner meanie.
STACK_CREATE = float(os.environ.get("STACK_CREATE", "285"))


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
    per-verb SETTLE floor, plus the scene redraw term, plus STACK_CREATE when a create
    lands on an already-occupied tile (taller stack -> bigger redraw). The aim itself is
    priced separately by the caller (``climb_search._pan_rounds``)."""
    settle = SETTLE.get(verb, SETTLE["absorb"])
    if verb == "create" and stacked:
        settle += STACK_CREATE
    return settle + STEPS_PER_EDGE * visible_edges(mem, view)


def is_stacked(mem, tile):
    """True if `tile` already holds a live object (so a create there STACKS)."""
    b = mem[mm.TILES_TABLE + mm.tidx(tile[0], tile[1])]
    return b >= mm.OBJECT_TILE
