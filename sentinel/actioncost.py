"""Per-action world-advance cost, in ``enemies.step`` units (one step == one ROM
``update_enemy_cooldowns``/$1317 tick, the quantity that gates enemy drain and
rotation).  Prices an action the same way the *game* advances the enemies while
that action executes.

Every term is the GAME-INTRINSIC cost derived from the ROM, NOT a fitted floor.
The per-verb SETTLE here is dither+redraw only; the old floors (create 290 / absorb
190 / transfer 300, STACK_CREATE 285) inflated that with DRIVER overhead (the aim
ring-SEARCH ``kbd_aim.fine_to_tile`` + read-back idle), eliminated by direct aiming.

The aim DWELL (coarse body-pan scroll + fine sights-cursor travel), also game time,
is priced separately from the keyboard-scroll cadence ($10EE/$1135 notches; cursor
from the $134C reset centre at FRAME_TICKS/unit), not here.

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

  * TRANSFER: moving the eye sets viewpoint-changed ($0C63), so the main loop takes
    the full-redraw path ``play_landscape_loop`` ($3642 -> $357D): the fixed occlusion
    ($245B)/$3700/fill/status foreground, two full ``plot_world`` passes ($35C3/$35C6),
    then ``wait_for_end_of_tune`` ($35D5) for the #$19 transfer tune ($1B82/$AB69) --
    a FIXED 96-frame note-hold run, duration-identical to the #$0 hyperspace tune
    (``TUNE_FRAMES``) the *hyperspace* path ($217F) waits for.  Modelled per-scene by
    ``projector.viewpoint_replot_frames``; this constant is the view-less fallback.

  * AIM is priced by the caller from the keyboard-scroll cadence (a +-8 bearing
    notch animates a 16-step horizontal scroll $10EE, a +-4 pitch notch an 8-step
    vertical scroll $1135, each followed by one ``plot_world``).

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

from sentinel import memmap as mm, projector
from sentinel.state import State

# Costs are now in FRAMES (video frames), the unit sentinel.enemies.advance_frames
# consumes: the $130C/$1335 Bresenham (205/256) and the $0C50 1-in-3 gate are applied
# INSIDE advance_frame per frame, so the cost model must NOT pre-scale by them.  Kept as
# FRAME_TICKS=1.0 (env-overridable) so every settle/pan term below reads as a frame count.
FRAME_TICKS = float(os.environ.get("FRAME_TICKS", "1.0"))

# --- ROM-cited per-phase frame counts ------------------------------------------
# Object dither animation loop ($1FA4 create / $86A5 absorb): 977904 cycles at the
# 19656-cycle PAL frame.
DITHER_FRAMES = float(os.environ.get("DITHER_FRAMES", str(977904.0 / 19656.0)))
# Hyperspace #$0 tune wait ($AB69), a static 96-frame countdown ($217F path only).
TUNE_FRAMES = float(os.environ.get("TUNE_FRAMES", "96"))
# One blocking plot_world ($2625) terrain-dominant redraw pass (py65 ~5 frames).
REDRAW_FRAMES = float(os.environ.get("REDRAW_FRAMES", "5"))
# Transfer settle ($357D): fixed #$19 tune wait (96) + fixed $245B/$3700/fill/status foreground (~176) + 2x plot_world; live 259-460f, modelled per-scene by projector.viewpoint_replot_frames (docs/render_cost.md). This constant is the view-less fallback (tune+fixed base only).
VIEWPOINT_REPLOT_FRAMES = float(
    os.environ.get(
        "VIEWPOINT_REPLOT_FRAMES",
        str(projector.TUNE_TRANSFER_FRAMES + projector.SETTLE_FIXED_FRAMES),
    )
)
# Post-create/absorb scene replot after the dither loop; VICE ~44 (vs incremental REDRAW 5).
POST_ACTION_REPLOT_FRAMES = float(os.environ.get("POST_ACTION_REPLOT_FRAMES", "44"))

# Redraw ticks per rasterised edge (frames/edge from the py65 plot_world
# measurement, * FRAME_TICKS): a minor scene-scaling correction. Validated.
STEPS_PER_EDGE = float(os.environ.get("STEPS_PER_EDGE", "0.02"))

# Half-width of the on-screen field of view in compass units: the ROM reloads the
# enemy/screen FOV width to $14 == 20 units each scan ($16F2), i.e. +-10 units.
FOV_HALF = 10

# --- game-intrinsic per-verb settle (ticks), derived from the frame counts ------
# create/absorb: dither loop + one incremental replot; transfer: viewpoint full-redraw.
SETTLE = {
    "absorb": FRAME_TICKS * (DITHER_FRAMES + POST_ACTION_REPLOT_FRAMES),
    "create": FRAME_TICKS * (DITHER_FRAMES + POST_ACTION_REPLOT_FRAMES),
    "transfer": FRAME_TICKS * VIEWPOINT_REPLOT_FRAMES,
}

# The ROM stacked-create dither is byte-identical to the bare-create dither: the loop
# frame count $2099 is loaded #$19 (25) unconditionally in update_object_on_screen
# ($1FA4), independent of stacking; put_object_in_tile ($1F16) differs only by the
# handful of instructions that set the on-object $40 flag and stacked-z ($1F3A-$1F63)
# before both paths converge at set_object_z ($1F76) -- < 1 frame. So NO tick surcharge
# (the live +285 was driver aim-search idle, not the game). Kept as a symbol for callers
# but ROM-zero; env-overridable only to reintroduce a VICE-measured surcharge if found.
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
    priced separately from the keyboard-scroll cadence."""
    if verb == "transfer" and view and view.get("h_angle") is not None:
        # Area-based plot_world replot cost, scene-dependent ~306-420 frames.
        settle = FRAME_TICKS * projector.viewpoint_replot_frames(State(mem), view)
    else:
        settle = SETTLE.get(verb, SETTLE["absorb"])
    if verb == "create" and stacked:
        settle += STACK_CREATE
    return settle + STEPS_PER_EDGE * visible_edges(mem, view)


def is_stacked(mem, tile):
    """True if `tile` already holds a live object (so a create there STACKS)."""
    b = mem[mm.TILES_TABLE + mm.tidx(tile[0], tile[1])]
    return b >= mm.OBJECT_TILE
