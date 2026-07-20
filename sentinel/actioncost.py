"""Per-action world-advance cost, in ``enemies.step`` units (one step == one ROM
``update_enemy_cooldowns``/$1317 tick, the quantity that gates enemy drain and
rotation).  Prices an action the same way the *game* advances the enemies while
that action executes.

Every term is the GAME-INTRINSIC cost derived from the ROM, NOT a fitted floor.
The per-verb SETTLE here is dither+redraw only: driver overhead is not game time.

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
    frame, then ``plot_world`` ($2625) re-plots the scene once.
    The stacked-vs-bare create path is byte-identical (< 1 frame difference), so
    there is NO stack surcharge.

  * TRANSFER: moving the eye sets viewpoint-changed ($0C63), so the main loop takes
    the full-redraw path ``play_landscape_loop`` ($3642 -> $357D): the fixed occlusion
    ($245B)/$3700/fill/status foreground, two full ``plot_world`` passes ($35C3/$35C6),
    then ``wait_for_end_of_tune`` ($35D5) for the #$19 transfer tune ($1B82/$AB69) --
    a FIXED 96-frame note-hold run, duration-identical to the #$0 hyperspace tune
    (``projector.TUNE_TRANSFER_FRAMES``) the *hyperspace* path ($217F) waits for.
    Modelled per-scene by ``projector.viewpoint_replot_frames``; the constant here is
    the view-less fallback.

  * AIM is priced by the caller from the keyboard-scroll cadence (a +-8 bearing
    notch animates a 16-step horizontal scroll $10EE, a +-4 pitch notch an 8-step
    vertical scroll $1135, each followed by one ``plot_world``).

Caveat: DITHER_FRAMES is a py65 foreground cycle-count (no raster-IRQ steal), so it
is a ~5-15% lower bound; the tune wait and the pan scroll counts are exact static
loop bounds.  The env overrides below let a VICE-measured frame count refine a ROM
number -- they are ROM measurements, not outcome fits.
"""

import os

from sentinel import projector

# Costs are now in FRAMES (video frames), the unit sentinel.enemies.advance_frames
# consumes: the $130C/$1335 Bresenham (205/256) and the $0C50 1-in-3 gate are applied
# INSIDE advance_frame per frame, so the cost model must NOT pre-scale by them.  Kept as
# FRAME_TICKS=1.0 (env-overridable) so every settle/pan term below reads as a frame count.
FRAME_TICKS = float(os.environ.get("FRAME_TICKS", "1.0"))

# --- ROM-cited per-phase frame counts ------------------------------------------
# Object dither animation loop ($1FA4 create / $86A5 absorb): 977904 cycles at the
# 19656-cycle PAL frame.
DITHER_FRAMES = float(os.environ.get("DITHER_FRAMES", str(977904.0 / 19656.0)))
# Transfer settle ($357D): fixed #$19 tune wait (96) + fixed $245B/$3700/fill/status foreground (~176) + 2x plot_world; live 259-460f, modelled per-scene by projector.viewpoint_replot_frames (docs/render_cost.md). This constant is the view-less fallback (tune+fixed base only).
VIEWPOINT_REPLOT_FRAMES = float(
    os.environ.get(
        "VIEWPOINT_REPLOT_FRAMES",
        str(projector.TUNE_TRANSFER_FRAMES + projector.SETTLE_FIXED_FRAMES),
    )
)
# Post-create/absorb scene replot after the dither loop; VICE ~44.
POST_ACTION_REPLOT_FRAMES = float(os.environ.get("POST_ACTION_REPLOT_FRAMES", "44"))

# --- game-intrinsic per-verb settle (ticks), derived from the frame counts ------
# create/absorb: dither loop + one incremental replot; transfer: viewpoint full-redraw.
SETTLE = {
    "absorb": FRAME_TICKS * (DITHER_FRAMES + POST_ACTION_REPLOT_FRAMES),
    "create": FRAME_TICKS * (DITHER_FRAMES + POST_ACTION_REPLOT_FRAMES),
    "transfer": FRAME_TICKS * VIEWPOINT_REPLOT_FRAMES,
}
