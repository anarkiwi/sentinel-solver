"""Keyboard-aim geometry: how many keystrokes (and thus game rounds) it costs to
pan the view from one heading to another, or to aim at a tile.

The Sentinel is driven by panning a view on a fixed keyboard lattice (verified in
scripts/kbd_aim.py against the ROM):

  * bearing ``h_angle`` moves in :data:`AZIMUTH_STEP` (8) unit steps -- D pans +8,
    S pans -8, wrapping mod 256 (the u-turn key is an EOR $80, i.e. +16 steps);
  * pitch ``v_angle`` moves in :data:`PITCH_STEP` (4) unit steps inside the clamp
    band -- L pitches +4, COMMA -4.

So the number of *keystrokes* to reach a target heading is the lattice distance
``|Delta h| / 8 + |Delta v| / 4``.  Each keystroke is consumed by one gated input
scan and its pan animates over a fixed number of game rounds, so keystrokes convert
to elapsed enemy rounds by a single scalar (:data:`ROUNDS_PER_PAN_STEP`, calibrated
in the planner) -- which is what lets a strategy search forecast how far the enemies
rotate, and how much they drain, while the player aims a move.

This module is pure geometry (no ROM, no State mutation) so it is cheap enough to
call for every candidate in a lookahead and unit-testable on its own.
"""

import math

AZIMUTH_STEP = 8  # h_angle keyboard step (D/S), wraps mod 256
PITCH_STEP = 4  # v_angle keyboard step (L/COMMA), clamped band
UTURN_STEP = (
    16  # a U-turn (EOR $80) flips 128 units = 16 lattice steps in ONE keystroke
)


def bearing_to(ex, ey, tx, ty):
    """The ``h_angle`` (0..255) an observer at tile (ex, ey) faces to look toward
    tile (tx, ty), on the game's 256-unit compass (mirrors the analytic estimate
    scripts/kbd_aim.py snaps the keyboard grid around, $1C10 vector math). Returns
    None when the tiles coincide (no bearing)."""
    dx, dy = tx - ex, ty - ey
    if dx == 0 and dy == 0:
        return None
    return int(round(math.atan2(dy, dx) / (2 * math.pi) * 256)) & 0xFF


def angle_dist(a, b):
    """Shortest distance between two 256-unit compass angles, 0..128 (i.e. the
    ``abs(((b - a) + 128) % 256 - 128)`` idiom, in one place)."""
    return abs(((b - a) + 128) % 256 - 128)


def h_steps(h0, h1):
    """Keystrokes to pan the bearing from ``h0`` to ``h1``: the shortest distance
    around the mod-256 circle, in 8-unit lattice steps."""
    return angle_dist(h0, h1) // AZIMUTH_STEP


def v_steps(v0, v1):
    """Keystrokes to pitch from ``v0`` to ``v1`` in 4-unit steps.  The keyboard pitch
    band ($CD..$35) is contiguous THROUGH the $FF->$00 wrap (L advances $FD->$01), and
    is only ~104 units wide, so the in-band keyboard distance is exactly the shortest
    circular distance -- a plain ``abs(v0 - v1)`` over-counts a wrap-crossing tilt
    (e.g. $F5 -> $05 is 4 steps, not 60)."""
    return angle_dist(v0, v1) // PITCH_STEP


def h_press_count(h0, h1):
    """Minimal keyboard bearing plan from ``h0`` to ``h1`` on the +-8 lattice, as
    ``(n_uturn, n_step)``: ``n_uturn`` is 0 or 1 U-turn presses (each an EOR $80 = +128,
    i.e. +16 lattice steps in ONE keystroke, $1B2F) and ``n_step`` the remaining +-8
    presses.  The U-turn is taken only when it STRICTLY lowers the total keystroke count --
    a target more than half a turn away -- since a U-turn plus a short correction beats
    stepping most of the way round: direct ``d`` presses vs ``1 + (16 - d)``, so the
    crossover is ``d >= 9`` (a bearing >= 72 units, past which each avoided +-8 press also
    avoids a full pan scroll, cutting real aiming time)."""
    d = h_steps(h0, h1)
    if 1 + (UTURN_STEP - d) < d:
        return (1, UTURN_STEP - d)
    return (0, d)


def bearing_rounds(h0, h1, rounds_per_step, rounds_per_uturn):
    """Enemy rounds to pan the bearing ``h0 -> h1`` using the minimal U-turn-aware key
    plan (:func:`h_press_count`).  Keeps the planner's move cost consistent with what the
    live driver actually keys: a far-bearing swing costs one U-turn + a short correction,
    not up to sixteen +-8 pans."""
    nu, ns = h_press_count(h0, h1)
    return nu * rounds_per_uturn + ns * rounds_per_step


def pan_steps(h0, v0, h1, v1):
    """Total keystrokes to pan the view from heading (h0, v0) to (h1, v1): the sum
    of the bearing and pitch lattice distances.  ``None`` coordinates contribute 0
    (an aim that carries no angle for that axis)."""
    n = 0
    if h0 is not None and h1 is not None:
        n += h_steps(h0, h1)
    if v0 is not None and v1 is not None:
        n += v_steps(v0, v1)
    return n
