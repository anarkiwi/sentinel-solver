"""Frame-unit glue for per-action world advance.

Per-scene settle pricing lives in :mod:`sentinel.settlecost` (the $12D0/$1F9F and
$357D/$35D5 laws, validated against tests/fixtures/settle_oracle_ls42.json); the
``SETTLE`` dict is only the tile-less fallback, that fixture's per-verb means.
"""

import os

# One frame == one $9663 -> $130C cooldown tick; the $1335 Bresenham (205/256) and the $0C50 1-in-3 gate are applied inside advance_frame, so costs stay in frames.
FRAME_TICKS = float(os.environ.get("FRAME_TICKS", "1.0"))

SETTLE = {
    "absorb": 86.0,
    "create": 112.0,
    "transfer": 330.0,
}
