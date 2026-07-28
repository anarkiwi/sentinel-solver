"""Measure landscapes so one can be matched to another: enemies, roughness, footing.

A board's difficulty is mostly its terrain statistics plus how many cones watch it, so
this scores every landscape number on the few features that decide a climb and reports
the nearest matches to a reference board.
"""

import argparse
import json

import numpy as np

from sentinel import enemies as enemy_mod, landscape, memmap as mm, terrain
from sentinel.game import Game


def heights(state):
    """The 32x32 height field, as the ROM's tile-byte high nibble."""
    grid = np.zeros((mm.N, mm.N), dtype=np.int16)
    for x in range(mm.N):
        for y in range(mm.N):
            grid[x, y] = terrain.tile_byte(state, x, y) >> 4
    return grid


def features(code):
    """Enemy count and terrain shape for the landscape a player types as ``code``."""
    state = Game.typed(code).state
    grid = heights(state)
    dx = np.abs(np.diff(grid, axis=0)).mean()
    dy = np.abs(np.diff(grid, axis=1)).mean()
    flat = sum(
        1
        for x in range(mm.N)
        for y in range(mm.N)
        if (terrain.tile_byte(state, x, y) & 0x0F) == 0
    )
    foes = enemy_mod.enemy_slots(state)
    return {
        "code": code,
        "seed": landscape.seed_for(code),
        "enemies": len(foes),
        "roughness": round(float((dx + dy) / 2.0), 4),
        "relief": int(grid.max() - grid.min()),
        "mean_z": round(float(grid.mean()), 3),
        "flat_tiles": flat,
        "start_eye": round(state.eye_z(), 3),
        "start_energy": state.energy,
    }


def distance(a, b):
    """How unlike ``b`` is to reference ``a``, on the features that shape a climb."""
    return (
        abs(a["roughness"] - b["roughness"]) / max(a["roughness"], 1e-6)
        + abs(a["relief"] - b["relief"]) / max(a["relief"], 1)
        + abs(a["flat_tiles"] - b["flat_tiles"]) / max(a["flat_tiles"], 1)
        + abs(a["mean_z"] - b["mean_z"]) / max(a["mean_z"], 1e-6)
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--like", type=int, default=335, help="reference landscape code")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=400)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--same-enemies", action="store_true", default=True)
    args = ap.parse_args(argv)

    ref = features(args.like)
    print(json.dumps({"reference": ref}))
    rows = []
    for code in range(args.start, args.stop):
        if code == args.like:
            continue
        try:
            row = features(code)
        except Exception:
            continue
        if args.same_enemies and row["enemies"] != ref["enemies"]:
            continue
        row["distance"] = round(distance(ref, row), 4)
        rows.append(row)
    rows.sort(key=lambda r: r["distance"])
    for row in rows[: args.top]:
        print(json.dumps(row))
    print(
        f"scanned {args.stop - args.start}, {len(rows)} with {ref['enemies']} enemies"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
