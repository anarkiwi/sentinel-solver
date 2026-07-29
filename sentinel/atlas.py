"""Atlas layer 2: per-landscape metrics, computed from a cached board.

A metric is one registered function of a :class:`Board` -- a :mod:`sentinel.statecache`
state plus its derived arrays -- so adding a metric is adding a function and re-running,
with no board regenerated.  CLI: ``python -m sentinel.atlas --start 0 --stop 64``.
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sentinel import landscape, memmap as mm, statecache

_XX, _YY = np.meshgrid(np.arange(mm.N), np.arange(mm.N), indexing="ij")
_TIDX = mm.TILES_TABLE + (_XX & 3) * 256 + ((_XX >> 2) & 7) * 32 + _YY

METRICS = {}


def metric(name):
    """Register ``name`` as a metric computed by the decorated ``fn(board)``."""

    def wrap(fn):
        METRICS[name] = fn
        return fn

    return wrap


class Board:
    """A cached state plus the whole-board arrays metrics read.

    ``heights``/``slopes`` are the 32x32 resolved ground fields -- terrain nibbles, or
    the bottommost object's height and slope 0 on an object tile, as
    :func:`sentinel.terrain.resolve_ground` defines them.
    """

    __slots__ = (
        "code",
        "state",
        "mem",
        "otype",
        "ox",
        "oy",
        "oz",
        "slots",
        "enemy_slots",
        "heights",
        "slopes",
    )

    def __init__(self, code, state):
        self.code = code
        self.state = state
        mem = np.frombuffer(bytes(state.mem), dtype=np.uint8)
        self.mem = mem
        flags = mem[mm.OBJECTS_FLAGS : mm.OBJECTS_FLAGS + mm.NUM_SLOTS]
        self.otype = mem[mm.OBJECTS_TYPE : mm.OBJECTS_TYPE + mm.NUM_SLOTS]
        self.ox = mem[mm.OBJECTS_X : mm.OBJECTS_X + mm.NUM_SLOTS]
        self.oy = mem[mm.OBJECTS_Y : mm.OBJECTS_Y + mm.NUM_SLOTS]
        self.oz = mem[mm.OBJECTS_Z_HEIGHT : mm.OBJECTS_Z_HEIGHT + mm.NUM_SLOTS]
        live = flags < 0x80
        self.slots = np.flatnonzero(live)
        self.enemy_slots = np.flatnonzero(
            live & np.isin(self.otype, list(mm.ENEMY_TYPES))
        )
        index = np.arange(mm.NUM_SLOTS, dtype=np.uint8)
        bottom = np.where((flags & 0xC0) == 0x40, flags & 0x3F, index)
        for _ in range(6):  # doubling: resolves a stack up to 64 deep
            bottom = bottom[bottom]
        tiles = mem[_TIDX]
        stacked = tiles >= mm.OBJECT_TILE
        self.heights = np.where(
            stacked, self.oz[bottom[tiles & 0x3F]], tiles >> 4
        ).astype(np.int16)
        self.slopes = np.where(stacked, 0, tiles & 0x0F)

    @property
    def player(self):
        return int(self.state.player)


# ---- metrics ---------------------------------------------------------------
@metric("seed")
def _seed(board):
    """The ROM PRNG seed the typed code maps to (its digits read as hex)."""
    return landscape.seed_for(board.code)


@metric("enemies")
def _enemies(board):
    """Occupied enemy slots: the Sentinel plus its sentries."""
    return int(board.enemy_slots.size)


@metric("enemy_list")
def _enemy_list(board):
    """Per enemy: slot, type name, tile (x, y) and height."""
    return [
        {
            "slot": int(s),
            "type": mm.TYPES[int(board.otype[s])],
            "x": int(board.ox[s]),
            "y": int(board.oy[s]),
            "z": int(board.oz[s]),
        }
        for s in board.enemy_slots
    ]


@metric("landscape_energy")
def _landscape_energy(board):
    """Absorbable energy standing on the board: ``mm.ENERGY_IN_OBJECTS`` summed over
    every occupied slot except the player's own robot.  The player's starting energy is
    NOT included (that is ``start_energy``), so this is the pool the board offers."""
    table = np.zeros(len(mm.TYPES), dtype=np.int32)
    for otype, value in mm.ENERGY_IN_OBJECTS.items():
        table[otype] = value
    slots = board.slots[board.slots != board.player]
    return int(table[board.otype[slots]].sum())


@metric("roughness")
def _roughness(board):
    """Mean absolute height step between neighbouring tiles, over both axes."""
    grid = board.heights
    dx = np.abs(np.diff(grid, axis=0)).mean()
    dy = np.abs(np.diff(grid, axis=1)).mean()
    return round(float((dx + dy) / 2.0), 4)


@metric("relief")
def _relief(board):
    """Highest ground height minus lowest."""
    return int(board.heights.max() - board.heights.min())


@metric("mean_z")
def _mean_z(board):
    """Mean ground height over the 32x32 tiles."""
    return round(float(board.heights.mean()), 3)


@metric("flat_tiles")
def _flat_tiles(board):
    """Tiles of slope 0 -- the only ones an object or the player can stand on."""
    return int(np.count_nonzero(board.slopes == 0))


@metric("start_tile")
def _start_tile(board):
    """The player robot's starting tile (x, y)."""
    return list(board.state.player_xy())


@metric("start_z")
def _start_z(board):
    """The player robot's starting integer height."""
    return int(board.oz[board.player])


@metric("start_eye")
def _start_eye(board):
    """The player's starting eye height, height + fraction/256."""
    return round(board.state.eye_z(), 3)


@metric("start_energy")
def _start_energy(board):
    """The energy the player is given at entry ($0C0A)."""
    return int(board.state.energy)


# ---- the atlas -------------------------------------------------------------
def board_for(code, regen=False):
    """``(Board, cache_hit)`` for landscape ``code``."""
    state, hit = statecache.state_for(code, regen)
    return Board(statecache.valid_code(code), state), hit


def measure(board, names=None):
    """The metric row for an already-built :class:`Board`."""
    row = {"code": board.code}
    for name in names or METRICS:
        row[name] = METRICS[name](board)
    return row


def row_for(code, names=None, regen=False):
    """The metric row for landscape ``code``, from cache unless ``regen``."""
    return measure(board_for(code, regen)[0], names)


def _chunk(args):
    codes, names, regen = args
    return [row_for(code, names, regen) for code in codes]


def scan(codes, names=None, regen=False, jobs=None):
    """Metric rows for ``codes``, chunked across worker processes."""
    codes = [statecache.valid_code(code) for code in codes]
    jobs = min(jobs or os.cpu_count() or 1, max(1, len(codes)))
    if jobs == 1:
        return _chunk((codes, names, regen))
    size = max(1, math.ceil(len(codes) / (jobs * 4)))
    parts = [(codes[i : i + size], names, regen) for i in range(0, len(codes), size)]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for part in pool.map(_chunk, parts):
            rows.extend(part)
    return rows


def distance(ref, other):
    """How unlike ``other`` is to reference ``ref``, on the features shaping a climb."""
    return sum(
        abs(ref[key] - other[key]) / max(abs(ref[key]), floor)
        for key, floor in (
            ("roughness", 1e-6),
            ("relief", 1),
            ("flat_tiles", 1),
            ("mean_z", 1e-6),
        )
    )


# ---- CLI -------------------------------------------------------------------
def _table(rows):
    """Scalar metrics as aligned columns; list/dict metrics on a continuation line."""
    if not rows:
        return ""
    keys = list(rows[0])
    flat = [k for k in keys if not isinstance(rows[0][k], (list, dict))]
    deep = [k for k in keys if k not in flat]
    cells = [[f"{row[k]}" for k in flat] for row in rows]
    width = [max(len(k), *(len(c[i]) for c in cells)) for i, k in enumerate(flat)]
    out = ["  ".join(k.rjust(width[i]) for i, k in enumerate(flat))]
    for row, cell in zip(rows, cells):
        out.append("  ".join(value.rjust(width[i]) for i, value in enumerate(cell)))
        for key in deep:
            out.append(f"    {key}: {json.dumps(row[key], separators=(',', ':'))}")
    return "\n".join(out)


def _codes(args):
    if args.codes:
        return [int(part) for part in args.codes.replace(",", " ").split()]
    start = args.start or 0
    stop = args.stop if args.stop is not None else start + 1
    return list(range(start, stop))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, help=f"first code (>= {statecache.MIN_CODE})")
    ap.add_argument(
        "--stop", type=int, help=f"one past the last (<= {statecache.MAX_CODE + 1})"
    )
    ap.add_argument("--codes", help="explicit codes, comma or space separated")
    ap.add_argument("--metrics", help=f"comma separated subset of: {','.join(METRICS)}")
    ap.add_argument("--regen", action="store_true", help="ignore the state cache")
    ap.add_argument("--format", choices=("table", "json", "jsonl"), default="table")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--like", type=int, help="rank the range by likeness to this code")
    ap.add_argument("--top", type=int, default=8, help="rows to keep under --like")
    ap.add_argument(
        "--any-enemies",
        action="store_true",
        help="under --like, do not require the same enemy count",
    )
    args = ap.parse_args(argv)

    names = args.metrics.split(",") if args.metrics else None
    if names:
        unknown = [name for name in names if name not in METRICS]
        if unknown:
            ap.error(f"unknown metric(s): {','.join(unknown)}")
    codes = _codes(args)
    if not codes:
        ap.error("nothing to do: pass --codes or --start/--stop")
    rows = scan(codes, names, args.regen, args.jobs)

    if args.like is not None:
        ref = row_for(args.like, names)
        rows = [r for r in rows if r["code"] != ref["code"]]
        if not args.any_enemies:
            rows = [r for r in rows if r.get("enemies") == ref.get("enemies")]
        for r in rows:
            r["distance"] = round(distance(ref, r), 4)
        rows.sort(key=lambda r: r["distance"])
        rows = [dict(ref, distance=0.0)] + rows[: args.top]

    if args.format == "json":
        print(json.dumps({"signature": statecache.signature(), "rows": rows}))
    elif args.format == "jsonl":
        for r in rows:
            print(json.dumps(r))
    else:
        print(_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
