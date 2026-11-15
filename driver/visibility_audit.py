#!/usr/bin/env python3
"""Whether a visibility table built at entry still answers a solved game's sight queries.

On every tile whose object stack equals entry's, the ROM's enemy sight check and the
player's sight of the tile are asked on the RAM after each search-player move and on the
entry RAM with the observers moved to where they now stand.
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from driver.emu import EmuMon
from driver.search_player import PRESETS, SearchPlayer
from sentinel import emu, exposure_jit, memmap as mm, terrain, threat
from sentinel.sight_table import enemies, stack
from sentinel.state import State

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILES = 31
OBSERVER_ROWS = (
    mm.OBJECTS_X,
    mm.OBJECTS_Y,
    mm.OBJECTS_Z_HEIGHT,
    mm.OBJECTS_Z_FRACTION,
    mm.OBJECTS_H_ANGLE,
    mm.OBJECTS_V_ANGLE,
    mm.OBJECTS_FLAGS,
    mm.OBJECTS_TYPE,
)
COUNTS = ("changed", "unchanged", "enemy", "enemy_diff", "player", "player_diff")


def snapshots(landscape, max_actions=120, **knobs):
    """``(entry RAM, [RAM after each move], won)`` of the search player's game."""
    bm = EmuMon(emu.enter(landscape))
    player = SearchPlayer(bm, log=lambda _m: None, **knobs)
    rams, execute = [], player._execute  # pylint: disable=protected-access

    def keep(planned):
        node = execute(planned)
        rams.append(bytes(bm.m.ram))
        return node

    entry = bytes(bm.m.ram)
    player._execute = keep  # pylint: disable=protected-access
    won = player.run(max_actions=max_actions)
    return entry, rams, won


def seen_by(st, tile, slot, watchers):
    """``{(enemy, full)}`` over ``watchers`` with sight of a phantom robot on ``tile``,
    or None where nothing can stand there."""
    old = terrain.tile_byte(st, *tile)
    if not threat._place_phantom(st, tile, slot):  # pylint: disable=protected-access
        return None
    rows = exposure_jit.exposures_of(np.frombuffer(st.mem, dtype=np.uint8), slot)
    threat._restore_tile(st, tile, old, slot)  # pylint: disable=protected-access
    return {(int(e), bool(f)) for e, _a, f in rows.tolist() if int(e) in watchers}


def observed(entry, now):
    """``entry``'s RAM with every observer of ``now`` (enemies, player) where it stands."""
    st = State.from_mem(entry)
    for s in enemies(now) | {now.player}:
        for row in OBSERVER_ROWS:
            st.mem[row + s] = now.mem[row + s]
    st.player = now.player
    return st


def compare(entry, ram):
    """Counts of one move's RAM against the entry table, and the tiles that disagree."""
    st0, now = State.from_mem(entry), State.from_mem(ram)
    then = observed(entry, now)
    watchers = enemies(now) & enemies(st0)
    slot = sorted(set(now.free_slots()) & set(then.free_slots()))[-1]
    me = now.tile_of(now.player)
    tally = dict.fromkeys(COUNTS, 0)
    tally["diff_tiles"] = []
    for x in range(TILES):
        for y in range(TILES):
            if stack(st0, x, y) != stack(now, x, y):
                tally["changed"] += 1
                continue
            tally["unchanged"] += 1
            a = seen_by(now, (x, y), slot, watchers)
            b = seen_by(then, (x, y), slot, watchers)
            differs = a != b
            if a is not None:
                tally["enemy"] += 1
                tally["enemy_diff"] += differs
            if (x, y) != me:
                pa = threat.player_sees_tile(now, (x, y), now.player)
                pb = threat.player_sees_tile(then, (x, y), now.player)
                tally["player"] += 1
                tally["player_diff"] += pa != pb
                differs |= pa != pb
            if differs:
                tally["diff_tiles"].append((x, y))
    return tally


def audit(landscape, knobs, out_dir):
    """Per-move tallies of ``landscape``; the game's RAM is cached under ``out_dir``."""
    path = os.path.join(out_dir, f"snap_{landscape:04d}.npz")
    if os.path.exists(path):
        z = np.load(path)
        entry, rams, won = bytes(z["entry"]), [bytes(r) for r in z["rams"]], z["won"]
    else:
        entry, rams, won = snapshots(landscape, **knobs)
        np.savez_compressed(
            path,
            entry=np.frombuffer(entry, np.uint8),
            rams=np.array([np.frombuffer(r, np.uint8) for r in rams]),
            won=won,
        )
    rows = [compare(entry, r) for r in tqdm(rams, desc=f"ls{landscape} moves")]
    return {"landscape": landscape, "won": bool(won), "moves": rows}


def summary(result):
    """One line per board: tiles changed, and the table's disagreements."""
    m = result["moves"]
    tot = {k: sum(r[k] for r in m) for k in COUNTS}
    return (
        f"ls{result['landscape']:<5d} won={result['won']!s:5s} moves={len(m):3d} "
        f"changed tiles end={m[-1]['changed'] if m else 0:3d} "
        f"max={max((r['changed'] for r in m), default=0):3d}/{TILES * TILES} | "
        f"enemy sight diff {tot['enemy_diff']}/{tot['enemy']} | "
        f"player sight diff {tot['player_diff']}/{tot['player']}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boards", type=int, nargs="+")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="wide")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "out", "logs", "vis"))
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for board in tqdm(args.boards, desc="boards"):
        results.append(audit(board, PRESETS[args.preset], args.out_dir))
        print(summary(results[-1]), flush=True)
    with open(
        os.path.join(args.out_dir, "visibility_audit.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(results, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
