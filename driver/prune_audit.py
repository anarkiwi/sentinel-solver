#!/usr/bin/env python3
"""Which search forks an entry-time sight table could prune without changing the game.

Plays a board's search game recording every fork (parent, move, frames, alive) and
every beam selection. A rule pruning only forks that are never selected leaves every
beam, so the game, unchanged; each rule is scored on forks and frames saved.
"""

import argparse
import json
import os

from tqdm import tqdm

from driver import search_player as sp
from driver.emu import EmuMon
from sentinel import emu
from sentinel.sight_table import SightTable, enemies

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seen(table, st, move):
    """``[fresh, n seeing, n seeing fully]`` of a hop's landing under ``table``."""
    name, tile, k = move
    if name != "hop":
        return None
    if not table.fresh(st, tile):
        return [False, 0, 0]
    got = table.seen(tile, k)
    if got is None:
        return [True, 0, 0]
    live = enemies(st)
    got = [f for e, f in got if e in live]
    return [True, len(got), sum(got)]


def record(landscape, max_actions=120, **knobs):
    """``(forks, selected keys, won)``: fork rows ``[parent key, move, seen, frames,
    alive]`` and the ``(parent key, move)`` of every node any beam kept."""
    bm = EmuMon(emu.enter(landscape))
    table = SightTable(bytes(bm.m.ram))
    forks, rows, chosen = [], {}, set()
    batches, select = sp.batches, sp.SearchPlayer._select

    def batches_seen(parents, branch):
        work = batches(parents, branch)
        for node, programs in zip(parents, work):
            st = sp._state(node.bm)  # pylint: disable=protected-access
            for move, _program in programs:
                rows[(node.key, move)] = [_seen(table, st, move), None, None]
        return work

    def select_seen(self, kids):
        for kid in kids:
            row = rows.get((kid.parent.key, kid.move))
            if row is not None and row[1] is None:
                row[1], row[2] = kid.facts["frames"], bool(kid.fields()["alive"])
        picked = select(self, kids)
        chosen.update((kid.parent.key, kid.move) for kid in picked)
        return picked

    sp.batches, sp.SearchPlayer._select = batches_seen, select_seen
    try:
        player = sp.SearchPlayer(bm, log=lambda _m: None, **knobs)
        player.progress = False
        won = player.run(max_actions=max_actions)
    finally:
        sp.batches, sp.SearchPlayer._select = batches, select
    for (key, move), (seen, frames, alive) in rows.items():
        forks.append([(key, move) in chosen, move, seen, frames or 0, bool(alive)])
    return forks, won


RULES = {
    "seen fully": lambda s: s is not None and s[0] and s[2] > 0,
    "seen at all": lambda s: s is not None and s[0] and s[1] > 0,
}


def score(forks):
    """Per rule: forks and frames pruned, and pruned forks a beam kept (must be 0)."""
    total = sum(f[3] for f in forks)
    out = {"forks": len(forks), "frames": total}
    for name, rule in RULES.items():
        hit = [f for f in forks if rule(f[2])]
        out[name] = {
            "forks": len(hit),
            "frames": sum(f[3] for f in hit),
            "kept_by_beam": sum(f[0] for f in hit),
            "alive": sum(f[4] for f in hit),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", type=int)
    parser.add_argument("--preset", choices=sorted(sp.PRESETS), default="wide")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "out", "logs", "prune"))
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    with tqdm(total=1, desc=f"ls{args.landscape}") as progress:
        forks, won = record(args.landscape, **sp.PRESETS[args.preset])
        progress.update(1)
    result = {"landscape": args.landscape, "won": bool(won), **score(forks)}
    path = os.path.join(args.out_dir, f"prune_{args.landscape:04d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({**result, "fork_rows": forks}, fh, default=str)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
