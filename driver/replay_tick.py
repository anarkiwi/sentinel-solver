#!/usr/bin/env python3
"""Replay live ticks saved by ``driver.play_player --ticks DIR`` on the simulator.

Each saved live image becomes a simulator player (:func:`driver.live_player.load_tick`)
that plays ``--ticks`` decision ticks; ``--jitter F ...`` reports its first action per
enemy-clock offset ``F``; ``--diff LANDSCAPE`` lists bytes differing from ``Game.typed``.
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from driver.live_player import load_tick
from sentinel import enemies
from sentinel.game import Game


def image_diff(mem, landscape):
    """``[(addr, live, typed)]`` for every byte where ``mem`` differs from
    ``Game.typed(landscape)``."""
    ref = np.frombuffer(bytes(Game.typed(landscape).state.mem), dtype=np.uint8)
    got = np.frombuffer(bytes(mem), dtype=np.uint8)
    idx = np.flatnonzero(ref != got)
    return [(int(a), int(got[a]), int(ref[a])) for a in idx]


def replay(path, ticks=1, offset=0, verbose=True):
    """The simulator player that played ``ticks`` decision ticks from ``path`` with
    the enemy clock first advanced ``offset`` frames."""
    player = load_tick(path, verbose=verbose)
    if offset:
        enemies.advance_frames(player.st, offset)
    for _ in range(ticks):
        player._tick()  # pylint: disable=protected-access
    return player


def first_action(job):
    """``(path, offset, (verb, tile) or None)``: the first action taken from ``path``
    after an ``offset``-frame clock shift, within ``ticks`` decision ticks."""
    path, offset, ticks = job
    player = replay(path, 0, offset, verbose=False)
    for _ in range(ticks):
        player._tick()  # pylint: disable=protected-access
        if player.trace:
            _f, verb, tile, *_ = player.trace[0]
            return path, offset, (verb, tuple(int(c) for c in tile))
    return path, offset, None


def jitter(paths, offsets, workers, ticks):
    """``{path: [first action per offset]}`` over a process pool, each replay allowed
    ``ticks`` decision ticks to act."""
    out = {p: [None] * len(offsets) for p in paths}
    jobs = [(p, k, ticks) for p in paths for k in offsets]
    with ProcessPoolExecutor(workers) as pool:
        for path, k, act in tqdm(
            pool.map(first_action, jobs), total=len(jobs), desc="jitter"
        ):
            out[path][offsets.index(k)] = act
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticks_dir")
    parser.add_argument("--only", type=int, nargs="*", help="tick numbers to replay")
    parser.add_argument("--ticks", type=int, default=1, help="decision ticks to play")
    parser.add_argument("--diff", type=int, default=None, metavar="LANDSCAPE")
    parser.add_argument("--jitter", type=int, nargs="+", default=None, metavar="F")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(args.ticks_dir, "t*.npz")))
    if args.only is not None:
        paths = [p for p in paths if int(os.path.basename(p)[1:5]) in args.only]
    if args.jitter is not None:
        for path, acts in jitter(paths, args.jitter, args.workers, args.ticks).items():
            stable = "STABLE" if len(set(acts)) == 1 else "FLIPS"
            print(os.path.basename(path), stable, acts)
        return 0
    for path in tqdm(paths, desc="ticks"):
        print(f"== {os.path.basename(path)}")
        if args.diff is not None:
            with np.load(path) as z:
                for addr, live, typed in image_diff(z["mem"], args.diff):
                    print(f"  ${addr:04X} live={live:#04x} typed={typed:#04x}")
        replay(path, args.ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
