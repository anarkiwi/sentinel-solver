#!/usr/bin/env python3
"""Record a search game's state after every committed move.

``python -m driver.trajectory N [--preset wide] [--out out/traj]`` plays landscape ``N``
with the search player on the emulator and writes ``out/traj/traj_NNNN.npz``: ``rams``,
the 64 KiB image at entry and after each move, and ``trace``, the moves as JSON.
"""

import argparse
import json
import os

import numpy as np

from driver import core, search_player as sp
from driver.emu import EmuMon
from sentinel import emu


class Recorder(sp.SearchPlayer):
    """A search player that keeps the machine's RAM after each move it commits."""

    def __init__(self, bm, **kw):
        super().__init__(bm, **kw)
        self.rams = [image(bm)]

    def _execute(self, planned):
        node = super()._execute(planned)
        self.rams.append(image(self.bm))
        return node


def image(bm):
    """The 64 KiB RAM image ``bm``'s game is in, as uint8."""
    return np.frombuffer(bytes(core.live_image(bm)), dtype=np.uint8).copy()


def record(landscape, preset="wide", max_actions=120):
    """``(won, rams, trace)`` of the search game on ``landscape``."""
    player = Recorder(EmuMon(emu.enter(landscape)), **sp.PRESETS[preset])
    won = player.run(max_actions)
    return won, np.stack(player.rams), player.trace


def load(path):
    """``(rams, trace, won)`` of a recorded game."""
    with np.load(path) as data:
        return data["rams"], json.loads(str(data["trace"])), bool(data["won"])


def main(argv=None):
    """Record one landscape's game."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("landscape", type=int)
    ap.add_argument("--preset", default="wide", choices=sorted(sp.PRESETS))
    ap.add_argument("--max-actions", type=int, default=120)
    ap.add_argument("--out", default="out/traj")
    args = ap.parse_args(argv)
    won, rams, trace = record(args.landscape, args.preset, args.max_actions)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"traj_{args.landscape:04d}.npz")
    np.savez_compressed(path, rams=rams, trace=json.dumps(trace), won=won)
    print(f"{path}: won={won} moves={len(trace)}")


if __name__ == "__main__":
    main()
