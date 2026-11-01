"""Per-call timings of the solver's hot kernels on one board: ``python -m
sentinel.tests.bench_kernels [landscape]``."""

import sys
import time

import numpy as np

from sentinel import landtable, occlusioncost
from sentinel.game import Game
from sentinel.playerbase import BOARD_TILES


def _time(label, fn, reps):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    print(f"{label:28s} {(time.perf_counter() - t) / reps * 1e3:8.3f} ms")


def main(landscape=335):
    st = Game.typed(landscape).state
    slot = st.player
    grids = landtable.lattice()
    tiles = [t for t in BOARD_TILES if not landtable.never_lands(st, t, slot)][::40]
    sizes = [landtable.candidates(st, t, slot, None, grids).size for t in tiles]
    print(f"{len(tiles)} tiles, candidates mean {np.mean(sizes):.0f}")
    _time("occlusion_cycles", lambda: occlusioncost.occlusion_cycles(st), 20)
    _time(
        "candidates",
        lambda: [landtable.candidates(st, t, slot, None, grids) for t in tiles],
        3,
    )
    idxs = [landtable.candidates(st, t, slot, None, grids) for t in tiles]
    _time(
        "_march(candidates)",
        lambda: [
            landtable._march(st, slot, None, grids, i, landtable.MAX_STEPS)
            for i in idxs
        ],
        3,
    )


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
