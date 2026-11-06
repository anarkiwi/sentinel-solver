"""Time one board's whole-lattice landing sweep, old prefilter against the current one.

python -m sentinel.tests.sweep_timing [landscape | captures.pkl index]
"""

import pickle
import sys
import time

import numpy as np

from sentinel import landscape, landtable as lt, memmap as mm, playerbase
from sentinel.state import State
from sentinel.tests.test_landing_shortcuts import _old_stop, _stop_args


def _best(fn, reps=3):
    fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return min(out) * 1e3


def main(source="42", index=0):
    """Print the stop_cells and sweep milliseconds, old then new."""
    if source.endswith(".pkl"):
        with open(source, "rb") as fh:
            mem, slot, eye_z = pickle.load(fh)[index]
        st = State(bytearray(mem))
    else:
        st = State(bytearray(landscape.generate(landscape.seed_for(int(source))).mem))
        slot, eye_z = st.player, None
    grids = lt.lattice(v_primary=True)
    comps = lt.cached_components(grids)
    lo, hi, flags = lt.surface_maps(st)
    args = _stop_args(st, slot, eye_z)
    old = lambda: _old_stop(*comps, lo, hi, flags & lt.NEVER_LANDS, *args)
    new = lambda: lt.stop_cells(*comps, lo, hi, flags, *args)
    drive = playerbase._cursor_drive(grids)

    def old_sweep():
        idx = np.flatnonzero(old()[1] > 0)
        lt._march(st, slot, eye_z, grids, idx, lt.MAX_STEPS)

    def sweep(keep):
        return lambda: lt.landing_pairs(
            st, slot, None, grids, lt.MAX_STEPS, keep, drive
        )

    rows = [
        ("stop old", old),
        ("stop new", new),
        ("old stop + march all", old_sweep),
        ("new sweep keep=all", sweep("all")),
        ("new sweep keep=first", sweep("first")),
        ("new sweep keep=cell", sweep("cell")),
    ]
    for name, fn in rows:
        print(f"{name:24s} {_best(fn):9.1f} ms")
    print("rays", comps[0].shape[0], "tiles", mm.N * mm.N)


if __name__ == "__main__":
    main(*sys.argv[1:2], *map(int, sys.argv[2:3]))
