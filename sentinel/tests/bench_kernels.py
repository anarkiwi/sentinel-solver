"""Per-call timings of the solver's hot kernels on one board: ``python -m
sentinel.tests.bench_kernels [landscape]``; ``... sweep [landscape [actions]]`` times the
landable-view sweep's steps; ``... grain`` derives :data:`landtable.CROSS_GRAIN`."""

import sys
import time

import numba
import numpy as np

from sentinel import landtable, occlusioncost, pancost, playerbase, projector
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


def _best(fn, args, reps):
    fn(*args)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        t.append(time.perf_counter() - t0)
    return min(t)


def crossing_grain(reps=7):
    """The rays per thread from which threading :func:`landtable.crossing_mask` beats
    the serial kernel at every thread count, on contiguous heading-ordered slices."""
    vx, vy, vz = landtable.heading_components(landtable.lattice())[2:]
    top = numba.get_num_threads()
    rng = np.random.default_rng(0)

    def rays(n):
        a = int(rng.integers(0, vx.size - n + 1))
        return (vx[a : a + n], vy[a : a + n], vz[a : a + n], 5, 3, -0x300, 0x100, 6000)

    args = rays(1 << 16)
    ser = _best(landtable._crossing_ser, args, reps)
    print(f"serial crossing: {ser / (1 << 16) * 1e9:.1f} ns/ray")
    g = 256
    while g * top <= vx.size:
        wins = True
        for p in sorted({2, 4, 8, 16, top} & set(range(2, top + 1))):
            args = rays(p * g)
            ser = _best(landtable._crossing_ser, args, reps)
            numba.set_num_threads(p)
            try:
                par = _best(landtable._crossing_par, args, reps)
            finally:
                numba.set_num_threads(top)
            print(
                f"grain {g:6d} threads {p:3d}: {ser * 1e3:8.3f} / {par * 1e3:8.3f} ms"
            )
            wins &= par < ser
        if wins:
            break
        g *= 2
    print(f"CROSS_GRAIN = {g}")
    return g


def sweep(landscape=335, actions=0):
    """Each step of :func:`playerbase._sweep` per lattice, on ``landscape`` after the phase
    player's first ``actions`` ticks; pricing is timed with cold and warm render caches.
    """
    from sentinel.phase_player import PhasePlayer

    player = PhasePlayer(Game.typed(landscape), verbose=False)
    if actions:
        player.run(max_actions=actions)
    st = player.st
    slot = st.player
    aim_from = (st.obj_h_angle[slot], st.obj_v_angle[slot])
    for v_primary in (True, False):
        grids = landtable.lattice(v_primary=v_primary)
        idx = landtable.landing_rays(st, slot, None, grids)
        print(f"v_primary={v_primary}: {idx.size} landing rays")
        _time("landing_rays", lambda: landtable.landing_rays(st, slot, None, grids), 3)
        _time(
            "_march",
            lambda: landtable._march(st, slot, None, grids, idx, landtable.MAX_STEPS),
            3,
        )
        _time(
            "_sweep (unpriced)", lambda: playerbase._sweep(st, v_primary, aim_from), 3
        )
        for label in ("price all (cold)", "price all (warm)"):
            if "cold" in label:
                pancost._NOTCH_CACHE.clear()
                projector._COST_CACHE.clear()
            got = playerbase._sweep(st, v_primary, aim_from)
            t = time.perf_counter()
            views = got.views()
            print(
                f"{label:28s} {(time.perf_counter() - t) * 1e3:8.3f} ms, {len(views)} tiles"
            )


if __name__ == "__main__":
    if sys.argv[1:] == ["grain"]:
        crossing_grain()
    elif sys.argv[1:2] == ["sweep"]:
        sweep(*(int(a) for a in sys.argv[2:]))
    else:
        main(*(int(a) for a in sys.argv[1:]))
