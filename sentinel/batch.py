"""Batched multi-state queries: each entry point takes a list of board states and returns,
per state, exactly what the single-state call returns.

Identical states in a batch are computed once and the result fanned back out.  Backend
selection per query kind follows from the device's occupancy (see docs/architecture.md,
"Batched queries").
"""

import hashlib

import numpy as np

try:
    import numba
except ImportError:  # pragma: no cover - numba absent
    numba = None

from sentinel import enemies, fleet, kern, landtable, memmap as mm


def width():
    """Independent states the CPU runs at once: the numba thread count."""
    return numba.get_num_threads() if numba is not None else 1


def state_key(state):
    """Digest of everything a state-stepping query reads: the memory image and the
    enemy clock's resume fields."""
    dig = hashlib.blake2b(state.mem, digest_size=16)
    dig.update(repr([int(getattr(state, f)) for f in enemies.CLOCK]).encode())
    return dig.digest()


def first_of(keys):
    """``(reps, back)``: the position of each distinct key's first occurrence, and per
    position the index into ``reps`` of its key."""
    seen = {}
    back = [seen.setdefault(k, len(seen)) for k in keys]
    reps = [0] * len(seen)
    for pos in range(len(keys) - 1, -1, -1):
        reps[back[pos]] = pos
    return reps, back


def cuda_pays(rays, states):
    """Whether ``rays`` x ``states`` threads fill the device: below that the launch leaves
    multiprocessors idle and the CPU's parallel loops, which saturate at one state, win.
    """
    if kern.backend() != "cuda":
        return False
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    return rays * states >= kern_cuda.resident_threads()


def landings(
    states,
    grids,
    slots=None,
    eye_z=None,
    max_steps=landtable.MAX_STEPS,
    keep="all",
    drive=None,
):
    """``[(rays, tiles)]`` per state: :func:`landtable.landing_pairs` -- the ascending
    lattice indices that land and their packed ``x*32+y`` tiles.  ``slots`` default to
    each state's player.

    ``keep`` names the rays the caller reads: ``"all"``, ``"first"`` (the lowest ray per
    tile) or ``"cell"`` (the least ``(drive, ray)`` per tile and cell, ``drive`` as
    :func:`landtable.claim_plan`).  Every such ray is returned, plus possibly others."""
    slots = [st.player for st in states] if slots is None else list(slots)
    if fleet.in_task():
        return fleet.call(
            "landing", (states, grids, slots, eye_z, max_steps, keep, drive)
        )
    return _landings(states, grids, slots, eye_z, max_steps, keep, drive)


def _landings(states, grids, slots, eye_z, max_steps, keep="all", drive=None):
    """:func:`landings` of explicit ``slots``, on this thread."""
    if cuda_pays(len(landtable.cached_components(grids)[0]), len(states)):
        from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

        return [
            got[:2]
            for got in kern_cuda.landing_tables(states, slots, eye_z, grids, max_steps)
        ]
    keys = [(state_key(st), s) for st, s in zip(states, slots)]
    reps, back = first_of(keys)
    todo = [states[i] for i in reps]
    sl = [slots[i] for i in reps]
    if len(reps) >= width():
        got = landtable.landing_pairs_many(
            todo, sl, eye_z, grids, max_steps, keep, drive
        )
    else:
        got = [
            landtable.landing_pairs(*one, eye_z, grids, max_steps, keep, drive)
            for one in zip(todo, sl)
        ]
    return [got[j] for j in back]


def _landings_fleet(asks):
    """The fleet's ``landing`` handler: tasks asking of the same lattice share a launch."""
    groups = {}
    for i, (_, grids, _, eye_z, max_steps, keep, drive) in enumerate(asks):
        shared = None if drive is None else bytes(drive)
        groups.setdefault(
            (tuple(map(tuple, grids)), eye_z, max_steps, keep, shared), []
        ).append(i)
    out = [None] * len(asks)
    for (_, eye_z, max_steps, keep, _drive), at in groups.items():
        states = [st for i in at for st in asks[i][0]]
        slots = [s for i in at for s in asks[i][2]]
        got = iter(
            _landings(
                states, asks[at[0]][1], slots, eye_z, max_steps, keep, asks[at[0]][6]
            )
        )
        for i in at:
            out[i] = [next(got) for _ in asks[i][0]]
    return out


fleet.register("landing", _landings_fleet)


def advance(states, n_frames, plotting=False):
    """:func:`enemies.advance_frames` of each state by ``n_frames`` (one count, or one per
    state), in place.  States that are identical and asked for the same frames run once.
    The enemy clock stays on the CPU, one state per thread."""
    frames = np.broadcast_to(np.asarray(n_frames, dtype=np.int64), (len(states),))
    keys = [(state_key(st), int(f)) for st, f in zip(states, frames)]
    reps, back = first_of(keys)
    enemies.advance_frames_many([states[i] for i in reps], frames[reps], plotting)
    for pos, j in enumerate(back):
        src = states[reps[j]]
        if pos != reps[j]:
            states[pos].mem[:] = bytes(src.mem)
            for f in enemies.CLOCK:
                setattr(states[pos], f, getattr(src, f))


def first_ray(rows, rays, tiles, count):
    """``(count, 1024)`` int64: the lowest of ``rays`` landing on each packed tile in each
    row ``rows`` names, -1 where none does."""
    none = np.iinfo(np.int64).max
    out = np.full((count, mm.N * mm.N), none, dtype=np.int64)
    np.minimum.at(out, (rows, tiles), rays)
    out[out == none] = -1
    return out


def first_by_tile(dense):
    """:func:`first_ray` of a dense landing matrix (tile per ray, -1 none)."""
    row, ray = np.nonzero(dense >= 0)
    return first_ray(row, ray, dense[row, ray], len(dense))
