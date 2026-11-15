"""Batched multi-state queries: each entry point takes a list of board states and returns,
per state, exactly what the single-state call returns.

Identical states in a batch are computed once and the result fanned back out; the backend
per query kind follows from the device's occupancy (docs/architecture.md, "Batched queries").
"""

import hashlib

try:
    import numba
except ImportError:  # pragma: no cover - numba absent
    numba = None

from sentinel import kern, landtable


def width():
    """Independent states the CPU runs at once: the numba thread count."""
    return numba.get_num_threads() if numba is not None else 1


def state_key(state):
    """Digest of the memory image, all a landing query reads."""
    return hashlib.blake2b(state.mem, digest_size=16).digest()


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
    """``[(rays, tiles)]`` per state: :func:`landtable.landing_pairs` from ``slots`` (each player).

    ``keep`` names the rays the caller reads: ``"all"``, ``"first"`` (the lowest ray per
    tile) or ``"cell"`` (the least ``(drive, ray)`` per tile and cell, ``drive`` as
    :func:`landtable.claim_plan`).  Every such ray is returned, plus possibly others."""
    slots = [st.player for st in states] if slots is None else list(slots)
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
