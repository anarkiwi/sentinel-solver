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

from sentinel import enemies, kern, landtable, memmap as mm
from sentinel.state import State


def width():
    """Independent states the CPU runs at once: the numba thread count."""
    return numba.get_num_threads() if numba is not None else 1


def state_key(state):
    """Digest of everything a state-stepping query reads: the 64 KB image and the enemy
    clock's resume fields."""
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


def landings(states, grids, slots=None, eye_z=None, max_steps=landtable.MAX_STEPS):
    """``[(rays, tiles)]`` per state: :func:`landtable.landing_pairs` -- the ascending
    lattice indices that land and their packed ``x*32+y`` tiles.  ``slots`` default to
    each state's player."""
    slots = [st.player for st in states] if slots is None else list(slots)
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
        got = landtable.landing_pairs_many(todo, sl, eye_z, grids, max_steps)
    else:
        got = [
            landtable.landing_pairs(*one, eye_z, grids, max_steps)
            for one in zip(todo, sl)
        ]
    return [got[j] for j in back]


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


class Pool:
    """Board states as arrays addressed by integer row: ``mem`` ``(P, 65536)`` uint8 and
    ``clock`` ``(P, 12)`` int64 (:data:`enemies.CLOCK`).  Queries take row indices and
    return fixed-shape arrays, so selection and scoring over a batch are vectorized numpy.
    """

    def __init__(self, mem=None, clock=None):
        self.mem = np.zeros((0, 1 << 16), np.uint8) if mem is None else mem
        self.clock = (
            np.zeros((0, len(enemies.CLOCK)), np.int64) if clock is None else clock
        )

    def __len__(self):
        return len(self.mem)

    @classmethod
    def from_states(cls, states):
        """A pool holding a copy of each of ``states``, in order."""
        pool = cls()
        pool.add(states)
        return pool

    def add(self, states):
        """Append copies of ``states``; the new rows' indices."""
        first = len(self)
        self.mem = np.concatenate(
            [self.mem] + [np.frombuffer(st.mem, dtype=np.uint8)[None] for st in states]
        )
        self.clock = np.concatenate(
            [self.clock, [[getattr(st, f) for f in enemies.CLOCK] for st in states]]
        ).astype(np.int64)
        return np.arange(first, len(self))

    def fork(self, idx):
        """Append a copy of each row of ``idx``: the expansion primitive; new indices."""
        first = len(self)
        self.mem = np.concatenate([self.mem, self.mem[idx]])
        self.clock = np.concatenate([self.clock, self.clock[idx]])
        return np.arange(first, len(self))

    def state(self, i):
        """Row ``i`` as an independent :class:`State`."""
        return State(bytearray(self.mem[i].tobytes()), *self.clock[i].tolist())

    def keys(self, idx):
        """Per row of ``idx`` the digest of its image and clock."""
        out = []
        for i in idx:
            dig = hashlib.blake2b(self.mem[i].tobytes(), digest_size=16)
            dig.update(self.clock[i].tobytes())
            out.append(dig.digest())
        return out

    def advance(self, idx, n_frames, plotting=False):
        """Advance rows ``idx`` in place by ``n_frames`` (one count or one per row);
        identical rows asked for the same frames run once."""
        idx = np.asarray(idx, dtype=np.int64)
        frames = np.broadcast_to(np.asarray(n_frames, dtype=np.int64), idx.shape)
        reps, back = first_of(list(zip(self.keys(idx), frames.tolist())))
        run = idx[reps]
        sub_mem, sub_clock = self.mem[run], self.clock[run]
        enemies.advance_rows(sub_mem, sub_clock, frames[reps], plotting)
        self.mem[idx], self.clock[idx] = sub_mem[back], sub_clock[back]

    def landing(
        self, idx, grids, slots=None, eye_z=None, max_steps=landtable.MAX_STEPS
    ):
        """``(B, 1024)`` int64: for each row of ``idx`` the lowest ray of ``grids`` landing
        on each packed tile (-1 none).  Identical rows are marched once."""
        idx = np.asarray(idx, dtype=np.int64)
        reps, back = first_of(
            [
                (k, -1 if slots is None else slots[j])
                for j, k in enumerate(self.keys(idx))
            ]
        )
        views = [self.state(i) for i in idx[reps]]
        slot = [v.player if slots is None else slots[r] for v, r in zip(views, reps)]
        pairs = landings(views, grids, slot, eye_z, max_steps)
        counts = [rays.size for rays, _t in pairs]
        first = first_ray(
            np.repeat(np.arange(len(pairs)), counts),
            np.concatenate([rays for rays, _t in pairs]),
            np.concatenate([tiles for _r, tiles in pairs]),
            len(pairs),
        )
        return first[back]
