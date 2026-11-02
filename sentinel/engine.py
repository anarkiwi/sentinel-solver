"""The search engine: a pool of board states stored as their mutable bytes, expanded,
advanced and evaluated a batch at a time.

A :class:`Pool` row is the footprint of :mod:`sentinel.footprint` (the bytes the modelled
routines can write) plus the twelve enemy-clock fields; every row shares one static image
for the rest.  A strategy drives it in three calls over row indices and keeps its own
logic in numpy over the returned fixed-shape arrays:

* :meth:`Pool.expand` ``(rows, ops)`` forks each row and applies an action sequence to the
  child, returning the child rows and the sequences that applied;
* :meth:`Pool.advance` ``(rows, frames)`` runs the enemy clock, rows that are identical
  and asked for the same frames running once;
* :meth:`Pool.evaluate` ``(rows)`` reads the decision fields (:data:`rows_jit.FIELDS`).

:meth:`Pool.landing` gives the ``(B, 1024)`` first-landing-ray table.  :func:`first` and
:func:`argbest` are the selection primitives.  Rows and their clocks live in host arrays, or on the device
when ``device`` is set; the kernels are the same specs either way.
"""

import numpy as np

from sentinel import (
    batch,
    enemies,
    footprint,
    kern,
    landtable,
    rows_jit,
)
from sentinel.state import State

IMAGE = footprint.IMAGE
VIEW_BYTES = 1 << 28
NOOP_ROW = np.array([[rows_jit.NOOP, 0, 0]], dtype=np.int64)


def first(mask):
    """Index of the first true entry of ``mask``, or -1: the original-order tie-break."""
    hit = np.flatnonzero(mask)
    return int(hit[0]) if hit.size else -1


def argbest(*keys):
    """Index of the largest row under the lexicographic order of ``keys`` (the first key
    is most significant), earliest first among equals."""
    order = np.lexsort([-np.asarray(k) for k in keys[::-1]])
    return int(order[0])


def pad_ops(sequences):
    """``(B, K, 3)`` int64 ops from ragged sequences of ``(op, a, b)``, ``NOOP``-padded."""
    width = max([len(s) for s in sequences] + [1])
    out = np.tile(NOOP_ROW, (len(sequences), width, 1))
    for i, seq in enumerate(sequences):
        if len(seq):
            out[i, : len(seq)] = seq
    return out


class Pool:
    """Board states by row: mutable bytes ``(P, M)`` uint8, clock ``(P, 12)`` int64, both
    on the device for a device pool, one shared static image."""

    def __init__(self, static, device=False):
        self.static = np.frombuffer(bytes(static), dtype=np.uint8).copy()
        self.addrs, amap = footprint.region()
        self.keep = np.ones(self.addrs.size, dtype=np.int64)
        self.keep[[amap[a] for a in footprint.LOS_TRACKERS if amap[a] >= 0]] = 0
        self.device = bool(device)
        self.n = 0
        if self.device:
            from numba import cuda  # pylint: disable=import-outside-toplevel

            self._cuda = cuda
            self._static_d = cuda.to_device(self.static)
            self._addrs_d = cuda.to_device(self.addrs)
            self._keep_d = cuda.to_device(self.keep)
        self.mut = self._empty(0)
        self.clock = self._empty_clock(0)

    @classmethod
    def from_states(cls, states, device=False):
        """A pool over the image of ``states[0]`` holding a copy of each state."""
        pool = cls(states[0].mem, device)
        pool.add(states)
        return pool

    def __len__(self):
        return self.n

    def _empty(self, rows):
        shape = (rows, self.addrs.size)
        if self.device:
            return self._cuda.device_array(shape, dtype=np.uint8)
        return np.zeros(shape, dtype=np.uint8)

    def _empty_clock(self, rows):
        shape = (rows, len(enemies.CLOCK))
        if self.device:
            return self._cuda.device_array(shape, dtype=np.int64)
        return np.zeros(shape, dtype=np.int64)

    def clocks(self, idx):
        """``(len(idx), 12)`` host int64 :data:`enemies.CLOCK` fields of rows ``idx``."""
        idx = np.asarray(idx, dtype=np.int64)
        if not self.device:
            return self.clock[idx]
        from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

        return kern_cuda.get_clock(self.clock, idx)

    def _put_clocks(self, idx, vals):
        """Clock rows ``idx`` <- the host ``(len(idx), 12)`` ``vals``."""
        if not self.device:
            self.clock[idx] = vals
            return
        from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

        kern_cuda.put_clock(self.clock, idx, vals)

    def _copy_clocks(self, src, dst):
        if not self.device:
            self.clock[dst] = self.clock[src]
            return
        from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

        kern_cuda.copy_clock(self.clock, src, dst)

    def _reserve(self, total):
        """Capacity for ``total`` rows, grown geometrically."""
        cap = len(self.mut)
        if total <= cap:
            return
        rows = max(total, 2 * cap, 16)
        mut, clock = self._empty(rows), self._empty_clock(rows)
        if self.n and self.device:
            mut[: self.n].copy_to_device(self.mut[: self.n])
            clock[: self.n].copy_to_device(self.clock[: self.n])
        elif self.n:
            mut[: self.n] = self.mut[: self.n]
            clock[: self.n] = self.clock[: self.n]
        self.mut, self.clock = mut, clock

    def _chunks(self, idx):
        step = max(1, VIEW_BYTES // IMAGE)
        return [idx[lo : lo + step] for lo in range(0, len(idx), step)]

    def view(self, idx):
        """``(len(idx), 65536)`` full images of rows ``idx``: the static image with each
        row's mutable bytes laid over it (a host or a device array)."""
        idx = np.asarray(idx, dtype=np.int64)
        if not self.device:
            img = np.empty((len(idx), IMAGE), dtype=np.uint8)
            img[:] = self.static
            img[:, self.addrs] = self.mut[idx]
            return img
        from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

        img = self._cuda.device_array((len(idx), IMAGE), dtype=np.uint8)
        rows = self._cuda.to_device(idx)
        grid, block = kern_cuda.grid2(IMAGE, len(idx))
        kern_cuda._fill_kernel[grid, block](self._static_d, img)
        grid, block = kern_cuda.grid2(self.addrs.size, len(idx))
        kern_cuda._overlay_kernel[grid, block](self._addrs_d, self.mut, rows, img)
        return img

    def _store(self, idx, img):
        """Rows ``idx`` <- the mutable bytes of the images ``img``."""
        if not self.device:
            self.mut[idx] = img[:, self.addrs]
            return
        from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

        grid, block = kern_cuda.grid2(self.addrs.size, len(idx))
        kern_cuda._store_kernel[grid, block](
            self._addrs_d, img, self._cuda.to_device(idx), self.mut
        )

    def add(self, states):
        """Append copies of ``states``; the new rows.  A state whose bytes outside the
        footprint differ from the static image does not belong in this pool."""
        first_row = self.n
        self._reserve(self.n + len(states))
        imgs = np.stack([np.frombuffer(st.mem, dtype=np.uint8) for st in states])
        outside = np.ones(IMAGE, dtype=bool)
        outside[self.addrs] = False
        if (imgs[:, outside] != self.static[outside]).any():
            raise ValueError("state differs from the pool's static image off-footprint")
        rows = np.arange(first_row, first_row + len(states))
        if self.device:
            self._store(rows, self._cuda.to_device(imgs))
        else:
            self.mut[rows] = imgs[:, self.addrs]
        self._put_clocks(
            rows,
            np.array(
                [[getattr(st, f) for f in enemies.CLOCK] for st in states],
                dtype=np.int64,
            ),
        )
        self.n += len(states)
        return rows

    def mutable(self, idx):
        """``(len(idx), M)`` host uint8: the mutable bytes of rows ``idx``."""
        idx = np.asarray(idx, dtype=np.int64)
        if not self.device:
            return self.mut[idx]
        from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

        return kern_cuda.gather_rows(self.mut, idx)

    def states(self, idx):
        """Rows ``idx`` as independent :class:`State` objects: only the mutable bytes cross
        from the device, laid over the static image on the host."""
        idx = np.asarray(idx, dtype=np.int64)
        mut, clock = self.mutable(idx), self.clocks(idx)
        out = []
        for row, clk in zip(mut, clock):
            img = self.static.copy()
            img[self.addrs] = row
            out.append(State(bytearray(img.tobytes()), *clk.tolist()))
        return out

    def state(self, i):
        """Row ``i`` as an independent :class:`State`."""
        return self.states([i])[0]

    def fork(self, idx):
        """Append a copy of each row of ``idx``; the new rows."""
        idx = np.asarray(idx, dtype=np.int64)
        new = np.arange(self.n, self.n + len(idx))
        self._reserve(self.n + len(idx))
        if self.device:
            from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

            grid, block = kern_cuda.grid2(self.addrs.size, len(idx))
            kern_cuda._copy_kernel[grid, block](
                self.mut, self._cuda.to_device(idx), self._cuda.to_device(new)
            )
        else:
            self.mut[new] = self.mut[idx]
        self._copy_clocks(idx, new)
        self.n += len(idx)
        return new

    def keys(self, idx):
        """``(len(idx),)`` int64 digest of each row's mutable bytes and clock, bar the
        :data:`footprint.LOS_TRACKERS`: equal rows have equal keys, so they are one entry of
        a transposition table."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros(len(idx), dtype=np.int64)
        if self.device:
            from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

            dev = self._cuda.device_array(len(idx), dtype=np.int64)
            kern_cuda._hash_kernel[kern_cuda._blocks(len(idx)), kern_cuda.BLOCK](
                self.mut, self.clock, self._cuda.to_device(idx), dev, self._keep_d
            )
            return dev.copy_to_host()
        rows_jit.hash_rows(
            self.mut[idx], np.ascontiguousarray(self.clock[idx]), out, self.keep
        )
        return out

    def expand(self, idx, ops):
        """``(children, ok)``: a fork of each row of ``idx`` with the matching ``(K, 3)``
        sequence of ``ops`` (:func:`pad_ops`) applied; ``ok[i]`` is false when a step of
        ``ops[i]`` did not take effect (that child is then partly built)."""
        idx = np.asarray(idx, dtype=np.int64)
        ops = np.ascontiguousarray(ops, dtype=np.int64)
        kids = self.fork(idx)
        ok = np.zeros(len(kids), dtype=np.int64)
        step = max(1, VIEW_BYTES // IMAGE)
        for lo in range(0, len(kids), step):
            rows = kids[lo : lo + step]
            img = self.view(rows)
            part = ok[lo : lo + step]
            if self.device:
                from sentinel import (
                    kern_cuda,
                )  # pylint: disable=import-outside-toplevel

                res = self._cuda.device_array(len(rows), dtype=np.int64)
                kern_cuda._ops_kernel[kern_cuda._blocks(len(rows)), kern_cuda.BLOCK](
                    img, self._cuda.to_device(ops[lo : lo + step]), res
                )
                part[:] = res.copy_to_host()
            else:
                rows_jit.apply_rows(img, ops[lo : lo + step], part)
            self._store(rows, img)
        return kids, ok.astype(bool)

    def evaluate(self, idx):
        """``{field: (B,) int64}`` of :data:`rows_jit.FIELDS` for each row of ``idx``."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros((len(idx), rows_jit.N_FIELDS), dtype=np.int64)
        step = max(1, VIEW_BYTES // IMAGE)
        for lo in range(0, len(idx), step):
            img = self.view(idx[lo : lo + step])
            part = out[lo : lo + step]
            if self.device:
                from sentinel import (
                    kern_cuda,
                )  # pylint: disable=import-outside-toplevel

                dev = self._cuda.device_array(part.shape, dtype=np.int64)
                kern_cuda._fields_kernel[kern_cuda._blocks(len(part)), kern_cuda.BLOCK](
                    img, dev
                )
                part[:] = dev.copy_to_host()
            else:
                rows_jit.fields_rows(img, part)
        return {name: out[:, k] for k, name in enumerate(rows_jit.FIELDS)}

    def advance(self, idx, n_frames, plotting=False):
        """Run the enemy clock of rows ``idx`` for ``n_frames`` (one count, or one per
        row) in place; rows that are identical and asked for the same frames run once.
        On a device pool the rows are advanced where they live: only a row stopped at a
        strip replot crosses to the host, to be priced there."""
        idx = np.asarray(idx, dtype=np.int64)
        if not idx.size:
            return
        frames = np.broadcast_to(np.asarray(n_frames, dtype=np.int64), idx.shape)
        pair = np.stack([self.keys(idx), frames], axis=1)
        _, reps, back = np.unique(pair, axis=0, return_index=True, return_inverse=True)
        back = back.reshape(-1)
        run, todo = idx[reps], frames[reps]
        step = max(1, VIEW_BYTES // IMAGE)
        for lo in range(0, len(run), step):
            rows = run[lo : lo + step]
            img = self.view(rows)
            if self.device:
                from sentinel import (
                    kern_cuda,
                )  # pylint: disable=import-outside-toplevel

                kern_cuda.advance_rows(
                    img, self.clock, rows, todo[lo : lo + step], plotting, _price
                )
            else:
                carry = np.ascontiguousarray(self.clock[rows])
                enemies.advance_rows(img, carry, todo[lo : lo + step], plotting)
                self.clock[rows] = carry
            self._store(rows, img)
        dup = np.flatnonzero(reps[back] != np.arange(len(idx)))
        if dup.size:
            self.fork_into(idx[reps[back[dup]]], idx[dup])

    def fork_into(self, src, dst):
        """Overwrite rows ``dst`` with copies of rows ``src``."""
        src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
        if self.device:
            from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

            grid, block = kern_cuda.grid2(self.addrs.size, len(src))
            kern_cuda._copy_kernel[grid, block](
                self.mut, self._cuda.to_device(src), self._cuda.to_device(dst)
            )
        else:
            self.mut[dst] = self.mut[src]
        self._copy_clocks(src, dst)

    def landing(
        self, idx, grids, slots=None, eye_z=None, max_steps=landtable.MAX_STEPS
    ):
        """``(B, 1024)`` int64: for each row of ``idx`` the lowest ray of ``grids`` landing
        on each packed tile (-1 none).  Identical rows are marched once."""
        idx = np.asarray(idx, dtype=np.int64)
        reps, back = batch.first_of(
            [
                (k, -1 if slots is None else slots[j])
                for j, k in enumerate(self.keys(idx).tolist())
            ]
        )
        views = self.states(idx[reps])
        slot = [v.player if slots is None else slots[r] for v, r in zip(views, reps)]
        pairs = batch.landings(views, grids, slot, eye_z, max_steps)
        counts = [rays.size for rays, _t in pairs]
        table = batch.first_ray(
            np.repeat(np.arange(len(pairs)), counts),
            np.concatenate([rays for rays, _t in pairs]),
            np.concatenate([tiles for _r, tiles in pairs]),
            len(pairs),
        )
        return table[back]


def _price(img, clock, request):
    """Price the $1FFC strip replot ``request`` of one stopped row: its ``(image,
    clock)`` after the stall is booked."""
    state = State(bytearray(img), *clock.tolist())
    enemies._price_replot(  # pylint: disable=protected-access
        state, *[int(v) for v in request]
    )
    return (
        np.frombuffer(state.mem, dtype=np.uint8).copy(),
        np.array([getattr(state, f) for f in enemies.CLOCK], dtype=np.int64),
    )


def default_device(rows):
    """Whether a pool expected to hold ``rows`` rows lives on the device: the backend is
    cuda and a round has a batch of more than one row."""
    return rows > 1 and kern.backend() == "cuda"
