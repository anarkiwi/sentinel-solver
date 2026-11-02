"""CUDA launch layer: one thread per (state, lattice ray) over the CUDA builds of the specs.

Each thread writes its own cell of a dense per-ray result (the packed landing tile, -1 for
a ray that does not end LOS_CLEAR), so results never depend on thread scheduling and need
no sort: the host compacts them in lattice order.  The state-independent lattice vectors
stay resident on the device between calls, and B board images march in one launch.
"""

# pylint: disable=no-value-for-parameter,too-many-function-args,comparison-with-callable

import functools
import hashlib
import os
import types

import numpy as np
from numba import cuda

from sentinel import (
    enemies_jit,
    kern,
    landtable,
    los_jit,
    memmap as mm,
    pricing_jit,
    rows_jit,
)

BLOCK = 128
LOS_CLEAR = los_jit.LOS_CLEAR
_march_one = kern.build(los_jit._march_one, "cuda")
_stop_one = kern.build(landtable._stop_one, "cuda")
_MEMO = {}
_MEMO_MAX = 8


@cuda.jit(cache=True)
def _clear_kernel(
    mems, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seeds, obss, out
):
    """``out[b, i]`` = packed tile of ray ``i`` marched against image ``b``, -1 unless
    the ray ends LOS_CLEAR."""
    i, b = cuda.grid(2)
    if i < ax_lo.shape[0] and b < mems.shape[0]:
        r = _march_one(
            mems[b], i, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seeds[b], obss[b]
        )
        out[b, i] = r[1] * mm.N + r[2] if r[0] == LOS_CLEAR else -1


@cuda.jit(cache=True)
def _stop_kernel(vx, vy, vz, lo, hi, slope, ex, ey, zseed, max_steps, edge, bits):
    """Set bit ``j`` of ``bits`` for every ray with a candidate landing tile."""
    j = cuda.grid(1)
    if j < vx.shape[0]:
        _first, ncand = _stop_one(
            np.int64(vx[j]),
            np.int64(vy[j]),
            np.int64(vz[j]),
            lo,
            hi,
            slope,
            ex,
            ey,
            zseed,
            max_steps,
            edge,
        )
        if ncand > 0:
            cuda.atomic.or_(bits, j >> 5, np.uint32(1 << (j & 31)))


def _blocks(n):
    return (n + BLOCK - 1) // BLOCK


def resident_threads():
    """Threads the device keeps in flight at once (0 under the simulator, which has no
    occupancy to fill)."""
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        return 0
    dev = cuda.get_current_device()
    return dev.MULTIPROCESSOR_COUNT * dev.MAX_THREADS_PER_MULTIPROCESSOR


def landing_dense(mems, vecs, params):
    """``(B, rays)`` int16 on the device for the ``(B, 65536)`` image stack ``mems`` and
    ``params`` rows (seed then observer, as :func:`landtable._march_params`): the packed
    landing tile of every ray of ``vecs`` against each image, -1 where it does not land.
    One launch; the result stays a device array."""
    n = vecs[0].shape[0]
    rows = np.asarray(params, dtype=np.int64)
    dense = cuda.device_array((len(mems), n), dtype=np.int16)
    _clear_kernel[(_blocks(n), len(mems)), BLOCK](
        cuda.to_device(np.ascontiguousarray(mems, dtype=np.uint8)),
        *vecs,
        cuda.to_device(np.ascontiguousarray(rows[:, :9])),
        cuda.to_device(np.ascontiguousarray(rows[:, 9:])),
        dense,
    )
    return dense


def landing_many(mems, vecs, params):
    """``[(index, tile)]`` per image of ``mems``: every CLEAR ray of ``vecs`` marched
    against that image from its ``params``, ascending.  Images go in launches of as many
    as :data:`landtable.RESULT_BYTES` of dense results hold."""
    step = max(1, landtable.RESULT_BYTES // (2 * vecs[0].shape[0]))
    out = []
    for lo in range(0, len(mems), step):
        part = slice(lo, lo + step)
        images = np.stack([np.asarray(m, dtype=np.uint8) for m in mems[part]])
        out += landtable._compact(
            landing_dense(images, vecs, params[part]).copy_to_host()
        )
    return out


def landing(mem, vecs, params):
    """:func:`landing_many` for one image."""
    return landing_many([mem], vecs, [params])[0]


def stop_mask(comps, lo, hi, slope, ex, ey, zseed, max_steps, edge):
    """Boolean per ray: :func:`landtable.stop_cells` reports a candidate landing tile."""
    n = comps[0].shape[0]
    bits = cuda.to_device(np.zeros((n + 31) // 32, dtype=np.uint32))
    surf = (cuda.to_device(a) for a in (lo, hi, slope))
    _stop_kernel[_blocks(n), BLOCK](*comps, *surf, ex, ey, zseed, max_steps, edge, bits)
    raw = bits.copy_to_host().view(np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n].astype(bool)


@functools.lru_cache(maxsize=4)
def device_vectors(key):
    """The seven int16 lattice component arrays of grids ``key``, resident on the device."""
    return tuple(
        cuda.to_device(np.ascontiguousarray(a)) for a in landtable.vectors(key)
    )


@functools.lru_cache(maxsize=4)
def device_components(key):
    """The int32 ``(vx, vy, vz)`` of grids ``key``, resident on the device."""
    comps = landtable.cached_components(key)
    return tuple(cuda.to_device(np.ascontiguousarray(a)) for a in comps)


def _tile_first(idx, tile):
    first = np.full(mm.N * mm.N, -1, dtype=np.int64)
    tiles, at = np.unique(tile, return_index=True)
    first[tiles] = idx[at]
    return first


def landing_tables(states, slots, eye_z, grids, max_steps):
    """``(index, tile, first)`` per state: ``grids``' whole-lattice sweep against each
    ``states[i]`` from ``slots[i]`` -- ascending CLEAR ray indices, their packed tiles,
    and per tile the lowest index (-1 none).  Memoized on the board bytes, which
    determine the march; boards repeated within the call march once, the rest in one
    launch."""
    key = tuple(tuple(g) for g in grids)
    memos = [
        (
            hashlib.blake2b(st.mem, digest_size=16).digest(),
            slot,
            eye_z,
            max_steps,
            key,
        )
        for st, slot in zip(states, slots)
    ]
    have = {m: _MEMO[m] for m in memos if m in _MEMO}
    todo = {}
    for i, memo in enumerate(memos):
        if memo not in have:
            todo.setdefault(memo, i)
    if todo:
        at = list(todo.values())
        got = landing_many(
            [np.frombuffer(states[i].mem, dtype=np.uint8) for i in at],
            device_vectors(key),
            [
                landtable._march_params(states[i], slots[i], eye_z, max_steps)
                for i in at
            ],
        )
        for memo, (idx, tile) in zip(todo, got):
            if len(_MEMO) >= _MEMO_MAX:
                _MEMO.pop(next(iter(_MEMO)))
            have[memo] = _MEMO[memo] = (idx, tile, _tile_first(idx, tile))
    return [have[memo] for memo in memos]


def landing_table(state, slot, eye_z, grids, max_steps):
    """:func:`landing_tables` for one state."""
    return landing_tables([state], [slot], eye_z, grids, max_steps)[0]


@cuda.jit(cache=True)
def _copy_kernel(mut, src, dst):
    """Row ``dst[b]`` of ``mut`` <- row ``src[b]``."""
    j, b = cuda.grid(2)
    if b < src.shape[0] and j < mut.shape[1]:
        mut[dst[b], j] = mut[src[b], j]


@cuda.jit(cache=True)
def _fill_kernel(static, out):
    """Every row of ``out`` <- the shared static image."""
    i, b = cuda.grid(2)
    if b < out.shape[0] and i < out.shape[1]:
        out[b, i] = static[i]


@cuda.jit(cache=True)
def _overlay_kernel(addrs, mut, idx, out):
    """``out[b, addrs[j]]`` <- the mutable byte ``j`` of row ``idx[b]``."""
    j, b = cuda.grid(2)
    if b < idx.shape[0] and j < addrs.shape[0]:
        out[b, addrs[j]] = mut[idx[b], j]


@cuda.jit(cache=True)
def _store_kernel(addrs, imgs, idx, mut):
    """The mutable bytes of row ``idx[b]`` <- ``imgs[b, addrs[j]]``."""
    j, b = cuda.grid(2)
    if b < idx.shape[0] and j < addrs.shape[0]:
        mut[idx[b], j] = imgs[b, addrs[j]]


def grid2(width, rows):
    """Launch geometry covering ``width`` columns by ``rows`` rows."""
    return (_blocks(width), rows), BLOCK


def advance_rows(imgs, clock, n_frames, plotting, price):
    """Advance row ``i`` of the device stack ``imgs`` by ``n_frames[i]`` frames in place,
    from and into the host ``(B, 12)`` ``clock``.  Rounds of :func:`_advance_kernel` run the
    live rows; a row stopped at a strip replot is handed to ``price(image bytes, clock row,
    request)``, which returns its image and clock row, and re-enters the next round."""
    left = np.array(n_frames, dtype=np.int64)
    while True:
        live = np.flatnonzero(left > 0)
        if not live.size:
            return
        carry = cuda.to_device(np.ascontiguousarray(clock[live]))
        rest = cuda.device_array(live.size, dtype=np.int64)
        replot = cuda.device_array((live.size, 4), dtype=np.int64)
        row_kernels().advance[_blocks(live.size), BLOCK](
            imgs,
            cuda.to_device(live),
            cuda.device_array((live.size, enemies_jit.ZP_N), dtype=np.int64),
            cuda.to_device(left[live]),
            bool(plotting),
            carry,
            rest,
            replot,
        )
        clock[live] = carry.copy_to_host()
        rest, replot = rest.copy_to_host(), replot.copy_to_host()
        for k in np.flatnonzero(replot[:, 0] >= 0):
            row = live[k]
            img, clock[row] = price(
                imgs[row].copy_to_host().tobytes(), clock[row], replot[k]
            )
            imgs[row].copy_to_device(img)
        left[live] = np.where(replot[:, 0] >= 0, np.maximum(rest, 0), 0)


@functools.cache
def row_kernels():
    """The device kernels over the clock and row specs, built on first use: their tables are
    device arrays, which numba cannot cache, so they compile once per process and only when
    a device pool asks for them."""
    advance_one = kern.build(enemies_jit._advance, "cuda")
    apply_ops = kern.build(rows_jit.apply_ops, "cuda")
    fields = kern.build(rows_jit.fields, "cuda")
    row_hash = kern.build(rows_jit.row_hash, "cuda")

    @cuda.jit
    def advance_kernel(mems, rows, zps, n_frames, plotting, carry, rest, replot):
        """One thread per live row ``b``: :func:`enemies_jit._advance` over ``mems[rows[b]]``
        from the clock fields ``carry[b]`` (updated in place), the frames left in ``rest[b]`` and
        the strip replot request, if any, in ``replot[b]`` (target, left column, columns,
        span)."""
        b = cuda.grid(1)
        if b < rows.shape[0]:
            zp = zps[b]
            for k in range(zp.shape[0]):
                zp[k] = 0
            c = carry[b]
            r = advance_one(
                mems[rows[b]],
                zp,
                n_frames[b],
                plotting,
                c[0],
                c[1],
                c[2],
                c[3],
                c[4],
                c[5],
                c[6],
                c[7],
                c[8],
                c[9],
                c[10],
                c[11],
            )
            for k in range(12):
                c[k] = r[k]
            rest[b] = r[12]
            replot[b, 0] = zp[enemies_jit.ZP_REPLOT] - 1
            replot[b, 1] = zp[enemies_jit.ZP_REPLOT_LEFT]
            replot[b, 2] = zp[enemies_jit.ZP_REPLOT_COLS]
            replot[b, 3] = zp[enemies_jit.ZP_REPLOT_SPAN]

    @cuda.jit
    def ops_kernel(imgs, ops, ok):
        b = cuda.grid(1)
        if b < imgs.shape[0]:
            ok[b] = apply_ops(imgs[b], ops[b])

    @cuda.jit
    def fields_kernel(imgs, out):
        b = cuda.grid(1)
        if b < imgs.shape[0]:
            fields(imgs[b], out[b])

    @cuda.jit
    def hash_kernel(mut, clock, idx, out, keep):
        b = cuda.grid(1)
        if b < idx.shape[0]:
            out[b] = row_hash(mut[idx[b]], clock[b], keep)

    return types.SimpleNamespace(
        advance=advance_kernel, ops=ops_kernel, fields=fields_kernel, hash=hash_kernel
    )


@functools.cache
def render_kernel():
    """The pricing kernel over :func:`pricing_jit.render_cycles`, built on first use: thread
    ``t`` of the launch owns scratch slot ``t`` and prices jobs ``t, t + threads, ...``.
    """
    render_one = kern.build(pricing_jit.render_cycles, "cuda")

    @cuda.jit
    def render_kernel_(mems, jobs, vis, model, scr, out):
        t = cuda.grid(1)
        slots = scr[0].shape[0]
        if t < slots:
            tile_work = (
                scr[14][t],
                scr[15][t],
                scr[16][t],
                scr[17][t],
                scr[18][t],
                scr[19][t],
            )
            obj_work = (
                scr[20][t],
                scr[21][t],
                scr[22][t],
                scr[23][t],
                scr[24][t],
                scr[25][t],
            )
            for j in range(t, jobs.shape[0], slots):
                out[j] = render_one(
                    mems[jobs[j, pricing_jit.J_MEM]],
                    jobs[j],
                    vis[jobs[j, pricing_jit.J_VIS]],
                    model,
                    scr[0][t],
                    scr[1][t],
                    scr[2][t],
                    scr[3][t],
                    scr[4][t],
                    scr[5][t],
                    scr[6][t],
                    scr[7][t],
                    scr[8][t],
                    scr[9][t],
                    scr[10][t],
                    scr[11][t],
                    tile_work,
                    obj_work,
                    scr[12][t],
                    scr[13][t],
                )

    return render_kernel_


_MODEL_DEVICE = {}
_PRICING = {"scr": (), "out": None}


def _simulated():
    return os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def pricing_pays(jobs):
    """Whether ``jobs`` pricing jobs give every multiprocessor a full block of one-job
    threads: below that the launch leaves multiprocessors idle and the CPU threads win.
    """
    if _simulated():
        return False
    return jobs >= BLOCK * cuda.get_current_device().MULTIPROCESSOR_COUNT


def _pricing_buffers(jobs):
    """``(slots, scratch, out)`` of a launch over ``jobs`` jobs, from the persistent
    device buffers: grown when a launch needs more slots, never freed between launches.
    A slot is one thread's scratch, so the slot count is capped by the threads the device
    keeps resident and by half the device memory."""
    per_slot = sum(a.nbytes for a in pricing_jit.scratch(1))
    slots = jobs
    if not _simulated():
        held = len(_PRICING["scr"][0]) * per_slot if _PRICING["scr"] else 0
        free = cuda.current_context().get_memory_info()[0]
        slots = min(slots, resident_threads(), max(1, (free + held) // 2 // per_slot))
    slots = max(1, slots)
    if not _PRICING["scr"] or len(_PRICING["scr"][0]) < slots:
        _PRICING["scr"] = ()
        _PRICING["scr"] = tuple(
            cuda.device_array((slots,) + a.shape[1:], dtype=a.dtype)
            for a in pricing_jit.scratch(1)
        )
    if _PRICING["out"] is None or len(_PRICING["out"]) < jobs:
        _PRICING["out"] = cuda.device_array(jobs, dtype=np.int64)
    return slots, tuple(a[:slots] for a in _PRICING["scr"]), _PRICING["out"][:jobs]


def render_cycles_many(mems, jobs, vis, model):
    """:func:`pricing_jit.render_many` on the device: ``(n,)`` int64 cycles of each job of
    ``jobs`` over the image stack ``mems`` and occlusion tables ``vis``, one thread per
    job up to the slots :func:`_pricing_buffers` grants, with one result download."""
    key = id(model)
    if key not in _MODEL_DEVICE:
        _MODEL_DEVICE[key] = (model, tuple(cuda.to_device(a) for a in model))
    slots, scr, out = _pricing_buffers(len(jobs))
    render_kernel()[_blocks(slots), BLOCK](
        cuda.to_device(np.ascontiguousarray(mems)),
        cuda.to_device(jobs),
        cuda.to_device(np.ascontiguousarray(vis)),
        _MODEL_DEVICE[key][1],
        scr,
        out,
    )
    return out.copy_to_host()
