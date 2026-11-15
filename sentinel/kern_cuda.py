"""CUDA launch layer: one thread per (state, lattice ray) over the CUDA builds of the specs.

Each thread writes its own cell of a dense per-ray result (packed landing tile, -1 when not
LOS_CLEAR), so results are scheduling-independent and the host compacts them in lattice order.
The lattice vectors stay resident on the device; B board images march in one launch.
"""

# pylint: disable=no-value-for-parameter,too-many-function-args,comparison-with-callable

import functools
import hashlib
import os

import numpy as np
from numba import cuda

from sentinel import (
    kern,
    landtable,
    los_jit,
    memmap as mm,
    pricing_jit,
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


_BUFFERS = {}


def scratch(name, shape, dtype):
    """A device array of ``shape`` over the buffer called ``name``, which persists between
    calls and grows geometrically: the contents are the caller's to fill."""
    size = int(np.prod(shape))
    buf = _BUFFERS.get(name)
    if buf is None or buf.dtype != np.dtype(dtype) or buf.size < size:
        have = 0 if buf is None or buf.dtype != np.dtype(dtype) else buf.size
        _BUFFERS[name] = None
        _BUFFERS[name] = buf = cuda.device_array(max(size, 2 * have, 1), dtype=dtype)
    return buf[:size].reshape(shape)


def stage(name, host):
    """``host`` copied into the persistent device buffer ``name``: one upload, no
    allocation once the buffer has grown to the largest call."""
    host = np.ascontiguousarray(host)
    view = scratch(name, host.shape, host.dtype)
    if host.size:
        view.reshape(-1).copy_to_device(host.reshape(-1))
    return view


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
    landing tile of every ray of ``vecs`` against each image, -1 where it does not land;
    a view of a persistent device buffer, valid until the next call."""
    n = vecs[0].shape[0]
    rows = np.asarray(params, dtype=np.int64)
    dense = scratch("landing_out", (len(mems), n), np.int16)
    _clear_kernel[(_blocks(n), len(mems)), BLOCK](
        stage("landing_img", np.asarray(mems, dtype=np.uint8)),
        *vecs,
        stage("landing_params", rows[:, :9]),
        stage("landing_obs", rows[:, 9:]),
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
    and per tile the lowest index (-1 none).  Memoized on the board bytes; the rest march
    in one launch."""
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
    keeps resident and by half the device memory free at the first launch."""
    per_slot = sum(a.nbytes for a in pricing_jit.scratch(1))
    slots = jobs
    if not _simulated():
        if "slots" not in _PRICING:
            free = cuda.current_context().get_memory_info()[0]
            _PRICING["slots"] = min(resident_threads(), max(1, free // 2 // per_slot))
        slots = min(slots, _PRICING["slots"])
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
        stage("price_mems", mems),
        stage("price_jobs", jobs),
        stage("price_vis", vis),
        _MODEL_DEVICE[key][1],
        scr,
        out,
    )
    return out.copy_to_host()
