"""One plot_world pass priced in integer cycles: visibility walk, terrain fill, objects.

A job is an int64 row of ``J_N`` columns (setup vector, buffer window, raster rows, view).
"""

import numba
import numpy as np
from numba import njit, prange

from sentinel import memmap as mm, objmodel
from sentinel.kern import spec
from sentinel.objectcost import pass_cycles
from sentinel.projector_jit import S_N, S_QUAD, SCAN_CAP, scan_scene
from sentinel.rendercost import F_N, R_N, TILE_COLS

J_SU = 0
J_HINT = S_N
J_SECT = S_N + 1
J_TOP = S_N + 2
J_BOT = S_N + 3
J_OBS = S_N + 4
J_PLAYER = S_N + 5
J_H = S_N + 6
J_V = S_N + 7
J_VIS = S_N + 8
J_BUFS = S_N + 9
J_N = J_BUFS + 12
WORK_VERTICES = objmodel.MAX_VERTICES + 2


@spec
def render_cycles(
    mem,
    job,
    vis,
    model,
    zp,
    cres,
    seen,
    cnt,
    rrow,
    rlo,
    rhi,
    out,
    ws,
    fill,
    left,
    right,
    tile_work,
    obj_work,
    slots,
    bufs,
):
    """Cycles of the pass ``job`` over image ``mem``, occlusion table ``vis`` and object
    ``model``; every array after ``model`` is the caller's scratch."""
    _n, nfill = scan_scene(
        mem,
        zp,
        job[J_SU : J_SU + S_N],
        vis,
        job[J_HINT],
        0,
        0,
        cres,
        seen,
        cnt,
        rrow,
        rlo,
        rhi,
        out,
        ws,
        fill,
    )
    for k in range(12):
        bufs[k >> 2, k & 3] = job[J_BUFS + k]
    for k in range(256):
        left[k] = 0
        right[k] = 0
    quad = job[J_SU + S_QUAD]
    cyc = pass_cycles(
        mem,
        zp,
        model,
        fill,
        nfill,
        (quad - 2) & 0xFF,
        (quad * 4) & 0xFF,
        job[J_SECT],
        bufs,
        job[J_TOP],
        job[J_BOT],
        job[J_OBS],
        job[J_PLAYER],
        job[J_H],
        job[J_V],
        tile_work,
        obj_work,
        left,
        right,
        slots,
    )
    return cnt[1] + cnt[2] + cyc


def scratch(slots):
    """Scratch for ``slots`` jobs in flight, one array per scratch argument of
    :func:`render_cycles` (the two work tuples flattened) with a leading slot axis."""

    def zi(*shape):
        return np.zeros((slots,) + shape, dtype=np.int64)

    return (
        zi(256),
        zi(256, mm.N + 1, 7),
        np.zeros((slots, 256, mm.N + 1), dtype=np.uint8),
        zi(3),
        zi(mm.N + 1),
        zi(mm.N + 1),
        zi(mm.N + 1),
        zi(1, 11),
        zi(1),
        zi(SCAN_CAP, TILE_COLS),
        zi(256),
        zi(256),
        zi(mm.NUM_SLOTS),
        zi(3, 4),
        zi(4, 4),
        zi(4),
        zi(4),
        zi(5),
        zi(R_N),
        zi(F_N),
        zi(WORK_VERTICES, 4),
        zi(WORK_VERTICES),
        zi(WORK_VERTICES),
        zi(5),
        zi(R_N),
        zi(F_N),
    )


@njit(cache=True, parallel=True)
def render_many(mem, jobs, vis, model, scr, out):
    """``out[j]`` <- :func:`render_cycles` of ``jobs[j]`` over its table
    ``vis[jobs[j, J_VIS]]``, one slot of ``scr`` per ``prange`` iteration."""
    slots = scr[0].shape[0]
    for t in prange(slots):  # pylint: disable=not-an-iterable
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
            out[j] = render_cycles(
                mem,
                jobs[j],
                vis[jobs[j, J_VIS]],
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


@njit(cache=True)
def render_serial(mem, jobs, vis, model, scr, out):
    """:func:`render_many` on the calling thread and the first slot of ``scr``: the
    launch a lone job should not pay for."""
    tile_work = (scr[14][0], scr[15][0], scr[16][0], scr[17][0], scr[18][0], scr[19][0])
    obj_work = (scr[20][0], scr[21][0], scr[22][0], scr[23][0], scr[24][0], scr[25][0])
    for j in range(jobs.shape[0]):
        out[j] = render_cycles(
            mem,
            jobs[j],
            vis[jobs[j, J_VIS]],
            model,
            scr[0][0],
            scr[1][0],
            scr[2][0],
            scr[3][0],
            scr[4][0],
            scr[5][0],
            scr[6][0],
            scr[7][0],
            scr[8][0],
            scr[9][0],
            scr[10][0],
            scr[11][0],
            tile_work,
            obj_work,
            scr[12][0],
            scr[13][0],
        )


def slot_count(jobs):
    """Scratch slots worth allocating for ``jobs`` jobs: a few per thread so uneven jobs
    average out, never more than the jobs."""
    return max(1, min(jobs, 4 * numba.get_num_threads()))
