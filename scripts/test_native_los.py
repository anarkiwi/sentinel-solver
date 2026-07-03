#!/usr/bin/env python3
"""Differential test: native_los.aim_target_native vs the EMULATED ROM aim path,
PARALLELISED across all CPU cores (the only cost is the py65 emulation; the native
side is microseconds, so we fan the emulated reference out with multiprocessing).

The emulated reference is the real C64 routine check_for_line_of_sight_to_tile
$1CDD driven in py65, set up exactly like the action-time player aim
(handle_player_actions $1B40-$1B46): prepare_vector_from_player_sights $1C10 then
$1CDD. We drive it directly (run $1CDD to its RTS) so the comparison is against the
TRUE ROM verdict.

(aim_oracle.aim_target bounds the $1CDD march at 20000 instructions for speed; a
near-horizontal ray sweeping to the board edge would otherwise burn 100k-400k
instructions, returning a TRUNCATED tile -- always los=False -- whereas the native
port runs to completion. We compare against the UNCAPPED ROM and skip the rare
ray that exceeds a generous budget.)

Each worker builds its own CodeEngine for one landscape and processes a chunk of
(h_angle, v_angle, cursor) samples, comparing the emulated ROM aim against the
native port. We Pool.map the chunks, aggregate the agreement rate, and collect any
divergences. The native back-to-back throughput is timed separately (no py65).

Usage: python3 scripts/test_native_los.py [landscape ...]
"""

import sys
import os
import time
import random
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _emu
from code_engine import CodeEngine
import native_los as nl
from game_state import OBJECTS_H_ANGLE, OBJECTS_V_ANGLE

A_SIGHTS_X = 0x0CC6
A_SIGHTS_Y = 0x0CC7
A_OBSERVER = 0x006E
A_DO_LOS = 0x0C6E
A_C58 = 0x0C58
A_TGT_X = 0x0024
A_TGT_Y = 0x0026
PREP_SIGHTS = 0x1C10
CHECK_LOS = 0x1CDD

MARCH_BUDGET = 600000


def rom_aim_uncapped(eng, h, v, cx, cy, ps, maxins=MARCH_BUDGET):
    """The TRUE ROM aim: prepare_vector_from_player_sights $1C10 then run
    check_for_line_of_sight_to_tile $1CDD to its RTS. Returns (tx, ty, los, capped)."""
    m = eng.mem
    cpu = eng.cpu
    sH = m[OBJECTS_H_ANGLE + ps]
    sV = m[OBJECTS_V_ANGLE + ps]
    sCX = m[A_SIGHTS_X]
    sCY = m[A_SIGHTS_Y]
    sOB = m[A_OBSERVER]
    sDO = m[A_DO_LOS]
    sTX = m[A_TGT_X]
    sTY = m[A_TGT_Y]
    s58 = m[A_C58]
    try:
        m[OBJECTS_H_ANGLE + ps] = h & 0xFF
        m[OBJECTS_V_ANGLE + ps] = v & 0xFF
        m[A_SIGHTS_X] = cx & 0xFF
        m[A_SIGHTS_Y] = cy & 0xFF
        m[A_OBSERVER] = ps
        m[A_DO_LOS] &= 0x7F
        m[A_C58] = 0xFF
        eng._call(PREP_SIGHTS, x=ps)
        n = _emu.call(cpu, m, CHECK_LOS, state=eng.state, maxins=maxins)
        eng.state["stop"] = False
        capped = n >= maxins
        los = (not capped) and ((cpu.p & 0x01) == 0)
        tx, ty = m[A_TGT_X], m[A_TGT_Y]
    finally:
        m[OBJECTS_H_ANGLE + ps] = sH
        m[OBJECTS_V_ANGLE + ps] = sV
        m[A_SIGHTS_X] = sCX
        m[A_SIGHTS_Y] = sCY
        m[A_OBSERVER] = sOB
        m[A_DO_LOS] = sDO
        m[A_TGT_X] = sTX
        m[A_TGT_Y] = sTY
        m[A_C58] = s58
    return tx, ty, los, capped


# ---- sample generation ------------------------------------------------------
def make_samples(ls, n_random, seed):
    rng = random.Random(seed + ls)
    cursors = [
        (0x50, 0x5F),
        (0x40, 0x5F),
        (0x60, 0x5F),
        (0x50, 0x4F),
        (0x50, 0x6F),
        (0x48, 0x55),
        (0x58, 0x6A),
    ]
    v_grid = (0x10, 0x28, 0x30, 0x38, 0x40, 0x60, 0x90, 0xB0, 0xD0, 0xE0, 0xF5)
    samples = []
    for h in range(0, 256, 4):
        for v in v_grid:
            samples.append((h, v, 0x50, 0x5F))
    for _ in range(n_random):
        samples.append((rng.randint(0, 255), rng.randint(0, 255), *rng.choice(cursors)))
    return samples


# ---- worker: one chunk on a fresh CodeEngine --------------------------------
_WORKER = {}


def _worker_init(ls):
    """Build the per-process CodeEngine + NativeState once per worker."""
    eng = CodeEngine(ls)
    _WORKER["eng"] = eng
    _WORKER["ps"] = eng.player_slot
    _WORKER["state"] = nl.NativeState.from_mem(eng.mem)
    _WORKER["ls"] = ls


def _worker_chunk(chunk):
    eng = _WORKER["eng"]
    ps = _WORKER["ps"]
    state = _WORKER["state"]
    agree = 0
    total = 0
    los_true_agree = 0
    los_true_total = 0
    capped_n = 0
    disagree = []
    for h, v, cx, cy in chunk:
        rtx, rty, rlos, capped = rom_aim_uncapped(eng, h, v, cx, cy, ps)
        if capped:
            capped_n += 1
            continue
        ntx, nty, nlos = nl.aim_target_native(state, h, v, cx, cy, ps)
        total += 1
        if (rtx, rty, rlos) == (ntx, nty, nlos):
            agree += 1
        else:
            disagree.append((h, v, cx, cy, (rtx, rty, rlos), (ntx, nty, nlos)))
        if rlos:
            los_true_total += 1
            if (rtx, rty) == (ntx, nty) and nlos:
                los_true_agree += 1
    return (agree, total, los_true_agree, los_true_total, capped_n, disagree)


def _chunks(lst, n):
    k = (len(lst) + n - 1) // n
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def run_landscape(ls, n_random=700, seed=1234, nproc=None):
    nproc = nproc or mp.cpu_count()
    eng0 = CodeEngine(ls)
    ps = eng0.player_slot
    px, py = eng0.mem[0x0900 + ps], eng0.mem[0x0980 + ps]
    state0 = nl.NativeState.from_mem(eng0.mem)

    samples = make_samples(ls, n_random, seed)
    chunks = _chunks(samples, nproc)

    t0 = time.perf_counter()
    with mp.Pool(processes=nproc, initializer=_worker_init, initargs=(ls,)) as pool:
        results = pool.map(_worker_chunk, chunks)
    wall = time.perf_counter() - t0

    agree = sum(r[0] for r in results)
    total = sum(r[1] for r in results)
    lta = sum(r[2] for r in results)
    ltt = sum(r[3] for r in results)
    capped_n = sum(r[4] for r in results)
    disagree = [d for r in results for d in r[5]]

    # native back-to-back throughput (no py65), measured single-process
    reps = max(1, 6000 // max(len(samples), 1))
    t0 = time.perf_counter()
    for _ in range(reps):
        for h, v, cx, cy in samples:
            nl.aim_target_native(state0, h, v, cx, cy, ps)
    per_nat = 1e6 * (time.perf_counter() - t0) / (reps * len(samples))

    # emulated per-call wall (parallel): wall / samples * nproc ~ serial-equiv
    per_rom_serial = 1e6 * wall * nproc / max(len(samples), 1)

    print(f"\n== landscape {ls:04d} == player slot {ps} @ ({px},{py})")
    print(
        f"  samples (uncapped) : {total}  (capped/skipped {capped_n})  "
        f"[{nproc} workers, wall {wall:.1f}s]"
    )
    print(
        f"  EXACT (tx,ty,los)  : {agree}/{total} " f"= {100.0*agree/max(total,1):.3f}%"
    )
    print(f"  los=True tiles     : {lta}/{ltt} match " f"= {100.0*lta/max(ltt,1):.2f}%")
    print(
        f"  speed: emulated ~{per_rom_serial:9.1f} us/call   "
        f"NATIVE {per_nat:7.2f} us/call   speedup "
        f"x{per_rom_serial/max(per_nat,1e-9):.0f}"
    )
    if disagree:
        print(f"  DISAGREEMENTS ({len(disagree)}); first 20:")
        for d in disagree[:20]:
            print(
                f"    h={d[0]:02x} v={d[1]:02x} cur=({d[2]:02x},{d[3]:02x}) "
                f"ROM={d[4]} NAT={d[5]}"
            )
    return agree, total, len(disagree)


def main():
    landscapes = [int(a) for a in sys.argv[1:]] or [0, 42, 9999]
    grand_agree = grand_total = grand_dis = 0
    for ls in landscapes:
        a, t, d = run_landscape(ls)
        grand_agree += a
        grand_total += t
        grand_dis += d
    print(
        f"\n==== OVERALL: {grand_agree}/{grand_total} exact "
        f"= {100.0*grand_agree/max(grand_total,1):.3f}%  "
        f"({grand_dis} disagreements) ===="
    )
    sys.exit(0 if grand_dis == 0 else 1)


if __name__ == "__main__":
    main()
