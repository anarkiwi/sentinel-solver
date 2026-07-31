#!/usr/bin/env python3
"""Measure the VIC-II badline steal live and check it against its derivation.

A badline pulls BA low three cycles before the VIC needs the bus and the 6510 runs on
to its first READ cycle, so the steal is ``43 - (write cycles at the window)``, 43 being
the 40 c-accesses plus the AEC lag.  ``python -m driver.badline 9795 --captures 10``.
"""

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from sentinel import (
    badline as bl,
    memmap as mm,
    passcost,
)  # noqa: E402  pylint: disable=wrong-import-position
from driver import clock, core  # noqa: E402  pylint: disable=wrong-import-position

REG_X, REG_Y, REG_PC = 1, 2, 3
REG_RASTER, REG_RASTER_CYCLE = 53, 54

# addressing modes that spend an extra cycle when the index crosses a page.
_ABS_X = frozenset([0x1D, 0x3D, 0x5D, 0x7D, 0xBD, 0xDD, 0xFD, 0xBC])
_ABS_Y = frozenset([0x19, 0x39, 0x59, 0x79, 0xB9, 0xD9, 0xF9, 0xBE])
_IND_Y = frozenset([0x11, 0x31, 0x51, 0x71, 0xB1, 0xD1, 0xF1])
_BRANCH = frozenset([0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0])


def cost_class(rec, nxt, zeropage):
    """The class whose MINIMUM observed delta is this instruction's unstolen cost.

    A steal is never negative, so the minimum over a class is its true cost -- provided
    the class splits the taken branch and the crossed index page from the rest."""
    op = rec.op
    if op in _BRANCH:
        after = (rec.registers[REG_PC] + 2) & 0xFFFF
        if nxt.registers[REG_PC] == after:
            return (op, 0, 0)
        offset = rec.p1 - 256 if rec.p1 > 127 else rec.p1
        dest = (after + offset) & 0xFFFF
        return (op, 1, 1 if (dest >> 8) != (after >> 8) else 0)
    if op in _ABS_X or op in _ABS_Y:
        base = rec.p1 | (rec.p2 << 8)
        index = rec.registers[REG_X if op in _ABS_X else REG_Y]
    elif op in _IND_Y:
        base = zeropage[rec.p1] | (zeropage[(rec.p1 + 1) & 0xFF] << 8)
        index = rec.registers[REG_Y]
    else:
        return (op, 0, 0)
    return (op, 0, 1 if ((base + index) >> 8) != (base >> 8) else 0)


def steals(cap):
    """One capture as ``(frame_position, steal, op, pc, frame)`` per instruction."""
    zeropage, history, origin = cap["zeropage"], cap["history"], cap["origin"]
    nominal, classed = {}, []
    for rec, nxt in zip(history, history[1:]):
        key = cost_class(rec, nxt, zeropage)
        delta = nxt.cycle - rec.cycle
        if delta < nominal.get(key, delta + 1):
            nominal[key] = delta
        classed.append((rec, key, delta))
    return [
        (
            (rec.cycle - origin) % passcost.PAL_FRAME_CYCLES,
            delta - nominal[key],
            rec.op,
            rec.registers[REG_PC],
            (rec.cycle - origin) // passcost.PAL_FRAME_CYCLES,
        )
        for rec, key, delta in classed
    ]


def solve_window(samples):
    """The one line-cycle at which every sampled steal equals its derived value."""
    for line_cycle in range(bl.LINE_CYCLES):
        windows = bl.window_positions(line_cycle)
        if all(bl.steal(s[2], s[0], windows) == s[1] for s in samples):
            return line_cycle
    return None


def capture(bm, count, frozen=False):
    """``count`` cpuhistory windows, each frame-anchored at the $9630 marker."""
    out = []
    with bm.halted():
        bm.run_until_pc(clock.FRAME_PC, timeout=6.0)
        if not frozen:
            acted = bm.mem_get(mm.PLAYER_NOT_ACTED, mm.PLAYER_NOT_ACTED)[0]
            bm.mem_set(mm.PLAYER_NOT_ACTED, bytes([acted & 0x7F]))
        for _ in range(count):
            bm.advance_instructions(1)
            bm.run_until_pc(clock.FRAME_PC, timeout=6.0)
            registers = bm.registers_get()
            zeropage = list(bm.mem_get(0x0000, 0x00FF))
            history = bm.cpuhistory_get(0xFFFF)
            bm.advance_instructions(1)  # stamp the marker to fix the raster origin
            marker = bm.cpuhistory_get(2)[-1].cycle
            origin = (
                marker
                - registers[REG_RASTER] * bl.LINE_CYCLES
                - registers[REG_RASTER_CYCLE]
            ) % passcost.PAL_FRAME_CYCLES
            out.append({"origin": origin, "zeropage": zeropage, "history": history})
    return out


def analyse(captures):
    """Steal histogram, per-frame totals, the derived window and the $9630 jitter."""
    samples, anchors = [], collections.Counter()
    for index, cap in enumerate(captures):
        for position, steal, op, pc, frame in steals(cap):
            if steal >= passcost.BADLINE_STEAL - 2:
                samples.append((position, steal, op, pc, (index, frame)))
            if pc == clock.FRAME_PC:
                anchors[position] += 1
    totals = collections.Counter()
    for sample in samples:
        totals[sample[4]] += sample[1]
    line_cycle = solve_window(samples)
    return {
        "badlines": len(samples),
        "steal": dict(sorted(collections.Counter(s[1] for s in samples).items())),
        "window_line_cycle": line_cycle,
        "derived_exactly": line_cycle is not None,
        "per_frame_total": dict(sorted(collections.Counter(totals.values()).items())),
        "anchor_9630": dict(sorted(anchors.items())),
        "samples": sorted(
            [pos, op, steal, n]
            for (pos, op, steal), n in collections.Counter(
                (s[0], s[2], s[1]) for s in samples
            ).items()
        ),
        "writers": {
            f"${pc:04X} ${op:02X} {steal}": n
            for (pc, op, steal), n in collections.Counter(
                (s[3], s[2], s[1]) for s in samples if s[1] < passcost.BADLINE_STEAL
            ).most_common()
        },
    }


def main(argv=None):
    """Boot a board, capture cpuhistory and report the exact steal."""
    ap = argparse.ArgumentParser(description="exact live VIC-II badline steal")
    ap.add_argument("landscape", nargs="?", default="9795")
    ap.add_argument("--captures", type=int, default=10)
    ap.add_argument(
        "--frozen", action="store_true", help="leave the enemy clock frozen"
    )
    ap.add_argument("--out", help="write the report as JSON")
    args = ap.parse_args(argv)
    os.environ.setdefault("NO_RECORD", "1")

    drv = core.SentinelDriver.boot(record_mount=os.path.join(core.boot.ROOT, "renders"))
    try:
        try:
            drv.bm.resource_set_int("WarpMode", 1)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[badline] warp resource set skipped: {exc}")
        drv.enter_landscape(int(args.landscape.zfill(4), 16))
        report = analyse(capture(drv.bm, args.captures, frozen=args.frozen))
    finally:
        drv.close()
    report["landscape"] = args.landscape
    report["frozen"] = bool(args.frozen)
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return 0 if report["derived_exactly"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
