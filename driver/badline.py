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
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from sentinel import (
    badline as bl,
    memmap as mm,
    passcost,
    writemap as wm,
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


def instructions(cap):
    """One capture as ``(position, cycle, op, pc, nominal, steal, frame)`` per
    instruction, positions taken modulo the PAL frame from the capture's own origin."""
    zeropage, history, origin = cap["zeropage"], cap["history"], cap["origin"]
    nominal, classed = {}, []
    for rec, nxt in zip(history, history[1:]):
        key = cost_class(rec, nxt, zeropage)
        delta = nxt.cycle - rec.cycle
        if delta < nominal.get(key, delta + 1):
            nominal[key] = delta
        classed.append((rec, key, delta))
    return (
        [
            (
                (rec.cycle - origin) % passcost.PAL_FRAME_CYCLES,
                rec.cycle,
                rec.op,
                rec.registers[REG_PC],
                nominal[key],
                delta - nominal[key],
                (rec.cycle - origin) // passcost.PAL_FRAME_CYCLES,
            )
            for rec, key, delta in classed
        ],
        nominal,
    )


def steals(cap):
    """One capture as ``(frame_position, steal, op, pc, frame)`` per instruction."""
    rows, _ = instructions(cap)
    return [(r[0], r[5], r[2], r[3], r[6]) for r in rows]


def solve_window(samples):
    """The one line-cycle at which every sampled steal equals its derived value."""
    for line_cycle in range(bl.LINE_CYCLES):
        windows = bl.window_positions(line_cycle)
        if all(bl.steal(s[2], s[0], windows) == s[1] for s in samples):
            return line_cycle
    return None


ROUTINES = (
    (0x0D03, "$0D03 mul/div/trig"),
    (0x1010, "other"),
    (0x1289, "$1289 pass head/tail"),
    (0x12C8, "other"),
    (0x130C, "$130C cooldown tick"),
    (0x1336, "other"),
    (0x16B5, "$16B5 dispatch/prnd/cursor"),
    (0x16E6, "$16E6 body"),
    (0x1887, "$1887 see"),
    (0x191F, "$191F exposure"),
    (0x1973, "$1973 meanie"),
    (0x1A08, "$1A08 reduce/discharge/tilescan"),
    (0x1B18, "other"),
    (0x1CBB, "$1CBB march"),
    (0x1E3F, "$1E3F object walk"),
    (0x1F16, "$1F16 place/redraw/span"),
    (0x2130, "other"),
    (0x2BA8, "$2BA8 tile address"),
    (0x2BD0, "other"),
    (0x31CA, "$31CA prnd"),
    (0x3200, "other"),
    (0x3470, "$3470 tune/sound call"),
    (0x3600, "other"),
    (0x8401, "$8401 relative"),
    (0x8600, "other"),
    (0x8CF0, "$8CF0 kbd scan + note tick"),
    (0x9200, "other"),
    (0x9287, "$9287 angle"),
    (0x93B0, "other"),
    (0x9508, "$9508 status bar"),
    (0x95E9, "$95E9 raster IRQ"),
    (0x96A1, "other"),
)
MODEL_SOURCES = ("passcost.py", "los.py", "relative.py", "enemies.py")
HANDLER_LOW, HANDLER_HIGH = 0x95E9, 0x96A0
IMG = os.path.join(os.path.dirname(HERE), "out", "sentinel_stage2.bin")


def routine_of(pc):
    """The ROM routine ``pc`` falls in, by the greatest entry at or below it."""
    name = "other"
    for start, label in ROUTINES:
        if pc >= start:
            name = label
    return name


def model_anchors():
    """Every ROM address the cost model counts a run from: the $XXXX it names."""
    root = os.path.join(os.path.dirname(HERE), "sentinel")
    found = set()
    for name in MODEL_SOURCES:
        with open(os.path.join(root, name), encoding="utf-8") as fh:
            found.update(
                int(m, 16) for m in re.findall(r"\$([0-9A-F]{4})\b", fh.read())
            )
    return found


def marker_law(rows):
    """$9630's frame position against the instruction boundary the IRQ was taken at.

    The interrupted instruction's own measured overrun is the 6510's 7-cycle interrupt
    sequence, so its unstolen cost alone gives the boundary.
    """
    gaps, boundaries = collections.Counter(), collections.Counter()
    for i, row in enumerate(rows):
        if row[3] != clock.FRAME_PC:
            continue
        j = i - 1
        while j >= 0 and HANDLER_LOW <= rows[j][3] <= HANDLER_HIGH:
            j -= 1
        if j < 0:
            continue
        boundary = rows[j][0] + rows[j][4]
        boundaries[boundary] += 1
        gaps[f"{row[0] - boundary} {wm.MNEMONIC[rows[j][2]]} {rows[j][4]}"] += 1
    return gaps, boundaries


def irq_entries(rows):
    """Every raster IRQ's entry, against the instruction boundary it was taken at.

    All five a frame vector through $95E9, so an entry is a step from outside the
    handler into it; the gap is the 6510's own interrupt sequence.  Each is keyed by
    the interrupted instruction's mnemonic AND its unstolen cost, which for a branch
    names the case: 2 not taken, 3 taken, 4 taken across a page.
    """
    gaps, lines = collections.Counter(), collections.Counter()
    for prev, row in zip(rows, rows[1:]):
        if row[3] != HANDLER_LOW or prev[3] == HANDLER_LOW:
            continue
        boundary = prev[0] + prev[4]
        gaps[f"{row[0] - boundary} {wm.MNEMONIC[prev[2]]} {prev[4]}"] += 1
        lines[boundary // bl.LINE_CYCLES] += 1
    return gaps, lines


def window_hits(rows, windows):
    """``(row index, window, steal)`` for every instruction a BA window falls in."""
    out = []
    for index, row in enumerate(rows):
        position, extent = row[0], row[4] + row[5]
        for window in windows:
            if position <= window < position + extent:
                out.append((index, window, bl.steal(row[2], position, windows)))
    return out


def anchor_resolution(rows, hits, anchors, into):
    """Bin every window by the offset from the model's last anchor before it.

    ``into`` accumulates ``keys`` (offset -> the steals seen there), ``routine`` and
    the covered/decoded counts across captures.
    """
    anchor_at, last = {}, None
    for index, row in enumerate(rows):
        anchor_at[index] = last
        if row[3] in anchors:
            last = (row[3], row[1])
    for index, window, steal in hits:
        into["routine"][routine_of(rows[index][3])] += 1
        into["count"]["windows"] += 1
        anchor = anchor_at[index]
        if anchor is None:
            continue
        into["count"]["anchor_covered"] += 1
        offset = rows[index][1] + (window - rows[index][0]) - anchor[1]
        into["keys"][(anchor[0], offset)].add(steal)
        image = into["image"]
        if image is not None and wm.steal_at(image, anchor[0], offset) == steal:
            into["count"]["statically_decoded"] += 1


def op_cycle_check(nominals):
    """Live minimum cost per (opcode, branch taken, page crossed) against the table."""
    bad = {}
    for (op, taken, page), measured in nominals.items():
        want = wm.OP_CYCLES[op] + taken + page
        if measured != want:
            bad[f"${op:02X} t{taken} p{page}"] = [measured, want]
    return bad


def frame_steal_check(rows, windows):
    """``badline.frame_steal`` over the live stream against the measured frame total."""
    per_frame = collections.defaultdict(list)
    for row in rows:
        per_frame[row[6]].append((row[0], row[2]))
    measured = collections.Counter()
    for index, _, steal in window_hits(rows, windows):
        measured[rows[index][6]] += steal
    agree = total = 0
    for frame, stream in per_frame.items():
        if measured[frame] < passcost.BADLINES_PER_FRAME * bl.MIN_STEAL:
            continue
        total += 1
        agree += bl.frame_steal(sorted(stream), windows) == measured[frame]
    return {"complete_frames": total, "frame_steal_agrees": agree}


def _capture_one(bm):
    """One cpuhistory window, frame-anchored at the $9630 marker."""
    with bm.halted():
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
    return {"origin": origin, "zeropage": zeropage, "history": history}


def capture(bm, count, frozen=False, warmup=0, stride=0):
    """``count`` cpuhistory windows, each frame-anchored at the $9630 marker.

    ``warmup`` frames are run first, so a board can be caught mid-march.  Captures are
    one frame apart and each spans several, so ``stride`` frames between them samples
    the run rather than its head -- what a mean over the frame total needs.
    """
    with bm.halted():
        bm.run_until_pc(clock.FRAME_PC, timeout=6.0)
        if not frozen:
            acted = bm.mem_get(mm.PLAYER_NOT_ACTED, mm.PLAYER_NOT_ACTED)[0]
            bm.mem_set(mm.PLAYER_NOT_ACTED, bytes([acted & 0x7F]))
    if warmup:
        clock.run_frames(bm, warmup)
    out = []
    for index in range(count):
        if index and stride:
            clock.run_frames(bm, stride)
        out.append(_capture_one(bm))
    return out


def resolution(captures):
    """The instruction-resolution report: the marker law, and what the anchors reach."""
    image = None
    if os.path.exists(IMG):
        with open(IMG, "rb") as fh:
            image = fh.read()
    windows = bl.window_positions()
    known = model_anchors()
    gaps, boundaries = collections.Counter(), collections.Counter()
    entry_gaps, entry_lines = collections.Counter(), collections.Counter()
    into = {
        "image": image,
        "keys": collections.defaultdict(set),
        "routine": collections.Counter(),
        "count": collections.Counter(),
    }
    frames, nominals = collections.Counter(), {}
    for cap in captures:
        rows, nominal = instructions(cap)
        cap_gaps, cap_boundaries = marker_law(rows)
        gaps.update(cap_gaps)
        boundaries.update(cap_boundaries)
        cap_entry_gaps, cap_entry_lines = irq_entries(rows)
        entry_gaps.update(cap_entry_gaps)
        entry_lines.update(cap_entry_lines)
        for key, value in nominal.items():
            nominals[key] = min(value, nominals.get(key, value))
        anchor_resolution(rows, window_hits(rows, windows), known, into)
        frames.update(frame_steal_check(rows, windows))
    ambiguous = {k: sorted(v) for k, v in into["keys"].items() if len(v) > 1}
    return {
        "marker_gap": dict(gaps.most_common()),
        "irq_boundary": dict(sorted(boundaries.items())),
        "irq_entry_gap": dict(entry_gaps.most_common()),
        "irq_entry_line": dict(sorted(entry_lines.items())),
        "anchor_keys": len(into["keys"]),
        "ambiguous_keys": len(ambiguous),
        "ambiguous": {
            f"${k[0]:04X}+{k[1]}": v for k, v in list(ambiguous.items())[:12]
        },
        "window_routine": dict(into["routine"].most_common()),
        "op_cycle_mismatch": op_cycle_check(nominals),
        **dict(into["count"]),
        **dict(frames),
    }


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
        **resolution(captures),
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
    ap.add_argument(
        "--warmup", type=int, default=0, help="frames to run before capturing"
    )
    ap.add_argument(
        "--stride", type=int, default=0, help="frames to run between captures"
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
        report = analyse(
            capture(
                drv.bm,
                args.captures,
                frozen=args.frozen,
                warmup=args.warmup,
                stride=args.stride,
            )
        )
    finally:
        drv.close()
    report["landscape"] = args.landscape
    report["frozen"] = bool(args.frozen)
    report["warmup"] = args.warmup
    report["stride"] = args.stride
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return 0 if report["derived_exactly"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
