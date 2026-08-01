#!/usr/bin/env python3
"""Regenerate :mod:`sentinel.writeruns` from the disassembly.

Every ``passcost`` term names the ROM address its cycles are counted from, so walking
that run over the image (:mod:`sentinel.writemap`) says which cycle offsets drive a
write -- all :mod:`sentinel.badline` needs to price a BA window landing in the term.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sentinel import writemap as wm  # noqa: E402  pylint: disable=wrong-import-position

IMG = os.path.join(ROOT, "out", "sentinel_stage2.bin")
SRC = os.path.join(ROOT, "sentinel", "passcost.py")
OUT = os.path.join(ROOT, "sentinel", "writeruns.py")

_TERM = re.compile(r"^([A-Z][A-Z0-9_]*) = \(?\s*(-?\d+)[^#\n]*(?:#(.*))?$", re.M)
_ADDR = re.compile(r"\$([0-9A-F]{4})")

# Terms whose comment names no address, or names one that is not the run's head.
EXTRA_ANCHORS = {
    "EXPOSURE_OTHER": 0x1925,
    "EXPOSURE_SENTRY": 0x1925,
    "EXPOSURE_SENTINEL": 0x1925,
    "UPDATE_DISPATCH_SENTRY": 0x16B5,
    "UPDATE_DISPATCH_SENTINEL": 0x16B5,
    "UPDATE_CURSOR_WRAP": 0x16D9,
    "SOUND_VOICE_LAST": 0x8F08,
    "REMOVE_GROUND": 0x1EEF,
    "SCAN_TEST_UNSEEN": 0x17B7,  # the $17B2 slot test: every branch of it runs
    "SCAN_TEST_FULL": 0x17B7,  # from $17B7, so their comments name a later address
    "SCAN_TEST_OTHER": 0x17B7,
    "SCAN_TEST_PARTIAL": 0x17B7,
    "WIDE_STEEP_COLUMN": 0x3115,  # the $3115 BCC not taken; its comment names $0076
    "WIDE_AREA": 0x31A4,  # $31A4's other branches; their comments name $0041
    "WIDE_AREA_LEFT": 0x31A4,
    "WIDE_AREA_RIGHT": 0x31A4,
}
# Constants that are not cost terms: their comment's address is incidental.
NOT_A_RUN = frozenset(
    ("BADLINES_PER_FRAME", "SHORT_IRQS_PER_FRAME", "SHORT_IRQ_WRAP", "BADLINE_FRAME")
)
_PRND_LAPS = [True] * 7 + [False]  # $31CD loads eight; the last $31E9 BNE falls through
# Runs whose loop the walk cannot take alone: the branches the charging term decides.
BRANCHES = {
    0x31CA: _PRND_LAPS,
    0x16D6: _PRND_LAPS,  # $16D6 JSR $31CA
    0x1272: _PRND_LAPS,  # $1272 JSR $31CA, the masked tile-coordinate draw
    0x1279: [True] + _PRND_LAPS,  # ... its rejection loops back to $1272
    0x1289: [False, True],  # the pass head's $128E BMI and $1294 BPL
}


def anchors(source):
    """``{anchor: max cycles counted from it}`` over every ``passcost`` term."""
    out = {}
    for name, value, comment in _TERM.findall(source):
        value = int(value)
        if value <= 0 or name in NOT_A_RUN:
            continue
        found = _ADDR.search(comment or "")
        anchor = EXTRA_ANCHORS.get(name, int(found.group(1), 16) if found else None)
        if anchor is None:
            continue
        out[anchor] = max(out.get(anchor, 0), value)
    return out


def write_offsets(image, anchor, cycles):
    """Cycle offsets, from ``anchor``, at which the run drives a write."""
    run = wm.walk(image, anchor, cycles, BRANCHES.get(anchor, ()))
    out = []
    for offset, _pc, op in run:
        out.extend(offset + c for c in wm.OP_WRITE_CYCLES[op] if offset + c < cycles)
    return tuple(out)


def boundary_offsets(image, anchor, cycles):
    """Cycle offsets, from ``anchor``, at which an instruction of the run starts.

    Walked one whole instruction past the run so the last one inside it has its own
    end as a boundary; jennings' table names how long that can be.
    """
    over = max(wm.OP_CYCLES) + 2  # a taken branch across a page is its own two more
    run = wm.walk(image, anchor, cycles + over, BRANCHES.get(anchor, ()))
    return tuple(offset for offset, _pc, _op in run if offset <= cycles)


def table(image, source):
    """``[(anchor, cycles, writes, starts)]`` for every anchored term, in address order."""
    return [
        (
            anchor,
            cycles,
            write_offsets(image, anchor, cycles),
            boundary_offsets(image, anchor, cycles),
        )
        for anchor, cycles in sorted(anchors(source).items())
    ]


HEADER = '''"""Where each counted ROM run writes, and where its instructions start.

A ``passcost`` term is a run counted from a ROM address, so a BA window landing ``d``
cycles into one is priced by the write cycles at ``d`` (:mod:`sentinel.badline`).
Do not edit by hand: ``python -m driver.writeruns``.
"""

import numpy as np

'''

BODY = '''

def _dense():
    """``(start, length, runs, owed)``: the write run and the cycles to the next
    instruction, at each cycle of each counted run."""
    start = np.full(0x10000, -1, dtype=np.int32)
    length = np.zeros(0x10000, dtype=np.int32)
    runs = np.zeros(sum(LENGTH), dtype=np.int8)
    owed = np.zeros(sum(LENGTH), dtype=np.int8)
    at = 0
    for anchor, cycles, offsets, starts in zip(ANCHOR, LENGTH, RUNS, STARTS):
        start[anchor], length[anchor] = at, cycles
        for offset in offsets:
            runs[at + offset] = 1
        for offset in sorted(offsets, reverse=True):
            if offset + 1 < cycles and runs[at + offset + 1]:
                runs[at + offset] = runs[at + offset + 1] + 1
        bounds = sorted(set(starts))
        for first, nxt in zip(bounds, bounds[1:]):
            for offset in range(first + 1, min(nxt, cycles)):
                owed[at + offset] = nxt - offset
        at += cycles
    return start, length, runs, owed


START, LENGTH_AT, RUN, OWED = _dense()
'''


def render(rows):
    """The generated module's source."""
    out = [HEADER, "ANCHOR = (\n"]
    out.extend(f"    0x{a:04X},\n" for a, _, _, _ in rows)
    out.append(")\nLENGTH = (\n")
    out.extend(f"    {c},\n" for _, c, _, _ in rows)
    out.append(")\nRUNS = (\n")
    for anchor, _, offsets, _starts in rows:
        out.append(f"    {offsets!r},  # ${anchor:04X}\n")
    out.append(")\nSTARTS = (\n")
    for anchor, _, _offsets, starts in rows:
        out.append(f"    {starts!r},  # ${anchor:04X}\n")
    out.append(")\n")
    out.append(BODY)
    return "".join(out)


def main(argv=None):
    """Rewrite ``sentinel/writeruns.py`` from the image."""
    ap = argparse.ArgumentParser(description="regenerate the per-term write-cycle map")
    ap.add_argument("--image", default=IMG)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    with open(args.image, "rb") as fh:
        image = fh.read()
    with open(SRC, encoding="utf-8") as fh:
        rows = table(image, fh.read())
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render(rows))
    print(
        f"[writeruns] {len(rows)} anchors, {sum(len(r[2]) for r in rows)} writes, "
        f"{sum(len(r[3]) for r in rows)} instructions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
