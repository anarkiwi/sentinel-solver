#!/usr/bin/env python3
"""Compare two ``play_player`` runs: each audit's first divergent row, and the saved
tick images byte for byte against the entry image's own differences.

``python -m driver.compare_runs A.json B.json --ticks A_DIR B_DIR``
"""

import argparse
import glob
import json
import os

import numpy as np

AUDITS = (
    "trace",
    "frame_audit",
    "settle_audit",
    "exact_audit",
    "aim_subframes",
    "wait_audit",
    "pan_passes",
)
IO_PAGE = 0xD000  # the CPU view's I/O page and above read chip registers, not RAM


def audits(a, b):
    """``key -> (rows a, rows b, differing rows, first differing index or None)``."""
    out = {}
    for key in AUDITS:
        x, y = a.get(key) or [], b.get(key) or []
        diff = [i for i, (p, q) in enumerate(zip(x, y)) if p != q]
        diff += list(range(min(len(x), len(y)), max(len(x), len(y))))
        out[key] = (len(x), len(y), len(diff), diff[0] if diff else None)
    return out


def _mems(directory):
    for path in sorted(glob.glob(os.path.join(directory, "t*.npz"))):
        with np.load(path) as z:
            yield os.path.basename(path), z["mem"]


def ticks(dir_a, dir_b):
    """Per tick ``(name, RAM bytes differing, new vs tick 0, healed vs tick 0, first
    new addresses)`` over RAM below the I/O page."""
    rows, base = [], None
    for (name, a), (_, b) in zip(_mems(dir_a), _mems(dir_b)):
        diff = a[:IO_PAGE] != b[:IO_PAGE]
        base = diff if base is None else base
        new = np.flatnonzero(diff & ~base)
        healed = int((base & ~diff).sum())
        rows.append((name, int(diff.sum()), len(new), healed, new[:8].tolist()))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--ticks", nargs=2, metavar=("A_DIR", "B_DIR"))
    args = parser.parse_args(argv)
    with open(args.a) as fa, open(args.b) as fb:
        a, b = json.load(fa), json.load(fb)
    for key, (na, nb, nd, first) in audits(a, b).items():
        row = "" if first is None else f" first #{first}: {_row(a, key, first)} vs "
        row += "" if first is None else f"{_row(b, key, first)}"
        print(f"{key:14} {na:4} vs {nb:4} rows, {nd} differ{row}")
    if args.ticks:
        rows = ticks(*args.ticks)
        first = next((r for r in rows if r[2]), None)
        print(f"ticks: {len(rows)} compared; entry image differs in {rows[0][1]} bytes")
        for name, n, new, healed, addrs in rows:
            print(f"  {name}: {n} differ, {new} new, {healed} healed {addrs or ''}")
        print("  first new difference:", first[0] if first else "none")
    return 0


def _row(run, key, i):
    rows = run.get(key) or []
    return rows[i] if i < len(rows) else None


if __name__ == "__main__":
    raise SystemExit(main())
