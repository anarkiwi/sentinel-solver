#!/usr/bin/env python3
"""Compare two ``play_player`` runs: each audit's first divergent row.

``python -m driver.compare_runs A.json B.json``
"""

import argparse
import json

AUDITS = ("trace", "lookahead_audit", "live_audit")


def audits(a, b):
    """``key -> (rows a, rows b, differing rows, first differing index or None)``."""
    out = {}
    for key in AUDITS:
        x, y = a.get(key) or [], b.get(key) or []
        diff = [i for i, (p, q) in enumerate(zip(x, y)) if p != q]
        diff += list(range(min(len(x), len(y)), max(len(x), len(y))))
        out[key] = (len(x), len(y), len(diff), diff[0] if diff else None)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args(argv)
    with open(args.a) as fa, open(args.b) as fb:
        a, b = json.load(fa), json.load(fb)
    for key, (na, nb, nd, first) in audits(a, b).items():
        row = "" if first is None else f" first #{first}: {_row(a, key, first)} vs "
        row += "" if first is None else f"{_row(b, key, first)}"
        print(f"{key:14} {na:4} vs {nb:4} rows, {nd} differ{row}")
    return 0


def _row(run, key, i):
    rows = run.get(key) or []
    return rows[i] if i < len(rows) else None


if __name__ == "__main__":
    raise SystemExit(main())
