#!/usr/bin/env python3
"""Play the suite boards with the search player on the emulator, in parallel processes,
and tabulate them against the committed results: ``python -m driver.emu_suite
[boards...]``; ``--out`` records them, ``--readme`` renders the README results block.

Other ``play_player`` flags pass through in ``--flag=value`` form.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys

from tqdm import tqdm

from driver.search_player import PRESETS
from sentinel.tests import readme_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = (0, 42, 60, 110, 298, 321, 335, 373, 1442, 9716, 9795, 9950)
MAX_ACTIONS = 120
_THREADS = {
    k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
}


def digest(trace):
    """Hex SHA-256 of the canonical JSON of an action trace."""
    blob = json.dumps(trace, separators=(",", ":"), default=int).encode()
    return hashlib.sha256(blob).hexdigest()


def summary(result, preset):
    """The committed record of one ``play_player`` result."""
    return {
        "landscape": result["landscape"],
        "enemies": result.get("enemies"),
        "actions": len(result["trace"]),
        "energy": result.get("final_energy"),
        "frames": result.get("frames"),
        "won": bool(result.get("won")),
        "preset": preset,
        "digest": digest(result["trace"]),
    }


def play(board, out_dir, max_actions, extra=()):
    """Run one board in its own process (``extra``: more ``play_player`` flags); its
    result JSON (and log beside it)."""
    out = os.path.join(out_dir, f"emu_{board:04d}.json")
    if os.path.exists(out):
        os.unlink(out)
    with open(out[:-5] + ".log", "w") as log:
        subprocess.run(
            [sys.executable, "-m", "driver.play_player", str(board), "--emu"]
            + ["--max-actions", str(max_actions), "--out", out]
            + list(extra),
            cwd=ROOT,
            env={**os.environ, **_THREADS},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not os.path.exists(out):
        return {"landscape": board, "won": "error", "trace": []}
    with open(out) as fh:
        return json.load(fh)


def rows(results, committed):
    """One line per board: the emulated run beside its committed record."""
    yield "board  won    actions  energy  wall_s  audit  cores | committed won actions energy"
    for board, res in sorted(results.items()):
        rec = committed.get(str(board), {})
        audit = res.get("lookahead_audit", [])
        same = sum(a[3] == a[4] for a in audit)
        stats = res.get("search", {}).get("stats", {})
        cores = stats.get("cpu_seconds", 0) / stats["seconds"] if stats else None
        yield (
            f"{board:5d}  {str(res.get('won')):5s}  {len(res.get('trace', [])):7d}  "
            f"{res.get('final_energy')!s:6s}  {res.get('wall_seconds')!s:6s}  "
            f"{same}/{len(audit):<3d}  {'-' if cores is None else f'{cores:.1f}':5s} | "
            f"{rec.get('won')!s:5s} {rec.get('actions')!s:7s} {rec.get('energy')}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boards", type=int, nargs="*", default=list(SUITE))
    parser.add_argument("--jobs", type=int, default=len(SUITE))
    parser.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "out", "logs", "emu"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="wide")
    parser.add_argument(
        "--tabulate", action="store_true", help="tabulate --out-dir's results, no play"
    )
    parser.add_argument("--out", help="write the boards' records here as JSON")
    parser.add_argument(
        "--readme", action="store_true", help="render the README results block"
    )
    args, extra = parser.parse_known_args(argv)
    extra = ["--preset", args.preset] + extra
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    if args.tabulate:
        for board in args.boards:
            path = os.path.join(args.out_dir, f"emu_{board:04d}.json")
            if os.path.exists(path):
                with open(path) as fh:
                    results[board] = json.load(fh)
        args.boards = []
    with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as pool:
        futures = {
            pool.submit(play, b, args.out_dir, args.max_actions, extra): b
            for b in args.boards
        }
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results[futures[fut]] = fut.result()
    for line in rows(results, readme_results.load(readme_results.SEARCH)):
        print(line)
    if args.out:
        records = {str(b): summary(r, args.preset) for b, r in results.items()}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(records, fh, sort_keys=True, indent=1)
            fh.write("\n")
    if args.readme:
        readme_results.rewrite()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
