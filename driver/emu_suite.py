#!/usr/bin/env python3
"""Play suite boards with ``play_player --emu`` in parallel processes and tabulate them
against the committed offline results: ``python -m driver.emu_suite [boards...]``."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys

from tqdm import tqdm

from sentinel.tests import readme_results, trace_suite

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = trace_suite.SUITE + (9716,)
_THREADS = {
    k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
}


def play(board, out_dir, max_actions):
    """Run one board in its own process; its result JSON (and log beside it)."""
    out = os.path.join(out_dir, f"emu_{board:04d}.json")
    if os.path.exists(out):
        os.unlink(out)
    with open(out[:-5] + ".log", "w") as log:
        subprocess.run(
            [sys.executable, "-m", "driver.play_player", str(board), "--emu"]
            + ["--max-actions", str(max_actions), "--out", out],
            cwd=ROOT,
            env={**os.environ, **_THREADS},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not os.path.exists(out):
        return {"won": "error", "trace": []}  # the run raised: its log says why
    with open(out) as fh:
        return json.load(fh)


def rows(results, offline):
    """One line per board: the emulated run beside the offline record."""
    yield "board  won    actions  energy  wall_s  audit  | offline won actions energy"
    for board, res in sorted(results.items()):
        off = offline.get(str(board), {})
        audit = res.get("lookahead_audit", [])
        same = sum(a[3] == a[4] for a in audit)
        yield (
            f"{board:5d}  {str(res.get('won')):5s}  {len(res.get('trace', [])):7d}  "
            f"{res.get('final_energy')!s:6s}  {res.get('wall_seconds')!s:6s}  "
            f"{same}/{len(audit):<3d} | {off.get('won')!s:5s} {off.get('actions')!s:7s} "
            f"{off.get('energy')}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boards", type=int, nargs="*", default=list(SUITE))
    parser.add_argument("--jobs", type=int, default=len(SUITE))
    parser.add_argument("--max-actions", type=int, default=trace_suite.ACTIONS["phase"])
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "out", "logs", "emu"))
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = {
            pool.submit(play, b, args.out_dir, args.max_actions): b for b in args.boards
        }
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results[futures[fut]] = fut.result()
    with open(readme_results.OFFLINE) as fh:
        offline = json.load(fh)
    for line in rows(results, offline):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
