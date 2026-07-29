#!/usr/bin/env python3
"""Solve the hardest-128 landscapes, one subprocess each, and record the outcome.

Each board also gets a cheap start-state probe (landable tiles, climb candidates) so a
loss can be classified as paralysis (nothing to do at move 0) rather than a death.
Results stream to JSONL so a long run can be watched or resumed.
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = "/scratch/anarkiwi/cbm/sentinel-solver"
CODES = os.path.join(ROOT, "out", "hardest_128.txt")
OUT = os.path.join(ROOT, "out", "hardest128_results.jsonl")
THREADS = os.environ.get("PER_BOARD_THREADS", "3")
WORKERS = int(os.environ.get("WORKERS", "20"))
TIMEOUT = int(os.environ.get("BOARD_TIMEOUT", "900"))
RESULT = re.compile(
    r"landscape (\d+): (WON|lost) in (\d+) actions / (\d+) frames, energy (\d+), dead=(\w+)"
)

PROBE = """
import json, sys
from sentinel.phase_player import PhasePlayer
from sentinel.playerbase import _Views
from sentinel.game import Game
p = PhasePlayer(Game.typed(int(sys.argv[1])))
v = _Views(p.st)
print(json.dumps({
    "landable": len(list(v.band())),
    "cands": len(p._climb_candidates(v, True)),
    "start_eye": round(p.st.eye_z(), 3),
}))
"""


def env():
    e = dict(os.environ)
    e.update(
        PYTHONPATH=ROOT,
        PYTHONUNBUFFERED="1",
        NUMBA_NUM_THREADS=THREADS,
        OMP_NUM_THREADS=THREADS,
    )
    return e


def one(code):
    t0 = time.time()
    row = {"code": code}
    try:
        pr = subprocess.run(
            [sys.executable, "-c", PROBE, str(code)],
            cwd=ROOT, env=env(), capture_output=True, text=True, timeout=300,
        )
        row.update(json.loads(pr.stdout.strip().splitlines()[-1]))
    except Exception as exc:  # pragma: no cover - diagnostic path
        row["probe_error"] = str(exc)[:120]
    try:
        rn = subprocess.run(
            [sys.executable, "-m", "sentinel.phase_player", str(code), "--quiet"],
            cwd=ROOT, env=env(), capture_output=True, text=True, timeout=TIMEOUT,
        )
        m = RESULT.search(rn.stdout)
        if m:
            row.update(
                won=m.group(2) == "WON", actions=int(m.group(3)),
                frames=int(m.group(4)), energy=int(m.group(5)),
                dead=m.group(6) == "True",
            )
        else:
            row["error"] = (rn.stdout + rn.stderr)[-200:]
    except subprocess.TimeoutExpired:
        row["timeout"] = TIMEOUT
    row["wall"] = round(time.time() - t0, 1)
    return row


def main():
    codes = [int(c) for c in open(CODES).read().split()]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["code"] for l in open(OUT) if l.strip()}
    todo = [c for c in codes if c not in done]
    print(f"{len(codes)} boards, {len(done)} already done, {len(todo)} to run", flush=True)
    with open(OUT, "a") as fh, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for row in pool.map(one, todo):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"ls{row['code']:<5} won={row.get('won')} acts={row.get('actions')} "
                f"landable={row.get('landable')} wall={row.get('wall')}s"
                f"{' TIMEOUT' if 'timeout' in row else ''}",
                flush=True,
            )


if __name__ == "__main__":
    main()
