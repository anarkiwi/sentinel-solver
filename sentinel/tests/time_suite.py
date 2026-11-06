"""Time the suite boards on a warm process: ``python -m sentinel.tests.time_suite
[--player phase|beam] [--width N] [--profile BOARD] [--capture] [boards...]``.

Plays board 0 first and compiles the device pricing kernel, then prints each board's
seconds, verdict, action count and frames.  ``--profile`` also runs one board under
cProfile and prints where its time went: planning, advancing, landing sweeps and the CUDA
driver waits.  ``--capture`` brackets the timed boards with the CUDA profiler start and
stop, for ``nsys profile --capture-range=cudaProfilerApi``.

``--grid player[:width]@backend,...`` runs every configuration over the boards and prints
one combined table with per-configuration totals and the load average at both ends: one
child process per backend (``--cuda-python`` is the interpreter of the device
environment), each warmed once and running its configurations back to back.
"""

import argparse
import cProfile
import json
import os
import pstats
import subprocess
import sys
import tempfile
import time

import numpy as np
from tqdm import tqdm

from sentinel import kern, projector
from sentinel.game import Game
from sentinel.tests import trace_suite

SPLIT = (
    ("planning", "beam_player.py", "plan"),
    ("advance", "engine.py", "advance"),
    ("landing", "batch.py", "landings"),
)


def split(prof):
    """``{part: seconds}``: the cumulative time of each of :data:`SPLIT`'s functions and
    the self time spent in numba's CUDA driver calls."""
    stats = pstats.Stats(prof).stats
    out = {}
    for part, file, func in SPLIT:
        out[part] = sum(
            v[3] for k, v in stats.items() if k[0].endswith(file) and k[2] == func
        )
    out["cuda api waits"] = sum(
        v[2] for k, v in stats.items() if k[2] == "safe_cuda_api_call"
    )
    out["total"] = sum(v[2] for v in stats.values())
    return out


def warm_device():
    """Compile the device pricing kernel, which the suite's first large launch would
    otherwise build mid-board."""
    if kern.backend() != "cuda":
        return
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    st = Game.typed(0).state
    jobs, _obs = projector._cost_jobs(  # pylint: disable=protected-access
        st, [(0, 0, st.player, projector.PLAY_MODE, None, None, 0)]
    )
    mems = np.frombuffer(st.mem, dtype=np.uint8)[None]
    vis = projector._occlusion_memo(st, st.player)[1][None]  # pylint: disable=W0212
    model = projector._object_backend()[1]  # pylint: disable=protected-access
    kern_cuda.render_cycles_many(mems, jobs, vis, model)


def timed(play, boards):
    """``{board: (seconds, won, actions, frames)}`` of ``play`` over ``boards``."""
    out = {}
    for number in tqdm(boards, unit="board"):
        start = time.perf_counter()
        rec = play(number)
        out[number] = [
            time.perf_counter() - start,
            int(rec["won"]),
            len(rec["trace"]),
            rec["frames"],
        ]
    return out


def parse_grid(spec):
    """``[(player, width, backend)]`` of ``player[:width]@backend`` items."""
    out = []
    for item in spec.split(","):
        who, backend = item.split("@")
        player, _, width = who.partition(":")
        out.append((player, int(width) if width else None, backend))
    return out


def run_configs(configs, boards):
    """``{label: timed}`` of each ``(player, width, backend)`` of this process's backend
    over ``boards``, after one warm-up board and the device kernel compile."""
    mine = [c for c in configs if c[2] == kern.backend()]
    if not mine:
        return {}
    trace_suite.play(0, player=mine[0][0], width=mine[0][1])
    warm_device()
    return {
        f"{player}{width or ''}@{backend}": timed(
            lambda n, p=player, w=width: trace_suite.play(n, player=p, width=w), boards
        )
        for player, width, backend in mine
    }


def combined(results, boards):
    """The table text of ``results``: a row per board, a column per configuration (seconds,
    won, actions, frames), and a totals row."""
    labels = list(results)
    head = "board " + " ".join(f"{label:>26s}" for label in labels)
    rows = [head, " " * 6 + " sec   won act frames   " * len(labels)]
    for number in boards:
        cells = " ".join(
            "{:7.1f} {:3d} {:3d} {:7d}  ".format(*results[label][number])
            for label in labels
        )
        rows.append(f"{number:5d} {cells}")
    totals = []
    for label in labels:
        secs, *counts = np.array(list(results[label].values())).sum(axis=0)
        totals.append("{:7.1f} {:3d} {:3d} {:7d}  ".format(secs, *map(int, counts)))
    rows.append("total " + " ".join(totals))
    return "\n".join(rows)


def grid(spec, boards, cuda_python):
    """Run :func:`run_configs` in one child process per backend and print the combined
    table between the load averages at the start and the end."""
    configs = parse_grid(spec)
    print("load at start:", os.getloadavg())
    results = {}
    for backend in dict.fromkeys(c[2] for c in configs):
        python = cuda_python if backend == "cuda" else sys.executable
        with tempfile.NamedTemporaryFile(suffix=".json") as out:
            subprocess.run(
                [python, "-m", "sentinel.tests.time_suite", "--grid", spec]
                + ["--json-out", out.name, *map(str, boards)],
                check=True,
                env={**os.environ, "SENTINEL_BACKEND": backend},
            )
            results.update(
                {
                    k: {int(b): v for b, v in r.items()}
                    for k, r in json.load(out).items()
                }
            )
    configs = [f"{p}{w or ''}@{b}" for p, w, b in configs]
    print(combined({k: results[k] for k in configs}, boards))
    print("load at end:", os.getloadavg())


def main(argv=None):
    """Print the timing table, and a profile split of ``--profile``'s board."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boards", type=int, nargs="*", default=list(trace_suite.SUITE))
    parser.add_argument("--player", choices=trace_suite.PLAYERS, default="beam")
    parser.add_argument("--width", type=int)
    parser.add_argument("--profile", type=int)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--grid", help="player[:width]@backend,... combined table")
    parser.add_argument("--cuda-python", default=sys.executable)
    parser.add_argument("--json-out", help="grid child: write the results here")
    args = parser.parse_args(argv)
    if args.grid and args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(run_configs(parse_grid(args.grid), args.boards), handle)
        return 0
    if args.grid:
        grid(args.grid, args.boards, args.cuda_python)
        return 0
    play = lambda n: trace_suite.play(n, player=args.player, width=args.width)
    play(0)
    warm_device()
    if args.capture:
        from numba import cuda  # pylint: disable=import-outside-toplevel

        cuda.profile_start()
    print("board  seconds  won  actions  frames")
    for number, rec in timed(play, args.boards).items():
        print(f"{number:5d} {rec[0]:8.1f} {rec[1]:4d} {rec[2]:8d} {rec[3]:7d}")
    if args.capture:
        cuda.profile_stop()
    if args.profile is not None:
        prof = cProfile.Profile()
        prof.runcall(play, args.profile)
        for part, secs in split(prof).items():
            print(f"{part:15s} {secs:8.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
