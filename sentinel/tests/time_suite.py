"""Time the suite boards on a warm process: ``python -m sentinel.tests.time_suite
[--player phase|beam] [--width N] [--profile BOARD] [boards...]``.

Plays board 0 first to warm the kernels, then prints each board's seconds, verdict, action
count and frames.  ``--profile`` also runs one board under cProfile and prints where its
time went: planning, advancing, landing sweeps and the CUDA driver waits.
"""

import argparse
import cProfile
import pstats
import time

from tqdm import tqdm

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


def main(argv=None):
    """Print the timing table, and a profile split of ``--profile``'s board."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boards", type=int, nargs="*", default=list(trace_suite.SUITE))
    parser.add_argument("--player", choices=trace_suite.PLAYERS, default="beam")
    parser.add_argument("--width", type=int)
    parser.add_argument("--profile", type=int)
    args = parser.parse_args(argv)
    play = lambda n: trace_suite.play(n, player=args.player, width=args.width)
    play(0)
    print("board  seconds  won  actions  frames")
    for number in tqdm(args.boards, unit="board"):
        start = time.perf_counter()
        rec = play(number)
        tqdm.write(
            f"{number:5d} {time.perf_counter() - start:8.1f} {int(rec['won']):4d} "
            f"{len(rec['trace']):8d} {rec['frames']:7d}"
        )
    if args.profile is not None:
        prof = cProfile.Profile()
        prof.runcall(play, args.profile)
        for part, secs in split(prof).items():
            print(f"{part:15s} {secs:8.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
