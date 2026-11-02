"""Profile the suite boards: ``python -m sentinel.tests.profile_suite [--top N] [--sort KEY]
[--callers NAME ...] [boards...]``; prints the heaviest functions of one trace_suite run.
"""

import argparse
import cProfile
import pstats

from sentinel.tests import trace_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boards", type=int, nargs="*", default=list(trace_suite.SUITE))
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sort", default="tottime", choices=("tottime", "cumulative"))
    parser.add_argument("--callers", nargs="*", default=[])
    parser.add_argument("--out", help="also write the raw profile here")
    args = parser.parse_args(argv)
    prof = cProfile.Profile()
    prof.runcall(trace_suite.run, args.boards)
    if args.out:
        prof.dump_stats(args.out)
    stats = pstats.Stats(prof).sort_stats(args.sort)
    stats.print_stats(args.top)
    for name in args.callers:
        stats.print_callers(name)


if __name__ == "__main__":
    main()
