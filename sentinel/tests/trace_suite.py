"""Play the README suite boards offline and record each full action trace:
``python -m sentinel.tests.trace_suite [--out path] [--compare path] [boards...]``.

``--compare`` exits non-zero unless every board's trace, frames, energy and verdict equal
the recorded file, so one baseline gates a change to any kernel or backend.
"""

import argparse
import json
import sys
import time

from tqdm import tqdm

from sentinel import actions
from sentinel.game import Game
from sentinel.phase_player import PhasePlayer

SUITE = (0, 42, 60, 110, 298, 321, 335, 373, 1442)


def play(landscape, max_actions=200):
    """One board's record: ``trace``, ``frames``, ``energy``, ``won``, ``dead``."""
    game = Game.typed(landscape)
    player = PhasePlayer(game, verbose=False)
    won = player.run(max_actions=max_actions)
    return {
        "trace": json.loads(json.dumps(player.trace, default=repr)),
        "frames": player.frames,
        "energy": game.energy,
        "won": bool(won),
        "dead": bool(actions.player_dead(game.state)),
    }


def run(boards):
    """``{landscape: record}`` for ``boards``, with progress."""
    out = {}
    with tqdm(boards, unit="board") as progress:
        for number in progress:
            progress.set_postfix(ls=number)
            out[str(number)] = play(number)
    return out


def diff(base, new):
    """The landscapes whose records differ."""
    return [k for k in sorted(set(base) | set(new)) if base.get(k) != new.get(k)]


def main(argv=None):
    """Record and/or compare the suite; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boards", type=int, nargs="*", default=list(SUITE))
    parser.add_argument("--out", help="write the records here as JSON")
    parser.add_argument("--compare", help="exit 1 unless these recorded traces match")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    got = run(args.boards)
    print(f"{len(got)} boards in {time.perf_counter() - start:.1f} s")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(got, handle, sort_keys=True, indent=1)
    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            base = json.load(handle)
        bad = diff({k: base[k] for k in got if k in base}, got)
        missing = [k for k in got if k not in base]
        print(
            "identical" if not (bad or missing) else f"DIFFER {bad} missing {missing}"
        )
        return 1 if bad or missing else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
