"""Play the README suite boards offline and record each board's result:
``python -m sentinel.tests.trace_suite [--player phase|beam] [--out path] [--compare path]
[--readme] [boards...]``.

A record is the board's landscape, enemy count, action count, frames, final energy, verdict
and a SHA-256 digest of the full action trace. ``--compare`` exits non-zero unless every
board's record equals the recorded file (default: the player's committed file,
``docs/results/offline.json`` for the phase player and ``docs/results/beam.json`` for the
beam player; any file with the same fields, or with full ``trace`` lists, works), so one
baseline gates a change to any kernel or backend. ``--out`` rewrites a result file;
``--readme`` renders the README results block from the committed data.
"""

import argparse
import hashlib
import json
import sys
import time

from tqdm import tqdm

from sentinel.beam_player import BeamPlayer
from sentinel.game import Game
from sentinel.phase_player import PhasePlayer
from sentinel.tests import readme_results

SUITE = (0, 42, 60, 110, 298, 321, 335, 373, 1442)
OFFLINE = readme_results.OFFLINE
BEAM = readme_results.BEAM
PLAYERS = ("phase", "beam")
RESULTS = {"phase": OFFLINE, "beam": BEAM}
ACTIONS = {"phase": 200, "beam": 60}


def digest(trace):
    """Hex SHA-256 of the canonical JSON of an action trace."""
    blob = json.dumps(trace, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def summary(record):
    """The committed form of a record: the full ``trace`` replaced by its count and digest."""
    out = {k: v for k, v in record.items() if k != "trace"}
    if "trace" in record:
        out["actions"] = len(record["trace"])
        out["digest"] = digest(record["trace"])
    return out


def make(player, game):
    """The ``player`` ("phase" or "beam") for ``game``."""
    if player == "beam":
        return BeamPlayer(game)
    return PhasePlayer(game, verbose=False)


def play(landscape, max_actions=None, player="phase"):
    """One board's record: ``landscape``, ``enemies``, ``trace``, ``frames``, ``energy``, ``won``."""
    game = Game.typed(landscape)
    enemies = len(game.enemy_slots())
    agent = make(player, game)
    won = agent.run(max_actions=max_actions or ACTIONS[player])
    return {
        "landscape": landscape,
        "enemies": enemies,
        "trace": json.loads(json.dumps(agent.trace, default=repr)),
        "frames": agent.frames,
        "energy": game.energy,
        "won": bool(won),
    }


def run(boards, player="phase"):
    """``{landscape: record}`` for ``boards`` played by ``player``, with progress."""
    out = {}
    with tqdm(boards, unit="board") as progress:
        for number in progress:
            progress.set_postfix(ls=number)
            out[str(number)] = play(number, player=player)
    return out


def diff(base, new):
    """The landscapes whose records differ on any field both sides carry."""

    def same(a, b):
        a, b = summary(a), summary(b)
        return all(a[k] == b[k] for k in a.keys() & b.keys())

    return [k for k in sorted(base.keys() & new.keys()) if not same(base[k], new[k])]


def main(argv=None):
    """Record and/or compare the suite; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boards", type=int, nargs="*", default=list(SUITE))
    parser.add_argument("--player", choices=PLAYERS, default="phase")
    parser.add_argument("--out", help="write the records here as JSON")
    parser.add_argument(
        "--compare",
        nargs="?",
        const="",
        help="exit 1 unless these recorded results match (default: the player's committed file)",
    )
    parser.add_argument(
        "--readme", action="store_true", help="render the README results block"
    )
    args = parser.parse_args(argv)
    if args.readme and not (args.out or args.compare is not None):
        readme_results.rewrite()
        return 0
    start = time.perf_counter()
    got = run(args.boards, args.player)
    print(f"{len(got)} boards in {time.perf_counter() - start:.1f} s")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {k: summary(v) for k, v in got.items()},
                handle,
                sort_keys=True,
                indent=1,
            )
            handle.write("\n")
        if args.readme:
            readme_results.rewrite()
    if args.compare is not None:
        with open(args.compare or RESULTS[args.player], encoding="utf-8") as handle:
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
