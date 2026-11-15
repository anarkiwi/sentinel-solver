#!/usr/bin/env python3
"""What every candidate move at each decision of the search player's game leads to.

The emulated game is deterministic from ``emu.enter(N)``: it is played once, its machine
cloned before each move; there every candidate the search generates is forced on a fork
and a fresh search continues from it to the end.
"""

import argparse
import json

from tqdm import tqdm

from driver import kbd_prog, prog, search_player as sp
from driver.emu import EmuMon
from sentinel import emu


def _quiet(_msg):
    pass


def _player(bm, knobs):
    player = sp.SearchPlayer(bm, log=_quiet, **knobs)
    player.progress = False
    return player


def ending(facts, actions):
    """The audited outcome of a line ending in move ``facts`` after ``actions`` moves."""
    return {
        "won": bool(facts["won"]),
        "alive": bool(facts["alive"] or facts["won"]),
        "energy": facts["energy"],
        "eye": round(facts["z"] + facts["z_frac"] / 256, 3),
        "actions": actions,
    }


def decisions(landscape, knobs, count):
    """``[(machine, move)]``: the first ``count`` decisions of the game ``knobs`` plays,
    each machine a clone taken before the move it chose."""
    player = _player(EmuMon(emu.enter(landscape)), knobs)
    out = []
    execute = player._execute

    def record(planned):
        out.append((player.bm.clone(), planned.move))
        return execute(planned)

    player._execute = record
    player.run(count)
    return out


def options(bm, knobs, cont, max_actions, progress):
    """``[(move, ending)]`` of every candidate the ``knobs`` search generates on ``bm``,
    forced, then continued by a ``cont`` search for at most ``max_actions`` moves."""
    (work,) = sp.batches([sp.Node(bm)], knobs["branch"])
    outs, twins = prog.forks(bm, [p for _m, p in work])
    rows = []
    for (move, _program), out, twin in zip(work, outs, twins):
        facts = kbd_prog.move_facts(out)
        end = None
        if facts["status"] == prog.HALT_OK:
            end, live = ending(facts, 1), facts["alive"] and not facts["won"]
            if live and max_actions > 1:
                player = _player(twin, cont)
                player.run(max_actions - 1)
                if player.audit:
                    end = ending(player.audit[-1][4], 1 + len(player.audit))
        rows.append((move, end))
        progress.update(1)
    return rows


def audit(landscape, knobs, cont, count, max_actions=120):
    """``[(decision, entry facts, chosen move, [(move, ending)])]`` of the first
    ``count`` decisions of ``landscape`` played by ``knobs``, continued by ``cont``."""
    real = decisions(landscape, knobs, count)
    out = []
    with tqdm(desc="options", unit="option") as progress:
        for d, (bm, chosen) in enumerate(real):
            facts = kbd_prog.move_facts(prog.execute(bm.clone(), sp.ENTRY))
            rows = options(bm, knobs, cont, max_actions - d, progress)
            out.append((d, ending(facts, d), chosen, rows))
    return out


def report(rows):
    """Text lines of :func:`audit` rows; the chosen move is starred."""
    for d, now, chosen, options_ in rows:
        yield f"decision {d:3d} E={now['energy']:2d} eye={now['eye']:6.3f} took {chosen}"
        for move, end in options_:
            mark = "*" if move == chosen else " "
            yield f"  {mark} {str(move):28s} {end if end else 'refused'}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", type=int)
    parser.add_argument("--game", choices=sorted(sp.PRESETS), default="wide")
    parser.add_argument("--preset", choices=sorted(sp.PRESETS), default="narrow")
    parser.add_argument("--decisions", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=120)
    parser.add_argument("--out", help="write the audit here as JSON")
    args = parser.parse_args(argv)
    rows = audit(
        args.landscape,
        sp.PRESETS[args.game],
        sp.PRESETS[args.preset],
        args.decisions,
        args.max_actions,
    )
    for line in report(rows):
        print(line)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1, default=int)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
