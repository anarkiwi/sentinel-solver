"""Retrograde A* regression over a recorded human win.

Hands the A* player the human's PRE-action state at event ``i`` (rebuilt as the
human-win tests do, enemy clock applied) and scans ``i`` DOWN from the last event:
the first ``i`` it cannot win from within budget is the board it cannot recover.
"""

import argparse
import json
import multiprocessing as mp
import os
import time

from sentinel import enemies, isoview, los, memmap as mm
from sentinel.astar_player import AStarPlayer
from sentinel.game import Game
from sentinel.stance_player import StancePlayer
from sentinel.test_human_win_logs import state_from_event, _load
from sentinel.tests.human_audit import _apply_truth

PLANNERS = {"astar": AStarPlayer, "stance": StancePlayer}
PLANNER = "astar"
NODE_BUDGET = 20000
TIME_BUDGET = 15.0  # per _search wall-clock cut
CAP = 600.0  # hard wall clock for one whole attempt
ESCALATE = 3.0  # cap multiplier for the re-run that confirms a loss
MAX_ACTIONS = 250


def state_at(name, i):
    """The human's PRE-action state at event ``i``: reconstructed objects/energy/aim
    plus the recorded enemy clock and cooldown accumulators."""
    data = _load(name)
    ev = data["events"][i]
    st = state_from_event(ev, data["landscape"])
    if ev.get("enemy_clock"):
        _apply_truth(st, {e["slot"]: e for e in ev["enemy_clock"]})
    for addr, key in (
        (mm.COOLDOWN_BRESENHAM, "cooldown_bresenham"),
        (mm.COOLDOWN_GATE, "cooldown_gate"),
    ):
        if ev.get(key) is not None:
            st.mem[addr] = ev[key] & 0xFF
    st.mem[mm.PLAYER_NOT_ACTED] = 0  # mid-game: the enemies are already running
    return st


def _human_action(name, i):
    """The human's own move at event ``i``, for the record."""
    ev = _load(name)["events"][i]
    return {
        "verb": ev["verb"],
        "otype": ev["otype"],
        "otype_name": mm.TYPES.get(ev["otype"]),
        "target": list(ev["target"]),
        "player_tile": [ev["player"]["x"], ev["player"]["y"]],
        "energy": ev["energy"],
    }


def attempt(name, i, node_budget=NODE_BUDGET, time_budget=TIME_BUDGET, planner=PLANNER):
    """Run a planner from the human's pre-action state at event ``i``, returning the
    outcome, its own action trace and the human's move.

    The whole-attempt cap is enforced by the PARENT killing this process: a signal
    raised inside the numba LOS march corrupts the dispatcher."""
    st = state_at(name, i)
    player = PLANNERS[planner](
        Game(st), node_budget=node_budget, time_budget=time_budget
    )
    t0 = time.time()
    won = player.run(max_actions=MAX_ACTIONS)
    return {
        "i": i,
        "won": won,
        "outcome": "won" if won else "lost",
        "seconds": round(time.time() - t0, 1),
        "expansions": player.expansions,
        "human_action": _human_action(name, i),
        "start": {
            "energy": int(st.energy),
            "player_tile": list(st.player_xy()),
            "enemies": len(enemies.enemy_slots(st)),
        },
        "player_trace": [
            {"frame": f, "verb": v, "tile": list(t), "energy": e}
            for f, v, t, e, _ in player.trace
        ],
        "player_energy_end": int(player.st.energy),
    }


def _worker(name, i, node_budget, time_budget, threads, planner, q):
    """One attempt on ``threads`` numba threads: ``los_jit.march_batch`` is
    ``parallel=True``, so an uncapped worker takes a thread per core (load 117/24)."""
    try:
        import numba  # pylint: disable=import-outside-toplevel

        numba.set_num_threads(max(1, threads))
    except ImportError:
        pass
    q.put(attempt(name, i, node_budget, time_budget, planner))


def _capped(name, i, seconds):
    return {
        "i": i,
        "won": None,
        "outcome": "capped",
        "seconds": round(seconds, 1),
        "expansions": None,
        "human_action": _human_action(name, i),
        "start": None,
        "player_trace": [],
        "player_energy_end": None,
    }


def _run(name, batch, budgets, cap, log):
    """Run one index per process, killing any that outlives ``cap``.

    Spawned, not forked: the numba LOS march leaves an OpenMP runtime in the parent
    that a fork aborts on."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    live = {}
    threads = max(1, (os.cpu_count() or 2) // max(1, len(batch)))
    for i in batch:
        p = ctx.Process(
            target=_worker,
            args=(
                name,
                i,
                budgets["node_budget"],
                budgets["time_budget"],
                threads,
                budgets.get("planner", PLANNER),
                q,
            ),
        )
        p.start()
        live[i] = (p, time.time())
    out = {}
    dead = {}
    while live:
        while not q.empty():
            rec = q.get()
            out[rec["i"]] = rec
            live.pop(rec["i"], (None, None))[0].join(5)
        for i, (p, t0) in list(live.items()):
            if not p.is_alive():
                # the result can still be in the queue's feeder thread: grace it
                dead.setdefault(i, time.time())
                if time.time() - dead[i] > 2.0:
                    out[i] = _capped(name, i, time.time() - t0)
                    live.pop(i)
            elif time.time() - t0 > cap:
                p.terminate()
                p.join(5)
                out[i] = _capped(name, i, time.time() - t0)
                live.pop(i)
        time.sleep(0.2)
    for rec in sorted(out.values(), key=lambda r: -r["i"]):
        ha = rec["human_action"]
        log(
            f"i={rec['i']:3d} human={ha['verb']:8s} {tuple(ha['target'])} "
            f"E={ha['energy']:2d} -> {rec['outcome']} in "
            f"{len(rec['player_trace'])} actions ({rec['seconds']}s)"
        )
    return out


def _budgets(over=None):
    return dict(
        {
            "node_budget": NODE_BUDGET,
            "time_budget": TIME_BUDGET,
            "cap": CAP,
            "escalate": ESCALATE,
            "planner": PLANNER,
        },
        **(over or {}),
    )


def _settle(name, results, candidates, budgets, log):
    """The highest index in ``candidates`` that genuinely failed: a capped one is
    re-run alone at ``escalate`` x the cap first, so a loss is never just a timeout."""
    while True:
        bad = [i for i in candidates if results[i]["won"] is not True]
        top = max(bad, default=None)
        if top is None or results[top]["outcome"] != "capped":
            return top
        cap = budgets["cap"] * budgets["escalate"]
        log(f"i={top}: capped -- re-running alone at cap={cap:.0f}s")
        results.update(_run(name, [top], budgets, cap, log))
        if results[top]["outcome"] == "capped":
            return top


def bisect(name, budgets=None, workers=None, log=None):
    """Bisect the handover axis for the boundary the planner cannot cross.

    Each round probes ``workers`` evenly spaced handovers inside the live interval at
    once, so the interval divides by ``workers+1`` per round, not by two."""
    budgets = _budgets(budgets)
    n = _load(name)["n_events"]
    workers = workers or max(1, (os.cpu_count() or 2) // 3)
    log = log or (lambda _m: None)

    results = {}
    lo, hi = -1, n  # lo: highest handover known to fail; hi: lowest known to win
    while hi - lo > 1:
        span = [i for i in range(lo + 1, hi) if i not in results]
        if not span:
            break
        k = min(workers, len(span))
        probes = sorted(
            {span[round(j * (len(span) - 1) / max(1, k - 1))] for j in range(k)}
        )
        log(f"interval ({lo}, {hi}) -- probing {probes}")
        results.update(_run(name, probes, budgets, budgets["cap"], log))
        top = _settle(name, results, probes, budgets, log)
        if top is not None:
            lo = max(lo, top)
        won = [i for i in probes if results[i]["won"] is True and i > lo]
        if won:
            hi = min(hi, *won)
    log(f"boundary: fails at {lo}, wins from {hi if hi < n else None}")
    ordered = [results[i] for i in sorted(results, reverse=True)]
    return {
        "fixture": name,
        "n_events": n,
        "method": "bisect",
        "budgets": dict(budgets, max_actions=MAX_ACTIONS),
        "first_loss": lo if lo >= 0 else None,
        "last_win": hi if hi < n else None,
        "attempts": ordered,
    }


def regress(
    name, budgets=None, workers=None, stop_at_loss=True, indices=None, log=None
):
    """Scan handover points from the last event backwards until the A* player first
    fails -- the exhaustive form of :func:`bisect`, one batch of moves at a time."""
    budgets = _budgets(budgets)
    n = _load(name)["n_events"]
    todo = list(indices) if indices is not None else list(reversed(range(n)))
    workers = workers or max(1, (os.cpu_count() or 2) // 3)
    log = log or (lambda _m: None)

    results = {}
    first_loss = None
    for k in range(0, len(todo), workers):
        batch = [
            b for b in todo[k : k + workers] if first_loss is None or b > first_loss
        ]
        if not batch:
            break
        results.update(_run(name, batch, budgets, budgets["cap"], log))
        first_loss = _settle(name, results, list(results), budgets, log)
        if stop_at_loss and first_loss is not None:
            break

    ordered = [results[i] for i in sorted(results, reverse=True)]
    return {
        "fixture": name,
        "n_events": n,
        "method": "linear",
        "budgets": dict(budgets, max_actions=MAX_ACTIONS),
        "first_loss": first_loss,
        "last_win": min((r["i"] for r in ordered if r["won"]), default=None),
        "attempts": ordered,
    }


def _acts(name, i, horizon, trace=None, start_tile=None):
    """The human's next ``horizon`` moves from event ``i`` as solid arrows, plus the
    A* player's own trace (dashed) if one is given."""
    out = []
    for ev in _load(name)["events"][i : i + horizon]:
        out.append(
            {
                "kind": "human",
                "verb": ev["verb"],
                "otype_name": mm.TYPES.get(ev["otype"]),
                "from": (ev["player"]["x"], ev["player"]["y"]),
                "to": tuple(ev["target"]),
                "energy": ev["energy"],
            }
        )
    here = tuple(start_tile) if start_tile else None
    for step in trace or ():
        tile = tuple(step["tile"])
        out.append(
            {
                "kind": "astar",
                "verb": step["verb"],
                "otype_name": None,
                "from": here or tile,
                "to": tile,
                "energy": step["energy"],
            }
        )
        if step["verb"] in ("transfer", "hyperspace"):
            here = tile
    return out


COST = {
    # ls335 event 121, ONE decision tick, 30 s at 8 numba threads; lattice size exact.
    "captions": (
        (
            "WHY ONE MOVE COSTS SO MUCH",
            "the planner cannot 'see' -- it asks the ROM: can a KEYBOARD AIM land on this tile?",
            "one aim = one bit-exact 6502 LOS march from the eye ($1C10 -> march $1D00)",
        ),
        (
            "1. SWEEP THE BEARING LATTICE",
            "the body pans in 8-unit notches: 32 bearings, every one a separate march",
            "32 rays",
        ),
        (
            "2. x THE PITCH BAND",
            "the body also pitches: 27 v_angles in the keyboard band",
            "32 x 27 = 864 rays",
        ),
        (
            "3. x THE SIGHTS WINDOW",
            "and the cursor roams a 64 x 64 px window inside each of those",
            "864 x 4096 = 3,538,944 rays -- for ONE tile's landability question",
        ),
        (
            "4. x EVERY CANDIDATE, x EVERY SEARCH NODE",
            "30 s of one tick: 11 hop-picks -> 90 hop prices -> 144 targeted aims -> 296 sweeps",
            "and that bought ONE expanded search node. 900 s later: still no move.",
        ),
    ),
}


def diagram(name, i, rec=None, horizon=8, path=None, with_astar=True, anim=False):
    """Write the annotated isometric diagram of the handover board at event ``i``."""
    data = _load(name)
    st = state_at(name, i)
    trace = rec["player_trace"] if (rec and with_astar) else None
    acts = [] if anim else _acts(name, i, horizon, trace, st.player_xy())
    outcome = rec["outcome"] if rec else "not run"
    tail = f" after {len(trace)} actions" if trace else ""
    notes = [
        "solid arrows = what the HUMAN did next, in order",
        "dashed arrows = what the A* player did instead",
        "red wedges = enemy scan cones on their true recorded facings",
        "gold diamond = the platform (stand on it, then hyperspace)",
    ]
    overlay = ()
    if anim:
        overlay = isoview.cost_layer(st, COST, _hop_candidates(st))
        notes = [
            "animated: one decision tick's aim-landability work, to scale in structure",
            "gold ray = the aim being marched; blue fan = the bearing lattice",
            "orange rings = candidate hop tiles, each needing its own sweep",
        ]
    svg = isoview.diagram(
        st,
        f"ls{data['entered_code']} @ human event {i}",
        [
            f"typed landscape {data['entered_code']} (generate seed "
            f"{data['landscape']}), event {i}/{data['n_events'] - 1}",
            f"A* handover here: {outcome.upper()}{tail}",
        ],
        acts,
        notes,
        overlay,
    )
    suffix = "_cost" if anim else ""
    out = path or os.path.join("renders", f"{name[:-5]}_ev{i}_iso{suffix}.svg")
    return isoview.write(svg, out)


def _hop_candidates(st, want=9):
    """A spread sample of the tiles a real keyboard aim can build on from here -- the
    candidate set every hop-pick prices."""
    tiles = sorted(los.landable_views(st))
    if len(tiles) <= want:
        return tiles
    step = len(tiles) / want
    return [tiles[int(k * step)] for k in range(want)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixture", nargs="?", default="ls335.json")
    parser.add_argument("--out", default=None, help="write the artifact JSON here")
    parser.add_argument(
        "--diagram",
        action="store_true",
        help="also write the annotated isometric SVG of the first losing board",
    )
    parser.add_argument("--horizon", type=int, default=8, help="human moves to draw")
    parser.add_argument("--node-budget", type=int, default=NODE_BUDGET)
    parser.add_argument("--time-budget", type=float, default=TIME_BUDGET)
    parser.add_argument("--cap", type=float, default=CAP)
    parser.add_argument(
        "--planner",
        choices=sorted(PLANNERS),
        default=PLANNER,
        help="which planner to score; 'stance' adds the stance-graph route generator",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--linear",
        action="store_true",
        help="exhaustive backward scan instead of the default bisection",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="explicit comma-separated handover indices (implies --linear)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="with --linear, keep scanning past the first loss",
    )
    args = parser.parse_args(argv)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    budgets = {
        "node_budget": args.node_budget,
        "time_budget": args.time_budget,
        "cap": args.cap,
        "planner": args.planner,
    }
    indices = [int(s) for s in args.indices.split(",")] if args.indices else None
    if args.linear or indices:
        out = regress(
            args.fixture,
            budgets=budgets,
            workers=args.workers,
            stop_at_loss=not args.all,
            indices=indices,
            log=log,
        )
    else:
        out = bisect(args.fixture, budgets=budgets, workers=args.workers, log=log)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        log(f"wrote {args.out}")
    if args.diagram and out["first_loss"] is not None:
        i = out["first_loss"]
        rec = next(a for a in out["attempts"] if a["i"] == i)
        log(f"wrote {diagram(args.fixture, i, rec, args.horizon)}")
    log(f"first_loss={out['first_loss']} last_win={out['last_win']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
