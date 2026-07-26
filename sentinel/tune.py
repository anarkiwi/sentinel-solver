"""Fit :class:`sentinel.policy.Policy` numerically instead of by hand.

Staged objectives: a CHEAP dense one (does the gate accept every move of a recorded
human WIN -- on a win, each rejection is a measured false positive) then an EXPENSIVE
one (solve boards under a NODE budget, so a trial is a pure function of the board)."""

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import subprocess
import sys
import time

from sentinel import actions, memmap as mm
from sentinel.game import Game
from sentinel.policy import Policy

# board spec: "<landscape>" or "<landscape>:<enemy slots to KEEP>"
REGRESSION = ("0", "42", "110")  # a loss here makes the trial infeasible
TARGETS = ("335:0", "335:0,1", "335:0,3", "335")
LOSS_PER_LOSS = 1000.0  # a lost board outweighs any frame/expansion saving
FRAME_SCALE = 1e4
EXPANSION_SCALE = 1e4
HUMAN_WINS = ("ls0.json", "ls42.json", "ls110.json", "ls335.json")


def build(spec):
    """The board a spec names: a landscape number, optionally with only some enemy
    slots kept (``"335:0"`` is the Sentinel alone)."""
    num, _, keep = spec.partition(":")
    game = Game.typed(int(num))
    if keep:
        wanted = {int(k) for k in keep.split(",")}
        foes = (mm.T_SENTRY, mm.T_SENTINEL, mm.T_MEANIE)
        for slot, otype, _x, _y in game.objects():
            if otype in foes and slot not in wanted:
                actions.remove_object(game.state, slot)
    return game


def human_rejections():
    """``(rejected, total)`` human creates the gate refuses over every recorded WIN.

    A human win means each of those moves survived in the real game, so a rejection
    is a measured false positive -- the cheap dense signal, with no search."""
    from sentinel.astar_player import AStarPlayer
    from sentinel.tests import human_regress

    bad = total = 0
    for name in HUMAN_WINS:
        try:
            data = human_regress._load(name)
        except (FileNotFoundError, KeyError):
            continue
        for i, ev in enumerate(data["events"]):
            if ev["verb"] != "create" or ev["otype"] not in (mm.T_BOULDER, mm.T_ROBOT):
                continue
            st = human_regress.state_at(name, i)
            player = AStarPlayer(Game(st))
            player.st = st
            total += 1
            cost = mm.ENERGY_IN_OBJECTS[ev["otype"]]
            if not player._affords(cost, player._settle("create")):
                bad += 1
    return bad, total


def solve(spec, node_budget, max_actions):
    """Solve one board in THIS process; the dict a worker prints."""
    from sentinel.astar_player import AStarPlayer

    game = build(spec)
    player = AStarPlayer(game, node_budget=node_budget)
    t0 = time.time()
    won = player.run(max_actions=max_actions)
    return {
        "spec": spec,
        "won": bool(won),
        "actions": len(player.trace),
        "frames": player.frames,
        "energy": int(game.energy),
        "dead": bool(actions.player_dead(game.state)),
        "expansions": player.expansions,
        "wall": round(time.time() - t0, 1),
    }


def run_board(policy, spec, args, threads):
    """Solve ``spec`` under ``policy`` in a SPAWNED process, capped at ``args.cap``.

    Spawned, not forked: the numba LOS march leaves an OpenMP runtime in the parent.
    A cap is a timeout verdict recorded as such, never silently a loss."""
    env = dict(os.environ)
    env.update(policy.as_env())
    env["NUMBA_NUM_THREADS"] = str(max(1, threads))
    cmd = [
        sys.executable,
        "-m",
        "sentinel.tune",
        "--solve",
        spec,
        "--node-budget",
        str(args.node_budget),
        "--max-actions",
        str(args.max_actions),
    ]
    try:
        out = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=args.cap, check=False
        )
    except subprocess.TimeoutExpired:
        return {"spec": spec, "won": False, "capped": True, "wall": args.cap}
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"spec": spec, "won": False, "error": out.stderr[-400:].strip()}


def board_loss(rec):
    """Loss for one board: a loss dominates, then frames, then expansions."""
    loss = 0.0 if rec.get("won") else LOSS_PER_LOSS
    return (
        loss
        + rec.get("frames", 0) / FRAME_SCALE
        + (rec.get("expansions", 0) / EXPANSION_SCALE)
    )


def suggest(trial):
    """One candidate policy from an Optuna trial."""
    return Policy(
        weight=trial.suggest_float("weight", 1.0, 2.5),
        top_targets=trial.suggest_int("top_targets", 1, 6),
        top_hops=trial.suggest_int("top_hops", 4, 16),
        top_clears=trial.suggest_int("top_clears", 0, 8),
        top_relocs=trial.suggest_int("top_relocs", 0, 8),
        pursue_branch=trial.suggest_int("pursue_branch", 1, 5),
        max_pursue=trial.suggest_int("max_pursue", 10, 60),
        max_reclaim=trial.suggest_int("max_reclaim", 2, 16),
        strand_prune=trial.suggest_int("strand_prune", 0, 1),
        target_eye=trial.suggest_float("target_eye", 8.0, 14.0),
        eye_per_hop=trial.suggest_float("eye_per_hop", 0.5, 1.5),
        margin_k=trial.suggest_float("margin_k", 0.0, 2.0),
        source_gate=trial.suggest_int("source_gate", 0, 1),
        build_toll=trial.suggest_int("build_toll", 0, 1),
        liquidity_gate=trial.suggest_int("liquidity_gate", 0, 1),
    )


def evaluate(policy, args, report=None):
    """Staged loss for one policy: cheap human agreement, then the boards.

    ``report(step, loss)`` gets the running loss after each board so a pruner can kill
    a hopeless trial before the expensive ones; it may raise to prune."""
    bad, _total = human_rejections()
    loss = bad * LOSS_PER_LOSS
    if report is not None:
        report(0, loss)
    if bad and not args.allow_rejections:
        return loss, [], bad
    threads = max(1, args.cores // args.workers)
    recs = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_board, policy, spec, args, threads)
            for spec in list(REGRESSION) + list(TARGETS)
        ]
        for i, fut in enumerate(cf.as_completed(futures), start=1):
            rec = fut.result()
            recs.append(rec)
            if rec["spec"] in REGRESSION and not rec.get("won"):
                loss += LOSS_PER_LOSS * 10  # infeasible: a regression board was lost
            loss += board_loss(rec)
            if report is not None:
                report(i, loss)
    return loss, recs, bad


def optimize(args):
    """Run the study, persisting to SQLite so a run is resumable."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{args.storage}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    if not study.trials:
        study.enqueue_trial(dataclasses.asdict(Policy()))  # the shipped player first

    def objective(trial):
        policy = suggest(trial)
        t0 = time.time()

        def report(step, loss):
            trial.report(loss, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        loss, recs, bad = evaluate(policy, args, report)
        trial.set_user_attr("boards", recs)
        trial.set_user_attr("human_rejections", bad)
        wins = sum(1 for r in recs if r.get("won"))
        print(
            f"trial {trial.number:4d} loss={loss:9.2f} wins={wins}/{len(recs)} "
            f"rej={bad} {round(time.time() - t0)}s",
            flush=True,
        )
        return loss

    study.optimize(objective, n_trials=args.trials)
    best = study.best_trial
    art = {
        "study": args.study,
        "n_trials": len(study.trials),
        "best_loss": best.value,
        "best_params": best.params,
        "best_boards": best.user_attrs.get("boards"),
        "shipped": dataclasses.asdict(Policy()),
        "budgets": vars(args),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=1, sort_keys=True, default=str)
    print(json.dumps({"best_loss": best.value, "best": best.params}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solve", help="worker mode: solve one board spec, print JSON")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4, help="boards solved in parallel")
    ap.add_argument("--cores", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--node-budget", type=int, default=4000)
    ap.add_argument("--max-actions", type=int, default=120)
    ap.add_argument("--cap", type=float, default=120.0, help="per-board wall cap (s)")
    ap.add_argument("--allow-rejections", action="store_true")
    ap.add_argument("--study", default="policy")
    ap.add_argument("--storage", default="out/tune.db")
    ap.add_argument("--out", default="out/tune.json")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if args.solve:
        print(json.dumps(solve(args.solve, args.node_budget, args.max_actions)))
        return 0
    return optimize(args)


if __name__ == "__main__":
    raise SystemExit(main())
