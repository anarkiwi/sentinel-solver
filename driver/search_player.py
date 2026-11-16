#!/usr/bin/env python3
"""One search core over the emulated game: a beam whose children are driver programs.

Moves are :func:`sentinel.moves.views`, children :func:`kbd_prog.compile_move`
programs expanded by one :func:`prog.forks` per parent, kept by ``keep_best`` per machine
state; a found line is played open loop on the session (receding horizon) and checked.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time

import numpy as np
from tqdm import tqdm

from driver import core, kbd_prog, prog
from driver.kbd_aim import A_SFLAG
from sentinel import memmap as mm, moves as bp
from sentinel.moves import WAIT_QUANTUM
from sentinel.state import State

PRESETS = {
    "wide": {"width": 16, "depth": 30, "branch": 3, "commit": 10},
    "narrow": {"width": 4, "depth": 12, "branch": 2, "commit": 1},
}
CPUS = len(os.sched_getaffinity(0))
FIELDS = ("alive", "won", "enemies", "area", "home", "seen", "eye", "energy")
ENTRY = kbd_prog.compile_move("wait")  # no wait: the facts as the machine stands


def machine_key(bm):
    """Digest of all a program's outcome on ``bm`` depends on: RAM, CPU registers and
    cycle, and the monitor's owed test, parked state and held tap."""
    m, mpu = bm.m, bm.m.mpu
    return (
        hashlib.blake2b(bytes(m.ram), digest_size=16).digest(),
        m.cycle,
        (mpu.a, mpu.x, mpu.y, mpu.pc, mpu.sp, mpu.p),
        bm._owed,
        bm._parked,
        bm._tap,
    )


def identity(keys):
    """``(n, 1)`` int64 rows, equal exactly where the :func:`machine_key` are equal."""
    ids = {}
    return np.array([[ids.setdefault(k, len(ids))] for k in keys], dtype=np.int64)


class Node:
    """A machine state on a line: its monitor and key, the move and facts that reached
    it from ``parent``, its frames and ``area``: the enemies left after each move of the
    line, summed, so an enemy taken sooner counts less at every move after."""

    __slots__ = ("bm", "key", "parent", "move", "program", "facts", "frames", "area")

    def __init__(self, bm, key=None, parent=None, move=None, facts=None):
        self.bm, self.key = bm, machine_key(bm) if key is None else key
        self.parent, self.facts = parent, facts
        self.move, self.program = move if move else (None, None)
        self.frames = (parent.frames if parent else 0) + (facts or {}).get("frames", 0)
        self.area = (parent.area if parent else 0) + (facts or {}).get("enemies", 0)

    def line(self):
        """The nodes from the root's child down to this one."""
        out, at = [], self
        while at.parent is not None:
            out.append(at)
            at = at.parent
        return out[::-1]

    def fields(self):
        """The objective's fields; a won game counts as alive."""
        f = self.facts
        return {
            "alive": int(f["alive"] or f["won"]),
            "won": f["won"],
            "enemies": f["enemies"],
            "area": self.area,
            "home": int(f["away"] == 0),
            "seen": int(f["seen"] != 0),
            "eye": f["z"] * 256 + f["z_frac"],
            "energy": f["energy"],
        }


def outcome_facts(outcome, move=None):
    """:func:`kbd_prog.move_facts` of a forked ``move``; one without a status stalled in
    a wait, a driver fault, raised with the records it ended on."""
    facts = kbd_prog.move_facts(outcome)
    if "status" not in facts:
        tail = kbd_prog.records(outcome)[-6:]
        raise RuntimeError(f"move {move} stalled: {outcome[:3]}, last records {tail}")
    return facts


def _state(bm):
    return State.from_mem(bytes(core.live_image(bm)))


def _reuse(st, view):
    """Whether the sights are on and stand on ``view``'s bearing."""
    me = st.player
    on = bool(st.mem[A_SFLAG] & 0x80)
    return on and (st.obj_h_angle[me], st.obj_v_angle[me]) == tuple(view[:2])


def move_program(st, kind, k, steps):
    """``(name, program)`` of one :func:`bp.views` move on ``st`` whose actions aim on
    ``steps``; the relocation from the platform with the Sentinel gone is the finish."""
    if kind == bp.WAIT:
        return "wait", kbd_prog.compile_move("wait", None, 0, WAIT_QUANTUM)
    if kind == bp.RELOCATE:
        gone = bool(st.obj_flags[0] & 0x80)
        home = tuple(st.player_xy()) == tuple(st.platform_xy)
        name = "finish" if gone and home else "relocate"
        return name, kbd_prog.compile_move(name)
    name = bp.KINDS[kind]
    view = steps if kind == bp.HOP else steps[0]
    return name, kbd_prog.compile_move(name, view, int(k), 0, _reuse(st, steps[0]))


def batches(parents, branch):
    """Per parent ``Node``, its ``[(move, program)]``, ``move`` ``(name, tile, k)``."""
    states = [_state(n.bm) for n in parents]
    imgs = np.stack([np.frombuffer(st.mem, dtype=np.uint8) for st in states])
    cols = bp.views(states, imgs, bp.Board.of_states(states), branch)
    out = [[] for _ in parents]
    for i, steps in enumerate(bp.action_views(states, cols)):
        if steps is None:
            continue
        parent, kind, tile, k = (
            int(cols[c][i]) for c in ("parent", "kind", "tile", "k")
        )
        name, program = move_program(states[parent], kind, k, steps)
        out[parent].append(((name, (tile // mm.N, tile % mm.N), k), program))
    return out


class Expansion:
    """One parent's candidate ``programs`` (``(move, program)``), each move's
    :func:`bp.best_case` ``bounds``, and ``done[j]``: ``(facts, twin, key)`` once forked.
    """

    __slots__ = ("programs", "bounds", "done")

    def __init__(self, st, programs):
        self.programs = programs
        self.bounds = [bp.best_case(st, *_named(m)) for m, _p in programs]
        self.done = [None] * len(programs)

    def pending(self):
        """Indices of the moves not yet forked."""
        return [j for j, d in enumerate(self.done) if d is None]


def _named(move):
    name, tile, k = move
    return {"hyperspace": "relocate"}.get(name, name), tile, k


def key_of(node):
    """:func:`bp.lexicographic`'s keys of ``node`` as a tuple, larger better."""
    f = node.fields()
    return (
        f["won"],
        -f["enemies"],
        -f["area"],
        f["home"],
        -f["seen"],
        f["eye"],
        f["energy"],
    )


def line_bound(parent, bound):
    """:func:`bp.best_case`'s ``bound`` of a move from ``parent`` as a :func:`key_of`
    bound: its enemies left added to the parent's ``area``."""
    if bound is None:
        return None
    return bound[:2] + (bound[1] - parent.area,) + bound[2:]


def kid_bound(node):
    """The bound its parent's expansion priced ``node``'s move at."""
    st = _state(node.parent.bm)
    return line_bound(node.parent, bp.best_case(st, *_named(node.move)))


def may_displace(job, cut):
    """Whether pending ``job`` (``(parent, expansion, index)``) could rank above ``cut``,
    the ``(key, frames)`` of the last survivor kept: its bound beats the key, or ties it
    with its frames, at least the parent's plus the move's least, no more than the cut's.
    The kept set only improves as children arrive, so a job this rejects stays rejected.
    """
    parent, exp, j = job
    bound = line_bound(parent, exp.bounds[j])
    if bound is None or bound > cut[0]:
        return True
    least = WAIT_QUANTUM if exp.programs[j][0][0] == "wait" else 1
    return bound == cut[0] and parent.frames + least <= cut[1]


def _order(bound):
    """Sort key putting unbounded moves first, then the best bound."""
    return (0,) if bound is None else (1,) + tuple(-v for v in bound)


def run_batches(parents, work):
    """``[(outcomes, monitors)]``: one :func:`prog.forks` per parent, the parents
    concurrently, the cores shared out by program count."""
    total = max(1, sum(map(len, work)))

    def one(job):
        node, programs = job
        if not programs:
            return [], []
        threads = max(1, round(CPUS * len(programs) / total))
        return prog.forks(node.bm, [p for _m, p in programs], threads=threads)

    with concurrent.futures.ThreadPoolExecutor(max(1, len(parents))) as pool:
        return list(pool.map(one, zip(parents, work)))


class SearchPlayer:
    """The search core on an ``EmuMon`` session; ``run`` plays by receding-horizon beam search.

    ``trace`` rows are ``[frames, move, tile, energy, eye]``, ``audit`` rows ``[move no, move,
    tile, planned facts, executed facts]``. With ``live`` (a BinMon on the same game) each
    move also plays there into ``live_audit``; the first disagreement (``diverged``) ends it.
    """

    def __init__(
        self,
        bm,
        width=16,
        depth=30,
        branch=3,
        commit=1,
        log=print,
        live=None,
        check=False,
    ):
        self.bm, self.log, self.live, self.check = bm, log, live, check
        self.progress = True
        self.width, self.depth, self.branch, self.commit = width, depth, branch, commit
        self.trace, self.audit, self.live_audit = [], [], []
        self.diverged = None
        self.won, self.frames = False, 0
        self.stats = {
            "children": 0,
            "bounded": 0,
            "frames": 0,
            "seconds": 0.0,
            "cpu_seconds": 0.0,
            "violations": [],
        }
        self._cache = {}

    def _expansions(self, parents, cache):
        """Per distinct parent, its :class:`Expansion`, built on first sight and kept in
        ``cache`` (seeded from the previous search's) on the parent's machine key."""
        nodes = list({n.key: n for n in parents}.values())
        fresh = [n for n in nodes if n.key not in cache and n.key not in self._cache]
        if fresh:
            for node, programs in zip(fresh, batches(fresh, self.branch)):
                cache[node.key] = Expansion(_state(node.bm), programs)
        return [(n, cache.setdefault(n.key, self._cache.get(n.key))) for n in nodes]

    def _evaluate(self, jobs):
        """Fork ``jobs`` (``(node, expansion, index)``), one batch per parent."""
        by_node = {}
        for node, exp, j in jobs:
            by_node.setdefault(node.key, (node, exp, []))[2].append(j)
        groups = list(by_node.values())
        t0, c0 = time.time(), time.process_time()
        work = [[exp.programs[j] for j in js] for _n, exp, js in groups]
        for (_node, exp, js), (outs, twins) in zip(
            groups, run_batches([g[0] for g in groups], work)
        ):
            for j, o, twin in zip(js, outs, twins):
                facts = outcome_facts(o, exp.programs[j][0])
                exp.done[j] = (facts, twin, machine_key(twin))
            self.stats["children"] += len(outs)
            self.stats["frames"] += sum(o.frames for o in outs)
        self.stats["seconds"] += time.time() - t0
        self.stats["cpu_seconds"] += time.process_time() - c0

    def _kids(self, expansions):
        """The forked, completed children of ``expansions``, parent then move order."""
        return [
            Node(twin, key, node, exp.programs[j], facts)
            for node, exp in expansions
            for j, done in enumerate(exp.done)
            if done is not None
            for facts, twin, key in (done,)
            if facts["status"] == prog.HALT_OK
        ]

    def _expand(self, parents, cache):
        """Every child of ``parents``, each forked once (``check``: against its bound)."""
        expansions = self._expansions(parents, cache)
        pending = [(n, e, j) for n, e in expansions for j in e.pending()]
        if pending:
            self._evaluate(pending)
        kids = self._kids(expansions)
        if self.check:
            for kid in kids:
                bound = kid_bound(kid)
                if bound is not None and kid.fields()["alive"] and key_of(kid) > bound:
                    self.stats["violations"].append([kid.move, bound, key_of(kid)])
        return kids

    def _step(self, beam, cache):
        """The next beam: children forked best bound first, each batch only of those that
        could still displace the ``width``-th survivor, then selected as from all."""
        if self.check:
            return self._select(self._expand(beam, cache))
        expansions = self._expansions(beam, cache)
        pending = sorted(
            ((n, e, j) for n, e in expansions for j in e.pending()),
            key=lambda job: _order(line_bound(job[0], job[1].bounds[job[2]])),
        )
        while True:
            picked = self._select(self._kids(expansions))
            if len(picked) == self.width:
                cut = (key_of(picked[-1]), picked[-1].frames)
                live = [job for job in pending if may_displace(job, cut)]
                self.stats["bounded"] += len(pending) - len(live)
                pending = live
            if not pending:
                return picked
            chunk, pending = pending[:CPUS], pending[CPUS:]
            self._evaluate(chunk)

    def _select(self, kids):
        """The best ``width`` living ``kids``, best first, one per machine state."""
        fields = {f: np.array([n.fields()[f] for n in kids]) for f in FIELDS}
        alive, keys = bp.lexicographic(fields) if kids else (np.zeros(0, bool), [])
        live = np.flatnonzero(alive)
        pick = bp.keep_best(
            identity([kids[i].key for i in live]),
            [np.asarray(k)[live] for k in keys],
            np.array([kids[i].frames for i in live], dtype=np.int64),
            self.width,
        )
        return [kids[live[i]] for i in pick]

    def search(self, root, desc="search"):
        """The best node ``depth`` plies (or a win) below ``root``."""
        beam, best, cache = [root], root, {}
        t0, c0, f0 = time.time(), self.stats["children"], self.stats["frames"]
        pbar = tqdm(
            range(self.depth),
            desc=desc,
            unit="ply",
            leave=False,
            disable=not self.progress,
        )
        with pbar as progress:
            for _ in progress:
                beam = self._step(beam, cache)
                if not beam:
                    break
                best = beam[0]
                dt = max(time.time() - t0, 1e-9)
                progress.set_postfix(
                    kids_s=round((self.stats["children"] - c0) / dt, 1),
                    frames_s=round((self.stats["frames"] - f0) / dt),
                    eye=round(best.fields()["eye"] / 256, 3),
                    energy=best.facts["energy"],
                )
                if best.facts["won"]:
                    break
        self._cache = cache
        return best

    def run(self, max_actions=120):
        """Play until won, ``max_actions`` moves, or no line survives; whether won."""
        root = Node(self.bm)
        if self.live is not None:
            self._mirror("entry", ENTRY, prog.execute(self.bm.clone(), ENTRY))
        pbar = tqdm(
            total=max_actions, desc="moves", unit="move", disable=not self.progress
        )
        with pbar as progress:
            while len(self.trace) < max_actions and not self.won and not self.diverged:
                line = self.search(root, f"move {len(self.trace) + 1}").line()
                if not line:
                    self.log("search: no surviving move")
                    break
                for planned in line[: self.commit]:
                    root = self._execute(planned)
                    progress.update(1)
                    if self.won or self.diverged or len(self.trace) >= max_actions:
                        break
        return self.won and not self.diverged

    def _mirror(self, name, program, outcome):
        """Interpret ``program`` on ``live`` and audit its facts against ``outcome``'s,
        the session's; a disagreement is ``diverged``, with where their records part."""
        live = prog.interpret(self.live, program)
        facts, got = kbd_prog.move_facts(outcome), kbd_prog.move_facts(live)
        n = len(self.trace) + (name != "entry")
        self.live_audit.append([n, name, facts, got])
        if got != facts:
            diff = {k: [facts.get(k), got.get(k)] for k in {**facts, **got}}
            diff = {k: v for k, v in diff.items() if v[0] != v[1]}
            self.diverged = {
                "move": n,
                "name": name,
                "facts": diff,
                "records": kbd_prog.first_difference(outcome, live),
            }
            self.log(f"LIVE MISMATCH at move {n} {name}: session vs live {diff}")

    def _execute(self, planned):
        """Play ``planned``'s move on the session and return the new root, logging any
        difference from what the search's child did."""
        name, tile, _k = planned.move
        outcome = prog.execute(self.bm, planned.program)
        facts = kbd_prog.move_facts(outcome)
        if self.live is not None:
            self._mirror(name, planned.program, outcome)
        node = Node(self.bm, None, None, (planned.move, planned.program), facts)
        self.audit.append([len(self.trace) + 1, name, list(tile), planned.facts, facts])
        if facts != planned.facts or node.key != planned.key:
            self.log(
                f"search: move {len(self.audit)} {name}: {planned.facts} != {facts}"
            )
        if name == "relocate":
            ram = self.bm.m.ram
            tile = (
                ram[mm.OBJECTS_X + facts["slot"]],
                ram[mm.OBJECTS_Y + facts["slot"]],
            )
        self.frames += facts["frames"]
        eye = round(node.fields()["eye"] / 256, 3)
        self.trace.append(
            [self.frames, name, list(map(int, tile)), facts["energy"], eye]
        )
        self.won = bool(facts["won"])
        return node


def ply_batches(landscape, ply, width=16, depth=30, branch=3, commit=1, moves=False):
    """``[(base EmuMon, [programs])]``: the :func:`prog.forks` batches of search ply
    ``ply`` (0: the root's) from the entry to ``landscape``; ``moves`` pairs each program
    with its move."""
    from driver.emu import EmuMon
    from sentinel import emu

    player = SearchPlayer(EmuMon(emu.enter(landscape)), width, depth, branch, commit)
    beam, cache = [Node(player.bm)], {}
    for _ in range(ply):
        beam = player._select(player._expand(beam, cache))
    return [
        (node.bm, programs if moves else [p for _m, p in programs])
        for node, programs in zip(beam, batches(beam, branch))
    ]


def main(argv=None):
    """Time one search ply's fork batches: ``python -m driver.search_player 335 4``."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("landscape", type=int)
    parser.add_argument("ply", type=int)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="wide")
    parser.add_argument("--each", action="store_true", help="time programs one by one")
    args = parser.parse_args(argv)
    work = ply_batches(args.landscape, args.ply, moves=True, **PRESETS[args.preset])
    rows = []
    for base, programs in tqdm(work, desc="batches"):
        jobs = [[p] for p in programs] if args.each else [programs]
        for job in jobs:
            fell, t0 = prog.STATS["fallback"], time.time()
            outs, _ = prog.forks(base, [p for _m, p in job])
            rows.append(
                {
                    "moves": [list(m) for m, _p in job],
                    "frames": sum(o.frames for o in outs),
                    "seconds": round(time.time() - t0, 3),
                    "fallbacks": prog.STATS["fallback"] - fell,
                }
            )
    print(json.dumps(rows, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
