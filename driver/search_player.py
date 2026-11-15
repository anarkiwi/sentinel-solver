#!/usr/bin/env python3
"""One search core over the emulated game: a beam whose children are driver programs.

Moves are :func:`sentinel.beam_player.views`, children :func:`kbd_prog.compile_move`
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
from sentinel import beam_player as bp, memmap as mm
from sentinel.phase_player import WAIT_QUANTUM
from sentinel.state import State

PRESETS = {
    "wide": {"width": 16, "depth": 30, "branch": 3, "commit": 10},
    "narrow": {"width": 4, "depth": 12, "branch": 2, "commit": 1},
}
CPUS = len(os.sched_getaffinity(0))
FIELDS = ("alive", "won", "enemies", "eye", "energy")


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
    """A machine state on a line: its monitor and key, and the move and facts that
    reached it from ``parent``."""

    __slots__ = ("bm", "key", "parent", "move", "program", "facts", "frames")

    def __init__(self, bm, key=None, parent=None, move=None, facts=None):
        self.bm, self.key = bm, machine_key(bm) if key is None else key
        self.parent, self.facts = parent, facts
        self.move, self.program = move if move else (None, None)
        self.frames = (parent.frames if parent else 0) + (facts or {}).get("frames", 0)

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
            "eye": f["z"] * 256 + f["z_frac"],
            "energy": f["energy"],
        }


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
    """The search core on an ``EmuMon`` session; ``run`` plays by receding-horizon beam
    search.  ``trace`` rows are ``[frames, move, tile, energy, eye]``, ``audit`` rows
    ``[move no, move, tile, planned facts, executed facts]``."""

    def __init__(self, bm, width=16, depth=30, branch=3, commit=1, log=print):
        self.bm, self.log = bm, log
        self.width, self.depth, self.branch, self.commit = width, depth, branch, commit
        self.trace, self.audit = [], []
        self.won, self.frames = False, 0
        self.stats = {"children": 0, "frames": 0, "seconds": 0.0, "cpu_seconds": 0.0}
        self._cache = {}

    def _expand(self, parents, cache):
        """The children of ``parents``; each parent's expansion is memoized on its
        machine key in ``cache``, seeded from the previous search's."""
        fresh = [n for n in parents if n.key not in cache and n.key not in self._cache]
        fresh = list({n.key: n for n in fresh}.values())
        if fresh:
            t0, c0 = time.time(), time.process_time()
            work = batches(fresh, self.branch)
            for node, programs, (outs, twins) in zip(
                fresh, work, run_batches(fresh, work)
            ):
                cache[node.key] = [
                    (move, kbd_prog.move_facts(o), twin, machine_key(twin))
                    for move, o, twin in zip(programs, outs, twins)
                ]
                self.stats["children"] += len(outs)
                self.stats["frames"] += sum(o.frames for o in outs)
            self.stats["seconds"] += time.time() - t0
            self.stats["cpu_seconds"] += time.process_time() - c0
        kids = []
        for node in parents:
            got = cache.setdefault(node.key, self._cache.get(node.key))
            kids += [
                Node(twin, key, node, move, facts)
                for move, facts, twin, key in got
                if facts["status"] == prog.HALT_OK
            ]
        return kids

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
        with tqdm(range(self.depth), desc=desc, unit="ply", leave=False) as progress:
            for _ in progress:
                beam = self._select(self._expand(beam, cache))
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
        with tqdm(total=max_actions, desc="moves", unit="move") as progress:
            while len(self.trace) < max_actions and not self.won:
                line = self.search(root, f"move {len(self.trace) + 1}").line()
                if not line:
                    self.log("search: no surviving move")
                    break
                for planned in line[: self.commit]:
                    root = self._execute(planned)
                    progress.update(1)
                    if self.won or len(self.trace) >= max_actions:
                        break
        return self.won

    def _execute(self, planned):
        """Play ``planned``'s move on the session and return the new root, logging any
        difference from what the search's child did."""
        name, tile, _k = planned.move
        facts = kbd_prog.move_facts(prog.execute(self.bm, planned.program))
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
