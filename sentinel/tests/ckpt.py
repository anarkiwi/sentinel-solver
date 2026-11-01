"""Checkpoint a live player so a stalled tick can be re-entered instead of replayed.

The game state is one 64 KB ``bytearray``, so a checkpoint is that image plus the
player scalars it does not carry; the rest of a player is cache keyed on a state
signature and rebuilds on demand.  See [architecture.md](../../docs/architecture.md).
"""

import argparse
import copy
import glob
import hashlib
import os
import pickle
import zlib

from tqdm import tqdm

from sentinel import actions, enemies
from sentinel.phase_player import ARBITRATE_ACTIONS, POLICIES, PhasePlayer
from sentinel.game import Game
from sentinel.playerbase import _Views
from sentinel.state import State

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(os.path.dirname(_PKG), "out", "ckpt")

# Decision state absent from the memory image; a field a player lacks is skipped
FIELDS = (
    "cursor",
    "last_bearing",
    "frames",
    "trace",
    "fire_reason",
    "_stale",
    "waited",
)


def snapshot(player, tick=0):
    """A picklable checkpoint of ``player`` at decision tick ``tick``.

    Fields are DEEP-COPIED: they are live mutable objects, so references would make
    every checkpoint alias the last tick.  ``start_mem`` is the board a restored player
    is CONSTRUCTED on, before the live image is written over it.  ``cycle_residual`` is
    the play loop's carried debt: state the image does not hold, so a checkpoint without
    it replays the tail on a different enemy clock.
    """
    return {
        "tick": tick,
        "start_mem": bytes(player.st.mem),
        "mem": bytes(player.st.mem),
        "cycle_residual": player.st.cycle_residual,
        "pass_phase": player.st.pass_phase,
        "body_stage": player.st.body_stage,
        "body_index": player.st.body_index,
        "body_partial": player.st.body_partial,
        "body_paid": player.st.body_paid,
        "steal_residue": player.st.steal_residue,
        "fields": {
            name: copy.deepcopy(getattr(player, name))
            for name in FIELDS
            if hasattr(player, name)
        },
    }


def restore(snap, cls=PhasePlayer, **kwargs):
    """Rebuild a player of ``cls`` sitting exactly at the checkpointed tick.

    Fields are DEEP-COPIED out so replaying a restored player cannot mutate the
    checkpoint it came from.
    """
    player = cls(Game(State(bytearray(snap["start_mem"]))), **kwargs)
    player.st.mem[:] = snap["mem"]
    player.st.cycle_residual = snap.get("cycle_residual", 0)
    player.st.pass_phase = snap.get("pass_phase", 0)
    player.st.body_stage = snap.get("body_stage", 0)
    player.st.body_index = snap.get("body_index", 0)
    player.st.body_partial = snap.get("body_partial", -1)
    player.st.body_paid = snap.get("body_paid", 0)
    player.st.steal_residue = snap.get("steal_residue", 0)
    for name, value in snap["fields"].items():
        setattr(player, name, copy.deepcopy(value))
    return player


def save(snaps, path):
    with open(path, "wb") as handle:
        handle.write(zlib.compress(pickle.dumps(snaps), 6))


def load(path):
    with open(path, "rb") as handle:
        return pickle.loads(zlib.decompress(handle.read()))


def describe(player):
    """One line naming the tile and eye a checkpoint sits on."""
    st = player.st
    return (
        f"tile={st.player_xy()} eye={st.eye_z():.3f} E={st.energy} "
        f"foes={len(enemies.enemy_slots(st))} frames={player.frames}"
    )


def capture(player, max_actions=200, snaps=None):
    """``BasePlayer.run``, recording a checkpoint before every decision tick."""
    for tick in range(max_actions):
        player._observe()
        if actions.won(player.st):
            return True, "won"
        if player._dead():
            return False, "dead"
        if snaps is not None:
            snaps.append(snapshot(player, tick))
        player._tick()
    player._observe()
    return actions.won(player.st), "action cap"


def verify(snaps, backs=(5, 10, 20), cls=PhasePlayer, **kwargs):
    """Fidelity gate: re-entry must reproduce the run's own trace tail.

    Ground truth is the trace in the LAST snapshot; restoring at tick ``t`` and
    replaying ``last - t`` ticks must reproduce it entry for entry.  A row with zero
    ``actions`` proves nothing, so callers must require a non-zero one.
    """
    last = snaps[-1]
    truth = last["fields"]["trace"]
    rows = []
    for back in backs:
        index = len(snaps) - 1 - back
        if index < 0:
            continue
        snap = snaps[index]
        player = restore(snap, cls=cls, **kwargs)
        capture(player, last["tick"] - snap["tick"])
        rows.append(
            {
                "from_tick": snap["tick"],
                "actions": len(truth) - len(snap["fields"]["trace"]),
                "identical": player.trace[: len(truth)] == truth,
            }
        )
    return rows


def replay_to(landscape, n_actions, cls=PhasePlayer, max_ticks=10000, **kwargs):
    """Play ``landscape`` from its start and checkpoint the first decision tick whose
    trace already holds ``n_actions`` actions; None if the run ends first."""
    player = cls(Game.typed(landscape), **kwargs)
    with tqdm(total=n_actions, unit="action", desc=f"ls{landscape}") as progress:
        for tick in range(max_ticks):
            progress.update(min(len(player.trace), n_actions) - progress.n)
            if len(player.trace) >= n_actions:
                return snapshot(player, tick)
            player._observe()
            if actions.won(player.st) or player._dead():
                return None
            player._tick()
    return None


def source_signature():
    """12 hex chars of sha256 over every module a replay runs: the world model and
    the planner both decide which state a given action index lands on."""
    digest = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(_PKG, "*.py"))):
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:12]


def cached_replay(landscape, n_actions):
    """:func:`replay_to`, memoized on disk under ``out/ckpt`` per source signature.

    The image embeds the ROM's memory, so it is regenerated from the cached release
    and never committed."""
    path = os.path.join(
        CACHE_ROOT, source_signature(), f"ls{landscape:04d}_a{n_actions}.ckpt"
    )
    if os.path.exists(path):
        return load(path)[0]
    snap = replay_to(landscape, n_actions)
    if snap is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}"
        save([snap], tmp)
        os.replace(tmp, path)
    return snap


def explain(player, options=POLICIES):
    """What the planner weighs at ``player``'s tick: every affordable climb, best score
    first, with whether its landing holds, and every arbitrated option as its fork
    plays it -- the actions taken and the score they end on."""
    views = _Views(player.st)
    climbs = [
        {"score": c[0], "tile": c[1], "k": c[2], "cost": c[3], "gain": c[4]}
        | {"lands": player._lands(c)}
        for c in sorted(
            player._climb_candidates(views, True), key=lambda c: c[0], reverse=True
        )
    ]
    played = []
    for name in options:
        twin = player._fork()
        moved = bool(getattr(twin, name)(_Views(twin.st)))
        for _ in range(ARBITRATE_ACTIONS if moved else 0):
            if actions.player_dead(twin.st) or actions.won(twin.st):
                break
            twin._tick()
        played.append(
            {
                "option": name,
                "moved": moved,
                "score": player._score(twin) if moved else None,
                "actions": [
                    (verb, tile, energy, eye)
                    for _f, verb, tile, energy, eye in twin.trace
                ],
            }
        )
    return {"climbs": climbs, "options": played}


def main(argv=None):
    """Replay a landscape to an action index and dump what the planner weighs there."""
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("landscape", type=int)
    parser.add_argument("actions", type=int, help="actions already taken at the tick")
    args = parser.parse_args(argv)
    snap = cached_replay(args.landscape, args.actions)
    if snap is None:
        print("run ended before that action")
        return 1
    player = restore(snap)
    print(describe(player))
    for kind, rows in explain(player).items():
        print(kind)
        for row in rows:
            print("  " + " ".join(f"{key}={val}" for key, val in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
