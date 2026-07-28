"""Checkpoint a live player so a stalled tick can be re-entered instead of replayed.

The game state is one 64 KB ``bytearray``, so a checkpoint is that image plus the
player scalars it does not carry; the rest of a player is cache keyed on a state
signature and rebuilds on demand.  See [fast_iteration.md](../../docs/fast_iteration.md).
"""

import copy
import pickle
import zlib

from sentinel import actions, enemies
from sentinel.phase_player import PhasePlayer
from sentinel.game import Game
from sentinel.state import State

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
    is CONSTRUCTED on, before the live image is written over it.
    """
    return {
        "tick": tick,
        "start_mem": bytes(player.st.mem),
        "mem": bytes(player.st.mem),
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
