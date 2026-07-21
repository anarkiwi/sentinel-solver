"""Shared loaders for the recorded human-win telemetry (``out/play_*.jsonl``).

Each log is JSONL: line 0 is a header, every later line a record carrying a base64
``mem`` dump of [0, 0x0CFF] plus the ``player`` position at that frame.  The logs are
gitignored fixtures -- callers guard with ``os.path.exists`` and skip when absent.
"""

import base64
import json
import os

from sentinel import memmap as mm
from sentinel.state import State

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "out")


def log_path(name):
    """Absolute path of a ``play_*.jsonl`` log by file name."""
    return os.path.join(LOG_DIR, name)


def state_from_record(rec):
    """The exact :class:`~sentinel.state.State` a record's ``mem`` dump captures."""
    raw = base64.b64decode(rec["mem"])
    mem = bytearray(0x10000)
    mem[0 : len(raw)] = raw
    return State(mem)


def records(path):
    """Every record line of `path` (line 0 is the header)."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    return [json.loads(ln) for ln in lines[1:]]


def tile_ladder(path):
    """Consecutive ``(from_tile, to_tile, pre_move_record)`` triples, stopping at the
    first ``done_flag`` (post-win noise).  Records on one tile collapse to the LAST of
    the run -- the pre-FIRE state, since the player builds while standing -- so
    successive from/to tiles differ by construction.
    """
    runs = []
    for rec in records(path):
        if rec.get("done_flag"):
            break
        pl = rec.get("player")
        if not pl:
            continue
        tile = (pl["x"], pl["y"])
        if runs and runs[-1][0] == tile:
            runs[-1] = (tile, rec)
        else:
            runs.append((tile, rec))
    return [(runs[i][0], runs[i + 1][0], runs[i][1]) for i in range(len(runs) - 1)]


def human_events(fixture_path):
    """Real player actions from a ``human_wins/*.json`` fixture.

    Drops the recorder's two artifact classes: a DRAIN TICK ($1838) moves energy with no
    object change and no player move, which ``_extract._classify`` mints as a transfer
    onto the player's own tile; and an enemy DISCHARGE TREE ($1A5D) appears as a create,
    though a player can only create boulders and robots. ls335: 168 rows, 130 actions.
    """
    with open(fixture_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    out = []
    for ev in rec["events"]:
        pl = ev.get("player") or {}
        own_tile = tuple(ev.get("target") or ()) == (pl.get("x"), pl.get("y"))
        if ev.get("verb") == "transfer" and own_tile:
            continue
        if ev.get("verb") == "create" and ev.get("otype") == mm.T_TREE:
            continue
        out.append(ev)
    return out
