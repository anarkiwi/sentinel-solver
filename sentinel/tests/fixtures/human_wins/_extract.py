"""Distil recorded HUMAN win logs into compact, permanent, non-copyrighted fixtures.

The three source logs are RAW ``watch_play/2`` recordings (gitignored, NEVER
committed -- each per-tick record carries a base64 dump of the loaded game's
memory ``[0x0000, 0x0CFF]``, which is copyrighted game data):

    out/play_20260707_193356.jsonl   entered code 0    -> generate seed 0
    out/play_20260707_194413.jsonl   entered code 42   -> generate seed 66
    out/play_20260707_203210.jsonl   entered code 335  -> generate seed 821

The header's ``landscape`` field is the CODE the player entered; the ROM's actual
board is generated from a different seed (42 -> 66, 335 -> 821, recovered by
matching ``sentinel.landscape.generate(seed)`` terrain byte-for-byte against the
logged terrain).  Only landscape 0 has code == seed.  The seed is what regenerates
the board, so it is what the fixture stores.

This distiller keeps ONLY reproducible game STATE -- object coordinates / types /
heights / flags, player position, aim angles, cursor, energy -- never any raw
``mem`` bytes and never the terrain (the terrain is regenerated at test time with
the audited byte-exact :func:`sentinel.landscape.generate`).  That is the
"cache for fixtures" the repo permits.

One fixture EVENT per player ACTION.  Actions are recovered by walking the
change-bracketed records (schema ``watch_play/2``: a record carrying a ``bracket``
field is emitted post-action) and DIFFING each bracket's object table against the
previous bracket's (the initial record ``t~=0`` seeds the first diff):

    a newly-occupied slot at tile T of type X  => create X at T
    a slot that became empty at tile T         => absorb (its old type) at T
    neither, player slot moved                 => transfer to the new player tile

The previous bracket is the pre-action world (``objects``, ``energy``, ``do_los``,
player position); the current (change) bracket carries the aim closest to the
fire (``hang``/``vang``/``cursor`` -- the sights settle on the built tile in the
change record).  Records at/after the first ``done_flag`` are post-win
next-landscape noise and are dropped.

Run ``python -m sentinel.tests.fixtures.human_wins._extract`` (with the raw logs
present) to regenerate the committed ``ls*.json`` fixtures.
"""

import base64
import json
import os

from sentinel import landscape, memmap as mm
from sentinel.terrain import tile_byte

_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
_OUT = os.path.join(_ROOT, "out")
_HERE = os.path.dirname(os.path.abspath(__file__))

# fixture filename (keyed by the entered code, the log's identity) -> (log, seed)
SOURCES = {
    "ls0.json": ("play_20260707_193356.jsonl", 0, 0),
    "ls42.json": ("play_20260707_194413.jsonl", 42, 66),
    "ls335.json": ("play_20260707_203210.jsonl", 335, 821),
}

_BUILDABLE = (mm.T_ROBOT, mm.T_TREE, mm.T_BOULDER)


def _occupied(mem):
    """{slot: [x, y, z_height, z_frac, type, flags]} for every occupied slot."""
    out = {}
    for s in range(mm.NUM_SLOTS):
        flags = mem[mm.OBJECTS_FLAGS + s]
        if flags & 0x80:
            continue
        out[s] = [
            mem[mm.OBJECTS_X + s],
            mem[mm.OBJECTS_Y + s],
            mem[mm.OBJECTS_Z_HEIGHT + s],
            mem[mm.OBJECTS_Z_FRACTION + s],
            mem[mm.OBJECTS_TYPE + s],
            flags,
        ]
    return out


def _mem(rec):
    return base64.b64decode(rec["mem"])


def _load_records(path):
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    return [json.loads(ln) for ln in lines[1:]]  # line 0 is the header


def _classify(pre_objs, post_objs, post_rec):
    """(verb, otype, target) for the action between the two object tables."""
    added = sorted(s for s in post_objs if s not in pre_objs)
    removed = sorted(s for s in pre_objs if s not in post_objs)
    if added:
        # a rare bracket window holds >1 new object (coincident enemy/rapid build):
        # prefer a player-buildable, lowest slot -- deterministic.
        pick = next((s for s in added if post_objs[s][4] in _BUILDABLE), added[0])
        o = post_objs[pick]
        return "create", o[4], [o[0], o[1]]
    if removed:
        pick = next((s for s in removed if pre_objs[s][4] in _BUILDABLE), removed[0])
        o = pre_objs[pick]
        return "absorb", o[4], [o[0], o[1]]
    pl = post_rec["player"]
    return "transfer", mm.T_ROBOT, [pl["x"], pl["y"]]


def extract(path, entered_code, seed):
    """Distil one log into the fixture dict."""
    recs = _load_records(path)
    initial = recs[0]  # t~=0 snapshot: the pre-state of the first action
    brackets = []
    for r in recs:
        if "bracket" not in r:
            continue
        if r.get("player") is None or r.get("done_flag"):
            break  # post-win noise
        brackets.append(r)

    events = []
    prev_rec = initial
    prev_objs = _occupied(_mem(initial))
    for post in brackets:
        post_objs = _occupied(_mem(post))
        verb, otype, target = _classify(prev_objs, post_objs, post)
        pre_pl = prev_rec["player"]
        post_pl = post["player"]
        events.append(
            {
                "verb": verb,
                "otype": otype,
                "target": target,
                # position from the pre-action record; aim from the change (post)
                # record -- the sights settle on the fired tile there.
                "player": {
                    "slot": pre_pl["slot"],
                    "x": pre_pl["x"],
                    "y": pre_pl["y"],
                    "z": pre_pl["z"],
                    "zf": pre_pl["zf"],
                    "hang": post_pl["hang"],
                    "vang": post_pl["vang"],
                },
                "cursor": [post["cursor"][0], post["cursor"][1]],
                "energy": prev_rec["energy"],
                "do_los": prev_rec["do_los"],
                "objects": [[s] + prev_objs[s] for s in sorted(prev_objs)],
            }
        )
        prev_rec = post
        prev_objs = post_objs

    return {
        "landscape": seed,
        "entered_code": entered_code,
        "source_log": os.path.basename(path),
        "n_events": len(events),
        "events": events,
    }


def main():
    for fname, (log, code, seed) in SOURCES.items():
        path = os.path.join(_OUT, log)
        if not os.path.exists(path):
            print(f"skip {fname}: raw log {path} absent (gitignored)")
            continue
        # sanity: the recovered seed's terrain must be the logged terrain.
        _assert_seed(path, seed)
        data = extract(path, code, seed)
        out = os.path.join(_HERE, fname)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        print(
            f"wrote {fname}: seed={seed} code={code} "
            f"events={data['n_events']} size={os.path.getsize(out)}B"
        )


def _assert_seed(path, seed):
    recs = _load_records(path)
    mem = _mem(recs[0])
    gen = landscape.generate(seed)
    for y in range(mm.N):
        for x in range(mm.N):
            lo = ((x << 3) & 0xE0) | (y & 0x1F)
            lb = mem[((x & 3) + 4) * 256 + lo]
            gb = tile_byte(gen, x, y)
            if lb >= mm.OBJECT_TILE or gb >= mm.OBJECT_TILE:
                continue
            assert lb == gb, f"seed {seed} terrain != log at ({x},{y})"


if __name__ == "__main__":
    main()
