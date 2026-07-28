"""Replay a human WIN through OUR executor and report the FIRST divergence.

Each recorded event carries the true board before that action, so applying the human's
own action through our machinery and comparing against the next event isolates one
falsifiable defect at a time, in order, with ground truth attached.

ls335 diverges at action 15 by ONE energy, from accumulated enemy phase.  The size of
that error is now MEASURED rather than inferred: ``human_clock.span_frames`` recovers the
exact frames the real game advanced between two events from the fixture's own
$1335/$0C50/$0C28 readings, and against those the bill is 0.993x in aggregate -- not the
~6% long that the quantised h_angle proxy suggested.  What the same measurement does
indict is elsewhere; see ``test_human_clock``.
"""

import sys
import types
from sentinel import memmap as mm
from sentinel.phase_player import PhasePlayer
from sentinel.test_human_win_logs import _is_player_action, _load, state_from_event
from sentinel.tests.human_clock import seed_clock

_PLAYER_TYPES = (mm.T_BOULDER, mm.T_ROBOT, mm.T_TREE)


def board(st):
    """(tile -> type) for every non-enemy object: what the player's actions changed."""
    out = {}
    for slot in range(mm.NUM_SLOTS):
        if st.is_empty(slot):
            continue
        t = int(st.obj_type[slot])
        if t in _PLAYER_TYPES:
            out[tuple(int(v) for v in st.tile_of(slot))] = t
    return out


def expected(ev):
    out = {}
    for row in ev["objects"]:
        x, y, otype = row[1], row[2], row[5]
        if int(otype) in _PLAYER_TYPES:
            out[(int(x), int(y))] = int(otype)
    return out


def replay(name, landscape=None, limit=None, quiet=False):
    data = _load(name)
    landscape = data["landscape"]  # the SEED; entered_code is the typed-in number
    evs = data["events"]
    st = state_from_event(evs[0], landscape)
    p = PhasePlayer(types.SimpleNamespace(state=st))
    st = p.st
    seed_clock(st, evs[0])  # phase is an input, not something the cost model must earn
    say = (lambda *a: None) if quiet else print
    say(f"{name}: replaying {len(evs)} human actions through our executor")
    for i, ev in enumerate(evs[: limit or len(evs)]):
        if not _is_player_action(ev):
            continue  # a discharge tree / drain tick is not an action to reproduce
        verb, otype, tile = ev["verb"], ev["otype"], tuple(ev["target"])
        fverb = {"create": "boulder" if otype == mm.T_BOULDER else "robot"}.get(
            verb, verb
        )
        p.st = st
        view = p._views_now().get(tile, band=True)
        if view is None:
            say(f"  [{i:3}] {verb:8} {tile}: DIVERGE -- no keyboard aim lands it")
            return i
        before_e = int(st.energy)
        ok = p._fire(fverb, tile, view)
        if not ok:
            say(
                f"  [{i:3}] {verb:8} {tile}: DIVERGE -- _fire refused "
                f"({p.fire_reason}), human did it at E={ev['energy']}"
            )
            return i
        nxt = evs[i + 1] if i + 1 < len(evs) else None
        if nxt is None:
            break
        want_xy = (nxt["player"]["x"], nxt["player"]["y"])
        got_xy = tuple(int(v) for v in st.player_xy())
        if got_xy != want_xy:
            say(
                f"  [{i:3}] {verb:8} {tile}: DIVERGE -- body at {got_xy}, "
                f"human at {want_xy}"
            )
            return i
        if int(st.energy) != int(nxt["energy"]):
            say(
                f"  [{i:3}] {verb:8} {tile}: DIVERGE -- energy {before_e}->"
                f"{int(st.energy)}, human {before_e}->{nxt['energy']}"
            )
            return i
        gb, eb = board(st), expected(nxt)
        if gb != eb:
            only_ours = {k: v for k, v in gb.items() if eb.get(k) != v}
            only_theirs = {k: v for k, v in eb.items() if gb.get(k) != v}
            say(f"  [{i:3}] {verb:8} {tile}: DIVERGE -- objects differ")
            say(f"          ours not theirs:   {only_ours}")
            say(f"          theirs not ours:   {only_theirs}")
            return i
    say("  no divergence found")
    return None


def first_divergence(name):
    """Index of the first action our executor cannot reproduce, or None."""
    return replay(name, quiet=True)


if __name__ == "__main__":
    replay(sys.argv[1] if len(sys.argv) > 1 else "ls335.json")
