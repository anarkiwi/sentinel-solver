"""Exact inter-action frame counts, recovered from the recorded enemy clock.

$130C adds $CD per frame and runs $1317 only on the carry, which decrements the
cooldowns only on every third carry ($0C50). So $1335 fixes the frame count mod 256,
$0C50 lifts it to mod 768, and the $0C28 rotation sawtooth picks the multiple.
"""

from sentinel import memmap as mm
from sentinel.enemies import ROTATION_COOLDOWN_RELOAD

STEP = mm.COOLDOWN_BRESENHAM_STEP
_MOD = 256
_GATE = 3
PERIOD = _MOD * _GATE  # (bres, gate) repeat every 768 frames
_INV_STEP = pow(STEP, -1, _MOD)
_ROUND_FRAMES = _GATE * _MOD / STEP


def carries(bres0, n):
    """$130C carries in `n` frames from accumulator `bres0`."""
    return (bres0 + STEP * n) // _MOD


def decrements(bres0, gate0, n):
    """$1317 cooldown decrements in `n` frames: every third carry, offset by the gate."""
    c = carries(bres0, n)
    return 0 if c <= gate0 else (c - gate0 + 2) // _GATE


def gate_after(bres0, gate0, n):
    """The $0C50 gate after `n` frames."""
    c = carries(bres0, n)
    return gate0 - c if c <= gate0 else 2 - ((c - gate0 - 1) % _GATE)


def rounds_between(a, b, strict=True):
    """Cooldown rounds between two recorded clocks, by consensus of the enemies.

    An enemy whose cooldown reloaded or sits on the stick value cannot vote; `strict`
    demands unanimity among voters, which is what makes the count exact.

    One voter is enough: `span_frames` still has to satisfy (bres, gate) AND the
    decrement count jointly, so a wrong delta yields no candidate rather than a wrong
    one. Demanding two voters only threw away every single-enemy board.
    """
    votes = {}
    for ca, cb in zip(a, b):
        ra, rb = ca["rot_cooldown"], cb["rot_cooldown"]
        if ra <= 1 or rb <= 1:
            continue
        if rb <= ra:
            d = ra - rb
        elif strict:
            continue
        else:
            d = (ra - 1) + (ROTATION_COOLDOWN_RELOAD - rb)
        votes[d] = votes.get(d, 0) + 1
    if not votes or (strict and len(votes) > 1):
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def _base(ev, nxt):
    """Smallest positive frame count matching both recorded (bres, gate) pairs."""
    b0, g0 = ev.get("cooldown_bresenham"), ev.get("cooldown_gate")
    b1, g1 = nxt.get("cooldown_bresenham"), nxt.get("cooldown_gate")
    if None in (b0, g0, b1, g1):
        return None
    r = ((b1 - b0) * _INV_STEP) % _MOD
    for k in range(_GATE):
        n = r + k * _MOD
        if n and gate_after(b0, g0, n) == g1:
            return n
    return None


def span_frames(ev, nxt, limit=8):
    """(frames, exact) between two events' pre-action clocks, or (None, False).

    Exact when every voting enemy agrees on an unreloaded round count, which pins the
    multiple of 768 outright; otherwise the reload-tolerant estimate picks the nearest.
    """
    n0 = _base(ev, nxt)
    if n0 is None:
        return None, False
    b0, g0 = ev["cooldown_bresenham"], ev["cooldown_gate"]
    cands = [n0 + k * PERIOD for k in range(limit)]
    exact = rounds_between(ev["enemy_clock"], nxt["enemy_clock"])
    if exact is not None:
        hits = [n for n in cands if decrements(b0, g0, n) == exact]
        if len(hits) == 1:
            return hits[0], True
    loose = rounds_between(ev["enemy_clock"], nxt["enemy_clock"], strict=False)
    if loose is None:
        return n0, False
    want = loose * _ROUND_FRAMES
    return min(cands, key=lambda n: abs(n - want)), False


def seed_clock(state, ev):
    """Restore an event's recorded enemy clock into `state`, making phase an input
    rather than something a replay must earn back through its own cost model."""
    mem = state.mem
    for e in ev.get("enemy_clock") or ():
        slot = e["slot"]
        state.obj_h_angle[slot] = e["h_angle"]
        state.obj_v_angle[slot] = e["v_angle"]
        mem[mm.ROTATION_SPEED_TABLE + slot] = e["rot_step"] & 0xFF
        mem[mm.ENEMIES_ROTATION_COOLDOWN + slot] = e["rot_cooldown"]
        mem[mm.ENEMIES_DRAINING_COOLDOWN + slot] = e["drain_cooldown"]
        mem[mm.ENEMIES_UPDATE_COOLDOWN + slot] = e["update_cooldown"]
    if ev.get("cooldown_bresenham") is not None:
        mem[mm.COOLDOWN_BRESENHAM] = ev["cooldown_bresenham"]
        mem[mm.COOLDOWN_GATE] = ev["cooldown_gate"]
