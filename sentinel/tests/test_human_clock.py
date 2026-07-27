"""The recorded clock as ground truth for what an action COST, and what it moved.

Every span here is measured from the fixture's own $1335/$0C50/$0C28 readings, so these
tests grade the cost model and the enemy advance against the real game rather than
against each other.
"""

import types

import pytest

from sentinel import actioncost, enemies, memmap as mm, projector
from sentinel.astar_player import AStarPlayer
from sentinel.test_human_win_logs import _load, _is_player_action, state_from_event
from sentinel.tests import human_clock as hc

FIXTURE = "ls335.json"  # the only watch_play/3 fixture: it carries the enemy clock

# Debt measured against the recorded clock; each may only improve.
EXACT_SPANS = 91  # spans whose frame count the clock pins outright
FACING_EXACT = 67  # of those, how many our enemy advance reproduces
SUB_FLOOR_SPANS = 8  # bracket pairs too close together to be two real actions
OVERCHARGED_SPANS = 17  # actions we bill more for than the whole elapsed time


def _events(name=FIXTURE):
    return _load(name)["events"]


def _exact_spans(evs):
    """(i, frames) for every span the recorded clock pins exactly."""
    out = []
    for i in range(len(evs) - 1):
        n, exact = hc.span_frames(evs[i], evs[i + 1])
        if exact:
            out.append((i, n))
    return out


@pytest.mark.parametrize("bres0", (0, 1, 90, 127, 205, 255))
@pytest.mark.parametrize("gate0", (0, 1, 2))
def test_closed_form_matches_the_stepped_clock(bres0, gate0):
    """carries/decrements/gate_after are the $130C/$1317 loop in closed form."""
    bres, gate, dec = bres0, gate0, 0
    for n in range(1, 800):
        acc = bres + hc.STEP
        bres = acc & 0xFF
        if acc > 0xFF:
            if gate:
                gate -= 1
            else:
                dec += 1
                gate = 2
        assert hc.carries(bres0, n) == (bres0 + hc.STEP * n) // 256
        assert hc.decrements(bres0, gate0, n) == dec
        assert hc.gate_after(bres0, gate0, n) == gate


def test_recorded_clock_pins_the_span_of_most_actions():
    """The fixture's own clock measures what each action cost, no model involved."""
    spans = _exact_spans(_events())
    assert len(spans) == EXACT_SPANS
    assert all(n > 0 for _, n in spans)


def test_advance_frames_reproduces_the_cooldown_clock_exactly():
    """Seeded with the recorded clock and advanced by the measured span, the
    $1335/$0C50 accumulator lands on the next event's recorded reading every time."""
    evs = _events()
    seed = _load(FIXTURE)["landscape"]
    for i, n in _exact_spans(evs):
        st = state_from_event(evs[i], seed)
        hc.seed_clock(st, evs[i])
        st.mem[mm.PLAYER_NOT_ACTED] = 0x00
        enemies.advance_frames(st, n)
        got = (st.mem[mm.COOLDOWN_BRESENHAM], st.mem[mm.COOLDOWN_GATE])
        want = (evs[i + 1]["cooldown_bresenham"], evs[i + 1]["cooldown_gate"])
        assert got == want, f"span {i} ({n}f): clock {got} != recorded {want}"


def test_enemy_facings_after_a_measured_span():
    """Over the same spans the FACINGS do not all follow: an enemy the ROM holds in a
    non-rotating branch of $16E6 rotates in ours.  Debt, pinned so it can only fall."""
    evs = _events()
    seed = _load(FIXTURE)["landscape"]
    exact = 0
    for i, n in _exact_spans(evs):
        st = state_from_event(evs[i], seed)
        hc.seed_clock(st, evs[i])
        st.mem[mm.PLAYER_NOT_ACTED] = 0x00
        enemies.advance_frames(st, n)
        slots = [e["slot"] for e in evs[i + 1]["enemy_clock"]]
        want = [e["h_angle"] for e in evs[i + 1]["enemy_clock"]]
        exact += [int(st.obj_h_angle[s]) for s in slots] == want
    assert exact >= FACING_EXACT, f"regressed: {exact} < {FACING_EXACT}"


def test_spans_below_the_rom_floor_are_not_two_actions():
    """A create/absorb cannot be followed by another action inside its own $1FA4 dither,
    nor a transfer inside its $35D5 tune wait.  Spans that short are one action recorded
    twice -- artifacts ``_is_player_action`` does not catch."""
    evs = _events()
    floors = {"transfer": projector.TUNE_TRANSFER_FRAMES}
    sub = [
        i
        for i, n in _exact_spans(evs)
        if _is_player_action(evs[i])
        and _is_player_action(evs[i + 1])
        and n < floors.get(evs[i]["verb"], actioncost.DITHER_FRAMES)
    ]
    assert len(sub) == SUB_FLOOR_SPANS, f"{sub}"


def test_billed_cost_against_measured_elapsed():
    """What the executor bills for an action, against what the action really took.

    ``span = settle(prev) + think + aim``, and think >= 0, so billing more than the whole
    span is proof of overcharge.  In aggregate the bill is close, and the measured side
    still CONTAINS think time, so a correct bill should sit under it.
    """
    evs = _events()
    seed = _load(FIXTURE)["landscape"]
    billed = measured = 0.0
    over = []
    for i, n in _exact_spans(evs):
        if not i or not (_is_player_action(evs[i]) and _is_player_action(evs[i - 1])):
            continue
        prev = evs[i - 1]
        st = state_from_event(evs[i], seed)
        p = AStarPlayer(types.SimpleNamespace(state=st))
        st = p.st
        st.obj_h_angle[st.player] = prev["player"]["hang"]
        st.obj_v_angle[st.player] = prev["player"]["vang"]
        p.cursor = list(prev["cursor"])
        p.last_bearing = (prev["player"]["hang"], prev["player"]["vang"])
        aim = (
            0.0  # a recorded transfer's hang/vang is the NEW body's facing, not an aim
            if evs[i]["verb"] == "transfer"
            else p._aim_frames(
                {
                    "h_angle": evs[i]["player"]["hang"],
                    "v_angle": evs[i]["player"]["vang"],
                    "cursor": list(evs[i]["cursor"]),
                }
            )
        )
        pverb = {"create": "boulder" if prev["otype"] == mm.T_BOULDER else "robot"}.get(
            prev["verb"], prev["verb"]
        )
        cost = aim + p._settle(pverb)
        billed += cost
        measured += n
        if cost > n:
            over.append(i)
    assert 0.9 <= billed / measured <= 1.1, f"aggregate bill {billed / measured:.3f}x"
    assert (
        len(over) <= OVERCHARGED_SPANS
    ), f"regressed: {len(over)} > {OVERCHARGED_SPANS}"
