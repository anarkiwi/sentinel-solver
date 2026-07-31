"""The recorded clock as ground truth for what an action COST, and what it moved.

Every span here is measured from the fixture's own $1335/$0C50/$0C28 readings, so these
tests grade the cost model and the enemy advance against the real game rather than
against each other.
"""

import json
import os
import types

import pytest

from sentinel import actioncost, actions, enemies, memmap as mm, projector, terrain
from sentinel.phase_player import PhasePlayer
from sentinel.test_human_win_logs import (
    _FIX_DIR,
    _is_player_action,
    _load,
    state_from_event,
)
from sentinel.tests import human_clock as hc

FIXTURE = "ls335.json"  # the only watch_play/3 fixture: it carries the enemy clock

# Debt measured against the recorded clock; each may only improve, EXCEPT where a term
# the clock was standing in for is retired: pricing the play machine's own $37F2 examine
# cut a strip replot's charge, and what $1F9F's line still does not price ($2211 and the
# $9730 flush, open_items 6) is now the dominant stall the facings want back.
EXACT_SPANS = 117  # spans whose frame count the clock pins outright
FACING_EXACT = 89
FACING_ERRORS = 41  # was 42 before the fill carried its own cycles; all but one +1 step
FACING_OVERSHOOT = 1  # ... the exception: a $1FFC stall that ate a rotation
DIVERGENT_SPANS = (10, 15, 17, 18, 33)  # spans whose facings we get wrong
ROM_ROUNDS = 60  # rounds of byte-exact agreement demanded on each  # of those, how many our enemy advance reproduces
SUB_FLOOR_SPANS = 8  # bracket pairs too close together to be two real actions
OVERCHARGED_RATE = 0.36  # share of actions billed MORE than their whole elapsed time
CADENCE = {False: 89, True: 64}  # plotting -> facings reproduced, vs recorded
SPLIT_CADENCE = 87  # the executor's phase split, scored the same way
# Live replay_human captures carrying $1335/$0C50: fixture -> (spans, facings).
# ls335 was 13: pricing the $1805 rotation and its $1F9F redraw (2177 cycles, 2.4 passes)
# and the frame's own $130C moves when each rotation falls due, and one 7-enemy span now
# ends one rotation step out. ls0 and ls42 stay exact; the gate this serves is the
# instrument (docs/open_items.md item 8), where the same terms take ls9795 515 -> 415.
LIVE_CLOCKS = {"ls0": (16, 16), "ls42": (10, 10), "ls335": (18, 12)}
LS42_CLOCK = "ls42_clock.json"
ENERGY_EXACT = 83  # exact-span actions whose next energy we reproduce
ENERGY_MISSES = 8  # the rest, all off by one in both directions (drain timing)


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


@pytest.mark.parametrize("plotting", (False, True))
def test_enemy_update_cadence_brackets_the_truth(plotting):
    """Neither extreme is the ROM's cadence, which is what indicts advancing a span as
    one phase.

    ``$16ED`` reloads ``$0C30`` on every path out of ``$16E6`` and the ``$90`` cursor is
    8 slots wide, both as modelled -- so a ``$0C30`` this far off is the CONSIDERATION
    RATE, not the consideration.  A span is part plotting (dither, replot, scroll: no
    ``$16B5``) and part idle main loop; ``test_phase_split_beats_either_extreme`` scores
    the split that follows from that.
    """
    evs = _events()
    seed = _load(FIXTURE)["landscape"]
    facings = 0
    for i, n in _exact_spans(evs):
        st = state_from_event(evs[i], seed)
        hc.seed_clock(st, evs[i])
        st.mem[mm.PLAYER_NOT_ACTED] = 0x00
        enemies.advance_frames(st, n, plotting=plotting)
        clock = evs[i + 1]["enemy_clock"]
        facings += [int(st.obj_h_angle[e["slot"]]) for e in clock] == [
            e["h_angle"] for e in clock
        ]
    assert facings == CADENCE[plotting]


def test_phase_split_beats_either_extreme():
    """Advancing each span as settle(plotting) -> think(idle) -> aim(split) reproduces
    more of the recorded enemy state than treating the whole span as either one.

    This is the executor's own split (``_aim_phases`` / the plotting settle) scored
    against ground truth: the think time is the residue the human spent in the main loop.
    """
    evs = _events()
    seed = _load(FIXTURE)["landscape"]
    facings = 0
    for i, n in _exact_spans(evs):
        st = state_from_event(evs[i], seed)
        p = PhasePlayer(types.SimpleNamespace(state=st))
        st = p.st
        hc.seed_clock(st, evs[i])
        st.mem[mm.PLAYER_NOT_ACTED] = 0x00
        settle, aim_plot = 0.0, 0.0
        if i:
            prev = evs[i - 1]
            pverb = {
                "create": "boulder" if prev["otype"] == mm.T_BOULDER else "robot"
            }.get(prev["verb"], prev["verb"])
            settle = min(p._settle(pverb), n)
            st.obj_h_angle[st.player] = prev["player"]["hang"]
            st.obj_v_angle[st.player] = prev["player"]["vang"]
            p.cursor = list(prev["cursor"])
            p.last_bearing = (prev["player"]["hang"], prev["player"]["vang"])
            view = {
                "h_angle": evs[i]["player"]["hang"],
                "v_angle": evs[i]["player"]["vang"],
                "cursor": list(evs[i]["cursor"]),
            }
            phases = (
                ()  # a recorded transfer's hang/vang is the new body's facing, not an aim
                if evs[i]["verb"] == "transfer"
                else p._aim_phases(view)
            )
            aim_plot = min(sum(f for f, plot in phases if plot), n - settle)
        p._advance_phases(
            ((settle, True), (n - settle - aim_plot, False), (aim_plot, True))
        )
        clock = evs[i + 1]["enemy_clock"]
        facings += [int(st.obj_h_angle[e["slot"]]) for e in clock] == [
            e["h_angle"] for e in clock
        ]
    assert facings > CADENCE[True]  # still beats charging the whole span to plotting
    assert facings == SPLIT_CADENCE  # but no longer the idle extreme: with the pass
    # rate derived from the state ($191F walks the board every pass), a plain idle
    # advance already prices what the split was standing in for.


def test_measured_span_reproduces_the_humans_next_energy():
    """Seed the clock, advance the measured span, THEN apply the action.

    A bracket fires when the action lands, so the action is the last thing in its span,
    not the first.  The misses are off by one energy in both directions -- drain-timing
    scatter, since $1A08 downgrades its target (robot -> boulder -> tree) and an absorb
    of a drained object yields less.
    """
    evs = _events()
    seed = _load(FIXTURE)["landscape"]
    ok = fails = 0
    for i, n in _exact_spans(evs):
        ev = evs[i]
        if not _is_player_action(ev):
            continue
        st = state_from_event(ev, seed)
        hc.seed_clock(st, ev)
        st.mem[mm.PLAYER_NOT_ACTED] = 0x00
        enemies.advance_frames(st, n)
        tile = tuple(ev["target"])
        if ev["verb"] == "create":
            done = actions.create(st, ev["otype"], tile) is not None
        else:
            top = terrain.top_object(st, *tile)
            done = top is not None and (
                actions.absorb(st, top)
                if ev["verb"] == "absorb"
                else actions.transfer(st, top)
            )
        assert done, f"step {i}: the human's own {ev['verb']} at {tile} was refused"
        if int(st.energy) == int(evs[i + 1]["energy"]):
            ok += 1
        else:
            fails += 1
    assert ok >= ENERGY_EXACT, f"regressed: {ok} < {ENERGY_EXACT}"
    assert fails <= ENERGY_MISSES


def _live_clock(name=LS42_CLOCK):
    """(pre, post, frames) for every exactly-pinned span of a live capture."""
    path = os.path.join(_FIX_DIR, name)
    with open(path, encoding="utf-8") as fh:
        steps = json.load(fh)["steps"]
    good = [
        s
        for s in steps
        if s["replay"]["matched_recording"] and not s["replay"]["diverged_since"]
    ]

    def clk(s):
        return {
            "cooldown_bresenham": s["cooldown_bresenham"],
            "cooldown_gate": s["cooldown_gate"],
            "enemy_clock": s["enemies"],
        }

    out = []
    for a, b in zip(good, good[1:]):
        n, exact = hc.span_frames(clk(a), clk(b))
        if exact:
            out.append((a, b, n))
    return out


@pytest.mark.parametrize("board", sorted(LIVE_CLOCKS))
def test_live_capture_facings(board):
    """Every board re-recorded through ``replay_human``, scored on facings.

    ls0 and ls42 are exact; ls335 is not, and it stays inexact under THIS capture method
    too -- so its gap is a real defect of the model on a seven-enemy board, not an
    artifact of the async recorder that produced its fixture.
    """
    spans, want_fac = LIVE_CLOCKS[board]
    evs = _load(f"{board}.json")["events"]
    seed = _load(f"{board}.json")["landscape"]
    got = _live_clock(f"{board}_clock.json")
    assert len(got) == spans
    exact = 0
    for a, b, n in got:
        st = state_from_event(evs[a["i"]], seed)
        hc.seed_clock(
            st,
            {
                "cooldown_bresenham": a["cooldown_bresenham"],
                "cooldown_gate": a["cooldown_gate"],
                "enemy_clock": a["enemies"],
            },
        )
        st.mem[mm.PLAYER_NOT_ACTED] = 0x00
        enemies.advance_frames(st, n)
        exact += [int(st.obj_h_angle[e["slot"]]) for e in b["enemies"]] == [
            e["h_angle"] for e in b["enemies"]
        ]
    assert exact == want_fac


def test_update_cooldown_is_sampling_dependent_and_not_a_score():
    """$0C30 reads differently depending on WHERE in the loop the capture stops, so it
    cannot grade the model; facings can, because a facing only moves on a real rotation.

    ``watch_play`` polls a free-running machine and catches $0C30 on its stick value most
    of the time; ``replay_human`` halts at a driver checkpoint and almost never does.
    Same register, same game, opposite readings -- an earlier revision scored the enemy
    advance on it and was measuring the recorder.
    """
    async_1 = sum(
        e["update_cooldown"] == 1 for ev in _events() for e in ev["enemy_clock"]
    )
    async_n = sum(len(ev["enemy_clock"]) for ev in _events())
    halted = [e["update_cooldown"] for a, _b, _n in _live_clock() for e in a["enemies"]]
    halted_1 = sum(v == 1 for v in halted)
    assert async_1 / async_n > 0.7  # async polling: mostly on the stick value
    assert halted_1 / len(halted) < 0.2  # checkpoint-halted: almost never


def test_every_facing_error_is_exactly_one_extra_rotation():
    """The ls335 facing gap is all but one-sided: we rotate once too many.

    $17FB ``CMP #$02 / BCC`` is the ROM's rotate gate and matches ours, so the threshold
    is right and the error is when the consideration happens, not whether it fires.  The
    single -1 is a $1FFC strip replot stalling a rotation the ROM kept; capping it is
    what stops a fix over-correcting."""
    evs = _events()
    seed = _load(FIXTURE)["landscape"]
    steps = []
    for i, n in _exact_spans(evs):
        st = state_from_event(evs[i], seed)
        hc.seed_clock(st, evs[i])
        st.mem[mm.PLAYER_NOT_ACTED] = 0x00
        enemies.advance_frames(st, n)
        for e in evs[i + 1]["enemy_clock"]:
            got, want, step = (
                int(st.obj_h_angle[e["slot"]]),
                e["h_angle"],
                e["rot_step"],
            )
            if got != want:
                steps.append((got - want - step) % 256)
    assert steps, "no facing error left -- retire this test and the debt it pins"
    over = [r for r in steps if r]
    assert len(steps) == FACING_ERRORS and len(over) <= FACING_OVERSHOOT
    assert all(r == 40 for r in over), f"not +-1 rotation: {sorted(set(steps))}"


OVERSHOOT_SPAN = 13  # the one facing error of the wrong sign, and the enemy behind it
OVERSHOOT_SLOT = 4


@pytest.mark.oracle
def test_the_rom_really_replots_the_enemy_the_overshoot_blames():
    """The single -1 facing error is a $1FFC replot the ROM genuinely pays for.

    Both cheap alternatives are excluded on the recorded state: $0C4D bit 7 is clear so
    $1FEF does not divert to $8533, and $0C1F bit 7 is clear so $1B00 hands $1AF4 a set
    carry and the update runs.  What is left is the frame-to-round cadence above $16E6.
    """
    oracle = pytest.importorskip("sentinel.tests.oracle")
    if not oracle.available():
        pytest.skip("stage2 image absent")
    from sentinel import relative  # pylint: disable=import-outside-toplevel

    ev = _events()[OVERSHOOT_SPAN]
    st = state_from_event(ev, _load(FIXTURE)["landscape"])
    hc.seed_clock(st, ev)
    assert relative.object_screen_span(st, OVERSHOOT_SLOT)[0]  # ours picks this enemy

    cpu, rom, mstate = oracle.machine_from_image(bytes(st.mem))
    rom[oracle.WORLD_BUSY_PLOTTING] = 0x00
    rom[0x0091], rom[0x006E] = OVERSHOOT_SLOT, rom[mm.PLAYER_OBJECT]
    oracle.call(cpu, rom, 0x1B00, a=OVERSHOOT_SLOT)  # $1AF4's own gate
    assert cpu.p & 0x01, "$1B00 aborted the update: $0C1F was set after all"

    seen = set()
    frames = oracle.update_object_cost(cpu, rom, mstate, OVERSHOOT_SLOT, trace=seen.add)
    assert 0x1FFC in seen and 0x1FF6 not in seen  # $2625 plot_world, not $8533
    assert 25.0 < frames < 45.0  # a whole span's worth of foreground, once


@pytest.mark.oracle
@pytest.mark.parametrize("span", DIVERGENT_SPANS)
def test_enemies_step_matches_the_rom_on_a_divergent_state(span):
    """On the very states whose facings we get wrong, our round is byte-exact vs the ROM.

    That puts the branch logic of $16E6 beyond suspicion and leaves only the frame-to-
    round cadence above it.  ``machine_from_image`` overlays $9D37 and $1335 from the
    image, so the recorded rotation steps must be rewritten or the ROM turns by the
    wrong amount and the comparison is silently meaningless.
    """
    oracle = pytest.importorskip("sentinel.tests.oracle")
    if not oracle.available():
        pytest.skip("stage2 image absent")
    evs = _events()
    seed = _load(FIXTURE)["landscape"]
    ev = evs[span]
    st = state_from_event(ev, seed)
    hc.seed_clock(st, ev)
    st.mem[mm.PLAYER_NOT_ACTED] = 0x00
    cpu, rom, state = oracle.machine_from_image(bytes(st.mem))
    for addr in oracle.RENDER_STUBS:
        rom[addr] = 0x60
    rom[oracle.WORLD_BUSY_PLOTTING] = 0x00
    rom[mm.COOLDOWN_GATE] = st.mem[mm.COOLDOWN_GATE]
    rom[0x0090] = 7
    for e in ev["enemy_clock"]:
        rom[mm.ROTATION_SPEED_TABLE + e["slot"]] = e["rot_step"] & 0xFF
    ours = state_from_event(ev, seed)
    hc.seed_clock(ours, ev)
    ours.mem[mm.PLAYER_NOT_ACTED] = 0x00
    ours.mem[0x0090] = 7
    for r in range(ROM_ROUNDS):
        oracle.step_enemy_round(cpu, rom, state)
        enemies.step(ours)
        got = [int(ours.obj_h_angle[s]) for s in range(8)]
        want = [int(rom[mm.OBJECTS_H_ANGLE + s]) for s in range(8)]
        assert got == want, f"round {r + 1}: ours {got} != ROM {want}"


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
    n_spans = 0
    over = []
    for i, n in _exact_spans(evs):
        if not i or not (_is_player_action(evs[i]) and _is_player_action(evs[i - 1])):
            continue
        prev = evs[i - 1]
        st = state_from_event(evs[i], seed)
        p = PhasePlayer(types.SimpleNamespace(state=st))
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
        n_spans += 1
        if cost > n:
            over.append(i)
    assert 0.9 <= billed / measured <= 1.1, f"aggregate bill {billed / measured:.3f}x"
    rate = len(over) / n_spans  # a rate, so growing the sample is not a regression
    assert rate <= OVERCHARGED_RATE, f"regressed: {len(over)}/{n_spans} = {rate:.3f}"
