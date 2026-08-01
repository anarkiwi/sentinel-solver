"""Per-verb settle laws against frozen-live ground truth.

``settle_oracle_ls42.json`` pairs each frozen-run action's live-measured settle
bracket with the jennings cycle count of the ROM's own settle path from that step's
sim image, so the law's residuals carry no enemy-phase feedback.
"""

import json
import os
import statistics

import pytest

from sentinel import actions, settlecost, terrain
from sentinel.game import Game
from sentinel.tests import oracle

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "settle_oracle_ls42.json")

BIAS_TOL_F = 1.5
RMS_TOL_F = 1.8
MAX_TOL_F = 3.0


def _fixture():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def _replay():
    """Yield (row, model_frames) pricing each action exactly as playerbase._settle."""
    fix = _fixture()
    game = Game.typed(fix["landscape_typed"])
    st = game.state
    carried = None
    for row in fix["rows"]:
        ach = row["aim_ach"]
        if ach:
            st.obj_h_angle[st.player] = ach["h"]
            st.obj_v_angle[st.player] = ach["v"]
            st.mem[0x0CC6], st.mem[0x0CC7] = ach["cur"]
            carried = {
                "h_angle": ach["h"],
                "v_angle": ach["v"],
                "cursor": list(ach["cur"]),
            }
        tile = tuple(row["tile"])
        verb = row["verb"]
        if verb == "transfer":
            top = terrain.top_object(st, *tile)
            model = settlecost.viewpoint_settle_frames(st, top, carried)
            assert actions.transfer(st, top)
        elif verb == "create":
            clone = st.clone()
            slot = actions.create(clone, row["otype"], tile)
            model = settlecost.action_settle_frames(clone, slot, carried)
            assert actions.create(st, row["otype"], tile) is not None
        else:
            slot = terrain.top_object(st, *tile)
            clone = st.clone()
            plot = clone.clone()
            actions.absorb(plot, slot)
            clone.obj_flags[slot] = 0x80
            model = settlecost.action_settle_frames(
                clone, slot, carried, plot_state=plot
            )
            assert actions.absorb(st, slot)
        yield row, model


def test_fixture_shape():
    fix = _fixture()
    assert fix["landscape_typed"] == 42
    rows = fix["rows"]
    assert len(rows) == 25
    assert {r["verb"] for r in rows} == {"absorb", "create", "transfer"}
    assert sum(1 for r in rows if not r["live_ok"]) == 1  # p25, the blind reuse


def test_measured_settles_split_by_verb():
    """The live settles differ by verb, as the per-scene law predicts: a create
    replots the strip WITH the new object, an absorb with it erased."""
    fix = _fixture()

    def meas(verb):
        return [
            r["measured_frames"]
            for r in fix["rows"]
            if r["verb"] == verb and r["live_ok"]
        ]

    assert statistics.mean(meas("create")) - statistics.mean(meas("absorb")) > 15


def test_settle_foreground_rate_is_composed():
    """SETTLE_FG = PAL - short IRQs - settle-regime body - store-heavy steal."""
    body = settlecost.SETTLE_BODY_FLAT + (205.0 / 256.0) * (
        settlecost.SETTLE_BODY_CARRY - settlecost.SETTLE_BODY_FLAT
    )
    assert settlecost.SETTLE_BODY == pytest.approx(body)
    assert settlecost.SETTLE_FG == pytest.approx(
        19656 - 477 - body - settlecost.SETTLE_STEAL
    )
    assert 17300 < settlecost.SETTLE_FG < 17700


def test_transfer_segment_constants_match_the_fixture_marks():
    """FILL_1090/STATUS_9508/STATUS_98B2 are constant across scenes; TAB_3700 stays
    in its band; $245B is scene-dependent and carries its own exact model."""
    fix = _fixture()
    for r in fix["rows"]:
        if r["verb"] != "transfer":
            continue
        m = r["marks"]
        assert m["0x35c3"] - m["0x35c0"] == settlecost.FILL_1090
        assert m["0x35cc"] - m["0x35c9"] == settlecost.STATUS_98B2
        # $9508 plots the energy meter, so its length breathes a few cycles with E
        assert abs(m["0x35c9"] - m["0x35c6"] - settlecost.STATUS_9508) < 20
        assert abs(m["0x35c0"] - m["0x35bd"] - settlecost.TAB_3700) < 20_000
        assert 2_000_000 < m["0x35bd"] - m["0x35ba"] < 3_500_000


@pytest.mark.oracle
def test_settle_law_predicts_the_frozen_measurement():
    """Every verb's law lands within MAX_TOL_F of the live bracket, unbiased."""
    if not oracle.available():
        pytest.skip("game image absent: object render terms fall back to floors")
    errs = {}
    for row, model in _replay():
        meas = row["measured_frames"]
        if not row["live_ok"] or meas is None or meas < 10:
            continue
        errs.setdefault(row["verb"], []).append(model - meas)
    for verb, es in errs.items():
        bias = statistics.mean(es)
        rms = (sum(e * e for e in es) / len(es)) ** 0.5
        worst = max(abs(e) for e in es)
        assert abs(bias) < BIAS_TOL_F, f"{verb} bias {bias:+.2f} f"
        assert rms < RMS_TOL_F, f"{verb} rms {rms:.2f} f"
        assert worst < MAX_TOL_F, f"{verb} worst {worst:.2f} f"
    assert sorted(errs) == ["absorb", "create", "transfer"]
    assert len(errs["create"]) == 8 and len(errs["absorb"]) == 11


@pytest.mark.oracle
def test_uturn_tune_is_24_frames():
    """The #$28 tune $1B84 starts decodes to 24 note-hold frames ($AB50 table)."""
    if not oracle.available():
        pytest.skip("game image absent")
    with open(oracle.IMG, "rb") as fh:
        img = fh.read()
    pos, length, total = 0x28, 0, 0
    while True:
        b = img[0xAB50 + pos]
        pos += 1
        if b == 0xFF:
            break
        if b >= 0xC8:
            length = ((b - 0xC8) * 4) & 0xFF
        else:
            total += length
    assert total == settlecost.TUNE_FRAMES[0x28] == 24
