"""Retrograde handover: the reconstructed board is the human's, and the phase player
closes the recorded ls335 endgame from it."""

import xml.etree.ElementTree as ET

from sentinel import memmap as mm
from sentinel.test_human_win_logs import _load
from sentinel.tests import human_regress as hr

_FIX = "ls335.json"
_LAST = _load(_FIX)["n_events"] - 1


def test_state_at_is_the_human_pre_action_board():
    """Player tile/energy and every enemy facing match the recorded event."""
    ev = _load(_FIX)["events"][_LAST - 3]
    st = hr.state_at(_FIX, _LAST - 3)
    assert st.player_xy() == (ev["player"]["x"], ev["player"]["y"])
    assert st.energy == ev["energy"]
    assert not st.mem[mm.PLAYER_NOT_ACTED]
    for e in ev["enemy_clock"]:
        assert st.obj_h_angle[e["slot"]] == e["h_angle"]
        assert st.mem[mm.ENEMIES_ROTATION_COOLDOWN + e["slot"]] == e["rot_cooldown"]


def test_planner_closes_the_recorded_ls335_endgame():
    """From the human's last recorded move the line is transfer then hyperspace."""
    rec = hr.attempt(_FIX, _LAST)
    assert rec["outcome"] == "won"
    assert [s["verb"] for s in rec["player_trace"]] == ["transfer", "hyperspace"]
    assert rec["human_action"]["verb"] == "transfer"


def test_regress_scan_reports_a_win(tmp_path):
    """A one-index scan runs the worker process and finds no losing handover."""
    out = hr.regress(_FIX, workers=1, indices=[_LAST])
    assert out["first_loss"] is None and out["last_win"] == _LAST
    assert out["attempts"][0]["won"] is True


def test_bisect_finds_the_boundary_without_walking_it(monkeypatch):
    """Against a synthetic oracle (wins from 90 up) the bisection lands on 89/90 and
    probes far fewer handovers than the interval is wide."""
    probed = []

    def fake_run(name, batch, _cap, _log):
        probed.extend(batch)
        out = {}
        for i in batch:
            rec = hr._capped(name, i, 0.0)
            rec["won"] = i >= 90
            rec["outcome"] = "won" if rec["won"] else "lost"
            out[i] = rec
        return out

    monkeypatch.setattr(hr, "_run", fake_run)
    out = hr.bisect(_FIX, workers=10)
    assert (out["first_loss"], out["last_win"]) == (89, 90)
    assert len(probed) < out["n_events"] // 3
    assert out["method"] == "bisect"


def test_capped_probe_is_escalated_before_being_called_a_loss(monkeypatch):
    """A capped attempt is re-run alone at the escalated cap; if it then wins it is
    not the boundary."""
    caps = []

    def fake_run(name, batch, cap, _log):
        caps.append(cap)
        rec = hr._capped(name, batch[0], cap)
        if cap > hr.CAP:
            rec["won"], rec["outcome"] = True, "won"
        return {batch[0]: rec}

    monkeypatch.setattr(hr, "_run", fake_run)
    results = {5: hr._capped(_FIX, 5, 1.0)}
    budgets = hr._budgets()
    assert hr._settle(_FIX, results, [5], budgets, lambda _m: None) is None
    assert caps == [budgets["cap"] * budgets["escalate"]]


def test_diagram_annotates_the_handover(tmp_path):
    """The SVG carries the human's next moves and the planner trace as arrows."""
    rec = hr.attempt(_FIX, _LAST)
    path = hr.diagram(_FIX, _LAST, rec, horizon=4, path=str(tmp_path / "d.svg"))
    root = ET.parse(path).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    n_human = len(_load(_FIX)["events"][_LAST:])
    assert len(root.findall(f"{ns}path")) == n_human + len(rec["player_trace"])
    text = " ".join(t.text or "" for t in root.findall(f".//{ns}text"))
    assert "hyperspace" in text and "WON" in text
