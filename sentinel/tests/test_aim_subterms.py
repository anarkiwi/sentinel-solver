"""Per-sub-term aim-cost fidelity against the live-recorded ``aim_subframes``.

Aim cost is path-dependent (the sights toggle only fires when the committed bearing
misses, a pan's per-notch replot is scene-dependent), so a fused per-verb epsilon is
ill-posed: these pin the SUB-TERM mechanisms instead.
"""

import json
import os

from sentinel import playerbase as pb

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "live_aim_subframes.json"
)

_TOGGLE_BOUND = (
    pb.TOGGLE_FRAMES + pb.H_SCROLL
)  # two gated scans ($9678) + the last pan notch's queued unbuffering steps ($10EE 16 h / $1135 8 v), which must drain before the next scan fires


def _rows():
    with open(_FIXTURE) as fh:
        data = json.load(fh)
    return [(ls, r) for ls, rows in sorted(data.items()) for r in rows]


def _fresh_cursor_drives():
    """(label, pixels, measured) for each step whose cursor drive started at the $134C
    re-centre: a sights toggle ran (so the cursor was reset) and the reuse fast path did
    not re-drive it first (which it does when a same-bearing probe misses)."""
    out = []
    for ls, rows in sorted(json.load(open(_FIXTURE)).items()):
        prev = None
        for row in rows:
            ach = row["ach"]
            if ach is None:  # a transfer keys no aim at all
                prev = None
                continue
            stale = prev is None or prev[:2] != ach[:2]
            prev = ach
            if not row["toggle"] or not stale:
                continue
            px = max(abs(ach[2][i] - pb.SIGHTS_CENTRE[i]) for i in (0, 1))
            out.append((f"{ls}/{row['step']}", px, row["cursor"]))
    return out


def _panned(row):
    return row["pan_h"] > 0 or row["pan_v"] > 0


def test_measured_aim_is_fully_attributed_to_its_four_sub_terms():
    """``aim_exact`` (the whole-step bracket minus the settle bracket) equals the four
    timed primitives exactly, so no aim frame is spent outside pan/toggle/cursor -- the
    attribution every charge-vs-measured claim below rests on."""
    for ls, row in _rows():
        parts = row["pan_h"] + row["pan_v"] + row["toggle"] + row["cursor"]
        assert parts == row["aim_exact"], f"{ls}/{row['step']}"


def test_reused_bearing_spends_no_aim_sub_term():
    """A step reusing the committed bearing keys nothing: no toggle, no pan, and the
    cursor drive is its whole aim cost."""
    for ls, row in _rows():
        if row["toggle"]:
            continue
        assert not _panned(row), f"{ls}/{row['step']}: panned with no sights toggle"
        assert row["aim_exact"] == row["cursor"], f"{ls}/{row['step']}"


def test_transfer_aim_is_zero_only_under_bearing_reuse():
    """Every recorded transfer fired on the bearing its same-tile create left committed,
    so it keyed nothing -- the only case the model charges a transfer 0 aim for."""
    for ls, row in _rows():
        if row["verb"] != "transfer":
            continue
        assert row["aim_exact"] == 0, f"{ls}/{row['step']}"
        assert (row["pan_h"], row["pan_v"], row["toggle"], row["cursor"]) == (
            0,
            0,
            0,
            0,
        )


def test_sights_toggle_is_bounded_by_two_scans_plus_one_notch_unbuffer():
    """The toggle pair outlasts that bound only with NO pan between sights OFF and ON,
    when the SPACE auto-repeat lock ($1236, cleared only by a scan seeing SPACE up)
    swallowed the ON press and ``sights_set`` retried -- the sole recorded violator, and
    it has no pan (``kbd_aim._one_scan_press`` now re-arms the latch with an idle scan).
    """
    for ls, row in _rows():
        if row["toggle"] > _TOGGLE_BOUND:
            assert not _panned(row), (
                f"{ls}/{row['step']}: toggle {row['toggle']}f > {_TOGGLE_BOUND}f with a "
                "pan between OFF and ON -- not the $1236 re-arm gap"
            )
        elif _panned(row):
            assert row["toggle"] <= _TOGGLE_BOUND, f"{ls}/{row['step']}"


def test_cursor_drive_costs_a_scan_per_pixel_plus_at_most_the_repeat_ramp():
    """``move_sights`` moves 1px per gated scan, and $0CC8 (reloaded #$6B at $11E0 on any
    scan with no direction key down) skips one scan per set bit first -- so a drive costs
    pixels + [0, popcount($6B)], and the model charges the reloaded-mask end of that."""
    drives = _fresh_cursor_drives()
    assert len(drives) >= 12
    for label, px, measured in drives:
        assert px <= measured <= px + pb.CURSOR_RAMP, f"{label}: {measured} ({px}px)"


def test_cursor_charge_is_exact_on_most_drives_and_never_under():
    """The charge is the ramp's upper end, so it never under-prices the dwell a drain gate
    budgets against, and lands exact wherever the mask was reloaded (the common case).
    """
    drives = _fresh_cursor_drives()
    charged = [(px + pb.CURSOR_RAMP if px else 0.0, m) for _lbl, px, m in drives]
    assert all(c >= m for c, m in charged)
    assert sum(c == m for c, m in charged) > len(charged) // 2


def test_charged_toggle_matches_the_measured_pair():
    """``TOGGLE_FRAMES`` prices the OFF+ON pair itself (the trailing notch scroll is
    already charged per notch), so it sits inside the measured pair's range."""
    measured = [row["toggle"] for _, row in _rows() if _panned(row) and row["toggle"]]
    assert measured
    assert min(measured) <= pb.TOGGLE_FRAMES <= max(measured)
