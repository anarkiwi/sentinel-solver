#!/usr/bin/env python3
"""Integration tests: the A* plan's enemy-phase premise vs the live ROM (ls42).

One frame-locks `enemies.advance_frame` against the running game (`instrument.race`);
the other audits the live plan step by step (`plan_audit`), asserting no step the plan
gates drain-safe is live-hot. Both xfail on known divergences the search plans over.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import core, instrument, plan_audit  # noqa: E402
from driver.boot import ROOT, TAP  # noqa: E402
from sentinel import statecmp as sc  # noqa: E402

# enter_landscape types f"{seed:04x}" -> "0042", the landscape-42 board
LS42_SEED = 0x42
FRAMES = 600  # >12 cooldown-gate periods; race() breaks early on the first CORE drift
SNAPSHOT = os.path.join(ROOT, "renders", core.CODE_ENTRY_SNAP)

_HAVE_DOCKER = os.system("docker info >/dev/null 2>&1") == 0
_SKIP = not (_HAVE_DOCKER and os.path.exists(TAP) and os.path.exists(SNAPSHOT))


@pytest.mark.skipif(_SKIP, reason="needs docker + game tape + code-entry snapshot")
@pytest.mark.xfail(
    strict=True,
    reason="Bounded sub-frame floor: the live cursor sweeps ~3.43 update_enemies "
    "calls/frame (measured 3-or-4, period-7), and the 3-vs-4 split rides a sub-frame "
    "cycle accumulator absent from the 64KB seed -- so a RAM-seeded sim drifts +-1 "
    "frame on when each enemy's slot is swept (first CORE divergence ~frame 50, "
    "enemy update_cd). Zero needs a cycle-accurate loop model + canonical sub-frame "
    "seed (parked: below the dominant aim-cost error). Strict: flags a true-zero fix.",
)
def test_enemy_sim_frame_locked_to_live_ls42():
    """From an identical seed the plot-independent enemy sim must reproduce the live
    ROM byte-for-byte per frame; fails on the first CORE divergence within FRAMES."""
    drv = core.SentinelDriver.boot(record_mount=instrument.RENDERS)
    try:
        drv.enter_landscape(LS42_SEED)
        result = instrument.race(drv.bm, FRAMES, follow=False, log=lambda *a: None)
    finally:
        drv.close()
    first = result["first"]
    assert sc.CORE not in first, "enemy sim diverged from live at frame {}: {}".format(
        first[sc.CORE][0],
        ", ".join(sc.format_divergence(d, "emu", "sim") for d in first[sc.CORE][1][:4]),
    )


@pytest.mark.skipif(_SKIP, reason="needs docker + game tape + code-entry snapshot")
@pytest.mark.xfail(
    strict=True,
    reason="Plan dwell prediction diverges from live: the enemy rotation phase (rcd) "
    "is mispredicted (identical facings, tens-of-frames rcd drift), so a step the plan "
    "gates drain-safe (pred pbody >= budget) is live-hot -- e.g. boulder (9,8) pbody "
    "pred ~232 vs live ~86 (budget ~158), the stale-trip that dead-ends ls42. Fix the "
    "per-step frame accounting (transfer settle, plotting-during-aim, cadence) so the "
    "plan's dwell windows match reality. Strict: flags the fix.",
)
def test_plan_dwell_prediction_matches_live_ls42():
    """No planned step may be predicted drain-safe (pred body-window >= step budget)
    while live reality is hot (live body-window < budget): the plan must not walk the
    body into a gaze it modelled empty. Runs the live A* and audits every step."""
    records = plan_audit.run_audit("0042", log=lambda *a: None)
    assert records, "plan audit captured no steps"
    bad = [r for r in records if r["pred_pbody"] >= r["budget"] > r["live_pbody"]]
    assert not bad, "plan predicted-safe but live-hot: " + "; ".join(
        f"{r['tag']} {r['verb']}{r['tile']} pbody pred={r['pred_pbody']:.0f} "
        f"live={r['live_pbody']:.0f} budget={r['budget']:.0f}"
        for r in bad
    )
