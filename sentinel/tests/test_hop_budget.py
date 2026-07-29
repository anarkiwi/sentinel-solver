"""HOP_FRAMES against live-measured ls42 hops, and the whole-step books.

``HOP_FRAMES`` is the exposure budget the REACTIVE player (``sentinel/player.py``) gates a
pedestal build on, so if it is wrong the gate clears hops the body cannot survive. It had
no fixture; ``live_ls42_hops.json`` is the first.  ``PhasePlayer`` does not use it: it
prices each hop from its own actions (``phase_player._hop_span``), because the constant was
measured on ls42's 745-879 f hops and ls335's cost ~1294 f.
"""

import json
import math
import os
import statistics

from sentinel import playerbase as pb

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE = os.path.join(FIXTURES, "live_ls42_hops.json")
UTURN_FIXTURE = os.path.join(FIXTURES, "live_ls335_uturns.json")


def _data():
    with open(FIXTURE) as fh:
        return json.load(fh)


def _uturns():
    """Every live per-u-turn measurement on record: ls42 p1 plus the ls335 win's eight."""
    with open(UTURN_FIXTURE) as fh:
        return _data()["uturns"] + json.load(fh)["uturns"]


def test_hop_frames_brackets_the_measured_hops():
    """A k=1 hop is boulder + robot + transfer. Both recorded hops land within 25% of
    HOP_FRAMES, and it never over-budgets -- an over-budget gate blocks hops that were
    survivable, which is how the live player once fell to taking no actions at all."""
    hops = [h["measured"] for h in _data()["hops"]]
    assert len(hops) >= 2
    for measured in hops:
        assert pb.HOP_FRAMES <= measured, (
            f"HOP_FRAMES {pb.HOP_FRAMES} over-budgets a {measured} f hop: the gate "
            "would block hops the body survives"
        )
        assert (
            abs(pb.HOP_FRAMES - measured) / measured < 0.25
        ), f"HOP_FRAMES {pb.HOP_FRAMES} vs measured {measured}"


def test_hop_step_verbs_sum_to_the_recorded_hop():
    """The hop totals are the sum of their own steps, not separately eyeballed."""
    steps = {s["step"]: s["measured"] for s in _data()["steps"]}
    for hop in _data()["hops"]:
        assert sum(steps[s] for s in hop["steps"]) == hop["measured"]


def test_whole_step_books_are_unbiased_and_bounded():
    """The charged-vs-measured books for the run the hops came from. Guards the two
    defects that ran through it: a swallowed u-turn (p1/p11 were +348 and +1158 before
    the fix) and any systematic per-step bias, which does not cancel -- it shifts when
    every later rotation commits."""
    errs = [s["measured"] - s["charged"] for s in _data()["steps"]]
    rms = math.sqrt(statistics.fmean(e * e for e in errs))
    assert rms < 40.0, f"whole-step rms {rms:.1f} f"
    assert abs(statistics.fmean(errs)) < 20.0, f"bias {statistics.fmean(errs):+.1f} f"
    assert max(abs(e) for e in errs) < 100.0, "a step is mispriced by 100+ frames"


def test_uturn_is_charged_as_an_action_tap_not_a_keystroke():
    """A u-turn goes through tap_action (want-flag $23), so it costs the idle+press
    scans and the action's own consumption -- not one keystroke. The constant is the
    POOLED live mean over both fixtures (n=9); the per-sample spread is 33..180 f, so
    only the mean is pinned, and nothing here claims a per-u-turn bound."""
    ut = [rec["measured"] for rec in _uturns()]
    assert len(ut) >= 9, "the pooled u-turn sample shrank"
    assert abs(pb.UTURN_FRAMES - statistics.fmean(ut)) <= 1.0, statistics.fmean(ut)
    assert pb.UTURN_FRAMES > 10 * pb.TAP_FRAMES  # it is nothing like a bare tap
