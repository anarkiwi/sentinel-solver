"""The human wins, replayed action by action through our own executor.

A planner failure is diffuse -- it is spread over a search tree and a dozen gates.  A
replay of a known-good line is not: it names the FIRST action our model cannot
reproduce, in order, against a recorded board.  These floors only ever move up.
"""

from sentinel.tests.human_replay import first_divergence


def test_ls0_replays_exactly():
    """Our executor reproduces the whole ls0 win: the harness itself is sound."""
    assert first_divergence("ls0.json") is None


def test_ls42_and_ls335_replay_at_least_as_far_as_measured():
    """Regression floors; the replay advances by OUR executor's charged pace.

    A human's own pace runs longer than the derived executor charges, so ls335
    repinned 19 -> 10 when settlecost landed; test_human_clock carries the
    exact-span instrument for the human fixtures."""
    assert (first_divergence("ls42.json") or 10**6) >= 14
    assert (first_divergence("ls335.json") or 10**6) >= 10
