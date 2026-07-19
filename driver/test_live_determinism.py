#!/usr/bin/env python3
"""Live play must be reproducible: same landscape, same trace, same frame counts.

The driver halts the CPU between primitives, so every emulated frame the game
advances should be one it deliberately ran; host-clock leakage (a swallowed
``run_until_pc`` timeout resumes the CPU and halts it wall-clock late) diverges here.
"""

import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from driver.boot import TAP  # noqa: E402
from driver import core  # noqa: E402

SNAPSHOT = os.path.join(ROOT, "renders", core.CODE_ENTRY_SNAP)
_HAVE_DOCKER = os.system("docker info >/dev/null 2>&1") == 0
_SKIP = not (_HAVE_DOCKER and os.path.exists(TAP) and os.path.exists(SNAPSHOT))

# Enemies frozen both sides: a divergence cannot be enemy-phase modelling.
_MAX_ACTIONS = 8
_CLOCK = re.compile(r"\[clock\] (p\d+) (\w+): charged=(\d+) measured=(\d+)")
_TRACE = re.compile(r"^f=\s*\d+\s+(\w+)\s+\((\s*\d+),\s*(\d+)\)\s+E=\s*(\d+)")


def _run(tag):
    """One frozen live run; returns (steps, trace) parsed from its log."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "driver.frozen_run",
            "42",
            "--player",
            "astar",
            "--max-actions",
            str(_MAX_ACTIONS),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env={**os.environ, "NO_RECORD": "1"},
    )
    log = out.stdout + out.stderr
    steps = [
        (m.group(2), int(m.group(3)), int(m.group(4))) for m in _CLOCK.finditer(log)
    ]
    trace = [
        (m.group(1), (int(m.group(2)), int(m.group(3))), int(m.group(4)))
        for m in _TRACE.finditer(log)
    ]
    assert steps, f"{tag}: no [clock] steps in log:\n{log[-2000:]}"
    return steps, trace


@pytest.mark.skipif(_SKIP, reason="needs docker + game tape + code-entry snapshot")
def test_two_live_runs_take_the_same_actions():
    """Identical landscape and player must yield an identical action sequence."""
    (_, trace_a), (_, trace_b) = _run("A"), _run("B")
    assert trace_a == trace_b, (
        "live play diverged between two identical runs (verb, tile, energy):\n"
        f"  A: {trace_a}\n  B: {trace_b}"
    )


@pytest.mark.skipif(_SKIP, reason="needs docker + game tape + code-entry snapshot")
@pytest.mark.xfail(
    strict=False,
    reason="Mostly deterministic since checkpoint installs became halted+toggled "
    "(vice-driver _checkpoint_for): the +-1 frame install lottery is gone. Residual, "
    "~2 runs in 9: same actions but per-step frames differ by tens-to-hundreds of "
    "frames (observed +93 / +17 / -715), the signature of a retry loop taking a "
    "different path (_run_to_scan passes, sentinel_execute range(3)/range(4) retries), "
    "not a frame-stepping race. Non-strict: passes are the common case.",
)
def test_two_live_runs_measure_the_same_frames():
    """Per-step measured frames must match: a differing count is host-clock leakage."""
    (steps_a, _), (steps_b, _) = _run("A"), _run("B")
    assert len(steps_a) == len(
        steps_b
    ), f"step counts differ: {len(steps_a)} vs {len(steps_b)}"
    drift = [
        (i, a, b) for i, (a, b) in enumerate(zip(steps_a, steps_b)) if a[2] != b[2]
    ]
    assert not drift, "measured frames differ between identical runs: " + "; ".join(
        f"step {i} {a[0]}: {a[2]} vs {b[2]} (d={b[2] - a[2]:+d})" for i, a, b in drift
    )
