#!/usr/bin/env python3
"""Integration test: the enemy-phase model vs the live ROM (ls42).

Frame-locks `enemies.advance_frame` against the running game (`instrument.race`).
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import core, instrument  # noqa: E402
from driver.boot import ROOT, TAP  # noqa: E402
from sentinel import statecmp as sc  # noqa: E402

# enter_landscape types f"{seed:04x}" -> "0042", the landscape-42 board
LS42_SEED = 0x42
FRAMES = 600  # >12 cooldown-gate periods; race() breaks early on the first CORE drift
SNAPSHOT = os.path.join(ROOT, "renders", core.CODE_ENTRY_SNAP)

_HAVE_DOCKER = os.system("docker info >/dev/null 2>&1") == 0
_SKIP = not (_HAVE_DOCKER and os.path.exists(TAP) and os.path.exists(SNAPSHOT))


@pytest.mark.skipif(_SKIP, reason="needs docker + game tape + code-entry snapshot")
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
