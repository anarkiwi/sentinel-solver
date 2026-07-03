#!/usr/bin/env python3
"""Robust boot of the Sentinel tape in asid-vice. The multi-stage tape load
occasionally JAMs the 6502 under warp (the container exits, closing the binmon socket),
so we retry the whole container launch until a connection survives long enough for the
game code to be resident (gen_enter.wait_for_load signature). Returns a connected
BinMon + the live ViceContainer; the caller closes both."""

import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vice_driver import BinMon, DiskMount, ViceContainer
import gen_enter

ROOT = os.path.abspath(os.path.join(HERE, ".."))
TAP = os.path.join(ROOT, "sentinel-gold.tap")


def kill_stale():
    """Remove any leftover asid-vice container that may still hold port 6502 (a
    SIGKILLed driver process can orphan a detached --rm container)."""
    try:
        ids = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "ancestor=asid-vice:latest"],
            capture_output=True,
            text=True,
        ).stdout.split()
        for cid in ids:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        if ids:
            time.sleep(1.5)
    except Exception:
        pass


def boot_loaded(log=print, attempts=4, record_mount=None):
    """Launch the container and wait for the loaded game. Retries on a load JAM.
    record_mount: optional host dir to mount at /renders (for AVI / snapshots).
    Returns (container, bm). Raises RuntimeError if all attempts fail."""
    if not os.path.exists(TAP):
        raise FileNotFoundError(
            f"{TAP} missing: place the game tape image there (not distributed)"
        )
    renders = record_mount or os.path.join(ROOT, "renders")
    last = None
    kill_stale()
    for attempt in range(attempts):
        container = ViceContainer(
            autostart="/work/sentinel.tap",
            mounts=[
                DiskMount(TAP, "/work/sentinel.tap", read_only=True),
                DiskMount(renders, "/renders", read_only=False),
            ],
            warp=True,
            silent=True,
        )
        try:
            container.start()
            # D4: host/port from env (BINMON_HOST/BINMON_PORT); host-loopback default.
            bm = BinMon(
                os.environ.get("BINMON_HOST", "127.0.0.1"),
                int(os.environ.get("BINMON_PORT", "6502")),
            )
            bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
            bm.exit()
            log(f"[boot {attempt}] connected; waiting for tape load ...")
            if gen_enter.wait_for_load(bm, log, total=80.0, poll=2.0):
                return container, bm
            log(f"[boot {attempt}] load signature never appeared; retrying")
        except Exception as e:
            last = e
            log(f"[boot {attempt}] failed: {type(e).__name__}: {e}")
        # tear down before retry
        try:
            container.stop()
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"boot_loaded failed after {attempts} attempts (last={last})")
