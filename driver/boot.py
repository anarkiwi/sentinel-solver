#!/usr/bin/env python3
"""Robust boot of the Sentinel tape in asid-vice. The multi-stage tape load
occasionally JAMs the 6502 under warp (the container exits, closing the binmon socket),
so we retry the whole container launch until a connection survives long enough for the
game code to be resident (wait_for_load signature). Returns a connected
BinMon + the live ViceContainer; the caller closes both."""

import os, sys, time, struct, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from vice_driver import BinMon, DiskMount, ViceContainer

ROOT = os.path.abspath(os.path.join(HERE, ".."))
TAP = os.path.join(ROOT, "sentinel-gold.tap")

# signature bytes of the loaded game in RAM, used to detect that the multi-stage
# tape load has finished (the ROM routines we later call are resident). $35A4 holds
# `A5 0B 85` (play_landscape: LDA player_object / STA) once loaded.
SIG_ADDR = 0x35A4
SIG_BYTES = bytes([0xA5, 0x0B, 0x85])


def wait_for_load(bm, log=print, total=80.0, poll=2.0):
    """Poll RAM until the game is resident (SIG_BYTES present at SIG_ADDR).
    The tape load is multi-stage and its timing under warp varies; polling a
    signature is more reliable than a fixed sleep. Returns True if loaded."""
    deadline = time.time() + total
    while time.time() < deadline:
        try:
            if bytes(bm.mem_get(SIG_ADDR, SIG_ADDR + 2)) == SIG_BYTES:
                log(f"  load complete (sig at ${SIG_ADDR:04x})")
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False


# Reusable boot snapshot: once the tape has loaded to the title screen we save the full
# VICE machine state so later runs can resume it instead of re-loading the ~50s tape.
# Written to the mounted /renders volume (gitignored) so it persists on the host.
BOOT_VSF_NAME = "boot.vsf"
# VICE binary-monitor snapshot opcodes (vice.texi): MON_CMD_DUMP saves a .vsf (full
# CPU+RAM+chip state) to a path inside the emulator process; MON_CMD_UNDUMP restores one.
SNAP_SAVE_OPCODE = 0x41  # body SR|SD|FL|FN
SNAP_LOAD_OPCODE = 0x42  # body FL|FN -> response = restored PC (2 bytes LE)


def save_snapshot(bm, container_path, save_roms=False, save_disks=False, timeout=30.0):
    """Save a VICE machine snapshot via the monitor (MON_CMD_DUMP $41) to
    ``container_path`` -- a path INSIDE the emulator, so point it at the mounted
    /renders volume for it to land on the host. ROMs/disks are omitted (SR=SD=0):
    RAM+CPU+chip state is all that is needed to resume the title screen."""
    fn = container_path.encode()
    body = (
        struct.pack("<BBB", int(bool(save_roms)), int(bool(save_disks)), len(fn)) + fn
    )
    bm.call(SNAP_SAVE_OPCODE, body, timeout=timeout)


def load_snapshot(bm, container_path, timeout=30.0):
    """Restore a VICE machine snapshot via the monitor (MON_CMD_UNDUMP $42) from
    ``container_path`` (a path INSIDE the emulator). Returns the restored PC, or None.
    Used to resume a saved code-entry screen and skip the ~50s tape load."""
    fn = container_path.encode()
    body = struct.pack("<B", len(fn)) + fn
    resp = bm.call(SNAP_LOAD_OPCODE, body, timeout=timeout)
    if resp is not None and len(resp.body) >= 2:
        return struct.unpack("<H", resp.body[:2])[0]
    return None


def save_boot_snapshot_if_missing(bm, renders, log=print):
    """Once the game has loaded to the title screen, save a reusable boot snapshot
    (``renders/boot.vsf``) via the VICE monitor IF one does not already exist. The
    file lives on the mounted /renders volume, which is gitignored (untracked).
    Returns the host path when a snapshot was written, else None (already present, or
    the save failed -- a boot snapshot is an optimisation, never fatal)."""
    host = os.path.join(renders, BOOT_VSF_NAME)
    if os.path.exists(host):
        return None
    log(f"[boot] no {BOOT_VSF_NAME}; saving boot snapshot -> {host}")
    try:
        save_snapshot(bm, "/renders/" + BOOT_VSF_NAME)
        return host
    except Exception as e:
        log(f"[boot] boot snapshot save failed ({type(e).__name__}: {e})")
        return None


def kill_stale():
    """Remove any leftover asid-vice container that may still hold port 6502 (a
    SIGKILLed driver process can orphan a detached --rm container)."""
    try:
        ids = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "ancestor=anarkiwi/asid-vice:latest"],
            capture_output=True,
            text=True,
        ).stdout.split()
        for cid in ids:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        if ids:
            time.sleep(1.5)
    except Exception:
        pass


def bridge_ip(container_id, log=print):
    """The docker bridge IP of a started asid-vice container. Host ``-p`` publishing
    is not reachable in this environment; the bridge IP is. Returns None on failure."""
    try:
        out = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                container_id,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        return out or None
    except Exception as e:  # docker missing / container gone
        log(f"  bridge-ip lookup failed: {e}")
        return None


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
            time.sleep(2)  # let docker assign the container its bridge IP
            # host: env override, else container bridge IP (host -p unreachable here), else loopback
            host = (
                os.environ.get("BINMON_HOST")
                or bridge_ip(container.container_id, log)
                or "127.0.0.1"
            )
            port = int(os.environ.get("BINMON_PORT", "6502"))
            log(f"[boot {attempt}] connecting binmon {host}:{port}")
            bm = BinMon(host, port)
            bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
            bm.exit()
            log(f"[boot {attempt}] connected; waiting for tape load ...")
            if wait_for_load(bm, log, total=80.0, poll=2.0):
                # loaded to the title screen: cache a reusable boot snapshot if absent.
                save_boot_snapshot_if_missing(bm, renders, log)
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
