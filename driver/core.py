#!/usr/bin/env python3
"""The foundation Sentinel game driver.

One class, :class:`SentinelDriver`, is the single entry point for driving the real
game in asid-vice: it boots the tape to the title screen, enters an arbitrary
landscape, and executes the game operations (aim at a tile, create an object,
absorb, transfer, hyperspace) with memory-verified results. It composes the
already-canonical pieces rather than re-deriving them, so there is ONE driver:

  * boot to title      -- :mod:`driver.boot` (ret/retry container + tape load,
                          reusable boot.vsf snapshot);
  * enter landscape N  -- :func:`navigate` (drive the real title menu by keyboard;
                          restores a code-entry snapshot to skip the tape boot);
  * keyboard aim/fire  -- :class:`driver.kbd_aim.KbdDriver` (the checkpoint-driven,
                          U-turn-aware sights driver);
  * state reads        -- :mod:`driver.sentinel_state` (live BinMon -> GameState).

Everything an operation needs to *decide* and *verify* -- the object-array reads,
the native aim snap, the container plumbing -- lives here, so the older standalone
climb/record experiments can drop their private copies and drive through this.
"""

import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from vice_driver import BinMon, DiskMount, ViceContainer, keys
from vice_driver.binmon import TAP_MODE_FIXED
from sentinel.state import State
from sentinel import los
from driver import boot, clock, kbd_aim
from driver import sentinel_state as gs

# ---- live RAM addresses (the ROM's object-array + sights layout) -------------
A_SLOT = 0x000B  # player_object
A_X = 0x0900  # objects_x + slot
A_Y = 0x0980  # objects_y + slot
A_TYPE = 0x0A40  # objects_type + slot
A_FLAGS = 0x0100  # objects_flags + slot ($80 empty; $40-$7F stacked-on-object-N)
A_H = 0x09C0  # objects_h_angle + slot
A_V = 0x0140  # objects_v_angle + slot
A_ZH = 0x0940  # objects_z_height + slot
A_ENERGY = 0x0C0A  # player_energy (6-bit)
A_LANDSCAPE_DONE = 0x0CDE  # bit6 set == landscape complete ($2198)
A_CURSOR_X = 0x0CC6  # sights cursor column
A_CURSOR_Y = 0x0CC7  # sights cursor row
A_ACTION_LATCH = 0x0CE4  # bit7 set mid-pan / queued-wrap (reject transient probes)

CX_CENTRE, CY_CENTRE = kbd_aim.CX_CENTRE, kbd_aim.CY_CENTRE

# object types + the create key each one is built with.
T_ROBOT, T_TREE, T_BOULDER = 0, 2, 3
_CREATE_KEY = {T_ROBOT: "R", T_BOULDER: "B", T_TREE: "T"}
K_TRANSFER, K_ABSORB, K_HYPERSPACE = "Q", "A", "H"


# ============================================================================
# monitor resilience + full-image reads (shared by the entry nav and plan runner)
# ============================================================================
def reconnect(bm, log=print):
    """Re-open a dropped monitor socket (a warp/AVI stall can close it mid-op)."""
    try:
        bm.close()
    except Exception:
        pass
    bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
    try:
        bm.exit()
    except Exception:
        pass
    log("   (reconnected monitor socket)")


def robust(bm, log, fn, tries=4):
    """Run a monitor op, reconnecting + retrying on a dropped socket."""
    from vice_driver.binmon import BinmonError

    for _ in range(tries):
        try:
            return fn()
        except (
            BinmonError,
            BrokenPipeError,
            ConnectionError,
            OSError,
            TimeoutError,
        ) as e:
            log(f"   monitor op dropped ({type(e).__name__}); reconnecting")
            reconnect(bm, log)
    return fn()


def live_image(bm):
    """The full 64 KB live memory image the simulator (``State``/``Game``) is defined
    over. It reads ROM tables such as ROTATION_SPEED_TABLE ($9D37) during enemy stepping
    (threat.ticks_until_seen -> enemies.step), so a 4 KB slice throws IndexError.

    Read in two 32 KB halves: mem_get's response length is a u16, so a single
    0x0000-0xFFFF request is 65536 bytes == 0 mod 2^16 and comes back empty.  Both
    halves are read HALTED: under auto_resume each half would resume the CPU, tearing
    the image across a host-timing-dependent number of frames."""
    with bm.halted():
        return bytearray(bm.mem_get(0x0000, 0x7FFF)) + bytearray(
            bm.mem_get(0x8000, 0xFFFF)
        )


# ============================================================================
# landscape entry -- drive the real title menu by keyboard (the proven live path)
#
# The ROM's "SECRET ENTRY CODE?" gate ($14DC-$14F2) computes the jump to play from the
# code-validation result, so it can't simply be bypassed; instead patch the three
# code-check sites to accept any code and navigate the menu as a player would. A
# code-entry-screen snapshot (renders/vice_code_entry.vsf) is restored when present to
# skip the ~50s tape load.
# ============================================================================
CODE_ENTRY_SNAP = "vice_code_entry.vsf"  # under the mounted /renders volume

# secret-code-check patches (accept any code)
CODE_PATCHES = [
    (0x14DF, bytes([0xA9, 0x1E])),
    (0x2565, bytes([0xEA, 0xEA])),
    (0x2570, bytes([0xEA, 0xEA])),
]


def landscape_from_digits(typed_digits):
    """The player types a 4-digit landscape number; the game stores it as a packed-BCD
    word and seeds the PRNG with both bytes (seed_prnd_from_landscape_number $33ED:
    $0C7B <- seed & $FF, $0C7C <- seed >> 8, :meth:`sentinel.prng.Prnd.seeded`).
    Decimal digits are numerically identical to hex nibbles, so the seed is the whole
    code parsed as hex: "0042" -> 0x0042, "0335" -> 0x0335. Inverse of the
    ``f"{landscape:04x}"`` in :meth:`SentinelDriver.enter_landscape`."""
    return int(typed_digits, 16)


def navigate(bm, typed_digits, log=print, snapshot_container=None, snapshot_host=None):
    """Boot under WARP to LANDSCAPE NUMBER, patch the code-checks, type the digits +
    dummy secret code, dismiss the preview, enter play -- the proven live-entry path.

    If snapshot_host exists, RESTORE the machine state saved there (skipping the ~50s tape
    boot) instead of booting; otherwise boot, then SAVE a snapshot at the code-entry screen
    (after the code-check patches, before the landscape digits) so the NEXT run is fast.
    The snapshot is landscape-agnostic -- the digits are always typed after restore."""

    def tap(name, hold=20, settle=40):
        """Tap a chord, then step hold+settle EMULATED frames: VICE releases a
        TAP_MODE_FIXED chord after ``hold`` frames and the menu consumes it within
        ``settle``, so what the game sees cannot depend on warp or the host clock."""
        robust(
            bm,
            log,
            lambda: bm.keymatrix_tap(
                [keys.lookup(name)], mode=TAP_MODE_FIXED, frames=hold
            ),
        )
        clock.run_frames(bm, hold + settle)

    def tap_text(t):
        for chord in keys.text_to_chords(t):
            ks = [keys.lookup(n) for n in chord]
            robust(
                bm,
                log,
                lambda ks=ks: bm.keymatrix_tap(ks, mode=TAP_MODE_FIXED, frames=20),
            )
            clock.run_frames(bm, 45)

    restored = False
    if snapshot_container and snapshot_host and os.path.exists(snapshot_host):
        try:
            log(f"restoring VICE snapshot {snapshot_host} (skipping tape boot)")
            robust(bm, log, lambda: boot.load_snapshot(bm, snapshot_container))
            try:
                bm.exit()  # resume the CPU after the restore leaves the monitor stopped
            except Exception:
                pass
            clock.run_frames(bm, 50)
            restored = True
        except Exception as e:
            log(f"  snapshot restore failed ({e}); falling back to full boot")

    if not restored:
        log("booting + loading (warp)...")
        if not boot.wait_for_load(bm, log):
            log("  tape-load signature never appeared; menu may not respond")
        for _ in range(3):
            tap("SPACE", hold=30, settle=250)  # title -> code-entry screen
        log("patching secret-code checks ($14DF/$2565/$2570)")
        for addr, data in CODE_PATCHES:
            robust(bm, log, lambda a=addr, d=data: bm.mem_set(a, d))
        if snapshot_container and snapshot_host:
            log(f"saving VICE snapshot -> {snapshot_host} (reuse to skip next boot)")
            try:
                robust(bm, log, lambda: boot.save_snapshot(bm, snapshot_container))
            except Exception as e:
                log(f"  snapshot save failed ({e}); continuing without it")
    log(f"typing landscape digits {typed_digits!r}")
    tap_text(typed_digits)
    tap("RETURN", hold=30, settle=150)
    tap_text("00000000")
    tap("RETURN", hold=30, settle=150)  # accepted code -> generate landscape + preview
    _enter_play(bm, tap, log)


def _generated(bm):
    """Landscape generation has finished: the ROM has installed the player object with
    its starting energy (both $0B and $0C0A read 0 from the code-entry screen on)."""
    return bool(bm.mem_get(A_ENERGY, A_ENERGY)[0] & 0x3F)


def _in_play(bm):
    """The interactive play loop is running: the busy-plotting gate ($0CE4 bit7), held
    SET across the code-entry menu, generation and the isometric preview, is released
    the first time the foreground loop opens for input."""
    return not bm.mem_get(A_ACTION_LATCH, A_ACTION_LATCH)[0] & 0x80


def _enter_play(bm, tap, log, chunk=25, gen_chunks=160, play_chunks=40, taps=6):
    """Land in the play loop after the secret-code RETURN, one leg per REAL predicate:
    generation installs the player object, then SPACE dismisses the isometric preview
    and the busy-plotting gate opens. Both legs are polled in EMULATED frames, so their
    length cannot depend on warp (measured: ~600 frames, then ~250 after SPACE)."""
    for _ in range(gen_chunks):
        if _generated(bm):
            break
        clock.run_frames(bm, chunk)
    else:
        raise RuntimeError("landscape entry: generation never installed a player")
    log(f"  landscape generated (player slot {bm.mem_get(A_SLOT, A_SLOT)[0]})")
    for _ in range(taps):
        tap("SPACE", hold=25, settle=60)  # dismiss the isometric preview
        for _ in range(play_chunks):
            if _in_play(bm):
                log("  in play loop ($0CE4 bit7 released)")
                return
            clock.run_frames(bm, chunk)
    raise RuntimeError("landscape entry: play never started ($0CE4 bit7 held set)")


# container / connection: the one home for the asid-vice plumbing the runners share.
bridge_ip = boot.bridge_ip  # single implementation lives in driver.boot


def free_stale_containers(log=print):
    """Remove leftover asid-vice containers THIS process orphaned (see
    ``boot.stale_filter``; a blanket sweep would kill a concurrent run's container)."""
    try:
        ids = subprocess.run(
            ["docker", "ps", "-aq", *boot.stale_filter()],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.split()
        if ids:
            subprocess.run(
                ["docker", "rm", "-f", *ids], capture_output=True, timeout=30
            )
            time.sleep(2)  # sleep-ok: docker rm teardown, outside the emulated machine
    except Exception as e:
        log(f"  container cleanup warning: {e}")


def connect_binmon(container, log=print):
    """Connect a BinMon to a started container (env BINMON_HOST/PORT override; else
    the container's bridge IP, else host loopback)."""
    time.sleep(2)  # sleep-ok: docker bridge IP assignment, no PC exists
    host = os.environ.get("BINMON_HOST") or bridge_ip(container.container_id, log)
    if not host:
        host = "127.0.0.1"
    port = int(os.environ.get("BINMON_PORT", "6502"))
    log(f"  connecting binmon {host}:{port}")
    bm = BinMon(host, port)
    bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
    bm.exit()
    return bm


class GameSession:
    """A booted, in-play asid-vice game handed to the plan runner: the live BinMon,
    the entered landscape, the entry-match check, the record start time and the AVI
    host path. Constructed by :func:`boot_and_play`; the runner builds its own
    ``Executor``/``KbdDriver`` on ``bm``."""

    def __init__(self, bm, landscape, t_start, entry_match, state0, video_host):
        self.bm = bm
        self.landscape = landscape
        self.t_start = t_start
        self.entry_match = entry_match
        self.state0 = state0
        self.video_host = video_host


def boot_and_play(tap, renders_host, typed_digits, video_name, log, play_fn, result):
    """Boot asid-vice into ``typed_digits``' landscape in play, optionally record an
    AVI, and hand a :class:`GameSession` to ``play_fn`` -- the emulator-side glue the
    live runner should NOT own. Handles container lifecycle, boot retries, binmon
    connect (bridge IP), title-menu navigation with the cached code-entry snapshot,
    the in-play check, and video start/stop. ``play_fn(session)`` runs the actual plan
    loop; ``result`` accumulates its output (and gates the retry: a mid-run drop after
    real actions is returned, not retried). Returns ``result``."""
    if not os.path.exists(tap):
        raise FileNotFoundError(
            f"{tap} missing: place the game tape image there (not distributed)"
        )
    landscape = landscape_from_digits(typed_digits)
    log(f"LIVE replanning mode: landscape {landscape}")
    os.makedirs(renders_host, exist_ok=True)
    video_host = os.path.join(renders_host, video_name)
    result.setdefault("video", video_host)
    if os.path.exists(video_host):
        try:
            os.remove(video_host)
        except OSError:
            pass

    boot_tries = 8
    for boot_try in range(boot_tries):
        free_stale_containers(log)
        container = ViceContainer(
            autostart="/work/sentinel.tap",
            mounts=[
                DiskMount(tap, "/work/sentinel.tap", read_only=True),
                DiskMount(renders_host, "/renders", read_only=False),
            ],
            warp=True,
            silent=True,
            binmon_port=boot.HOST_BINMON_PORT,
        )
        t_start = time.time()
        try:
            with container:
                # Host -p publishing is broken here (127.0.0.1:6502 unreachable); connect
                # via the started container's docker bridge IP. BINMON_HOST env overrides.
                time.sleep(2)  # sleep-ok: docker bridge IP assignment, no PC exists
                bm_host = os.environ.get("BINMON_HOST") or bridge_ip(
                    container.container_id, log
                )
                if not bm_host:
                    log(
                        "  could not determine container bridge IP; falling back to 127.0.0.1"
                    )
                    bm_host = "127.0.0.1"
                bm_port = int(os.environ.get("BINMON_PORT", "6502"))
                log(f"  connecting to binmon at {bm_host}:{bm_port} (bridge IP)")
                bm = BinMon(bm_host, bm_port)
                try:
                    bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
                except (ConnectionError, OSError, TimeoutError) as e:
                    log(
                        f"  connect to {bm_host}:{bm_port} failed ({type(e).__name__}: {e})"
                        f" -- is the port reachable from this network namespace? "
                        f"set BINMON_HOST to the container bridge IP."
                    )
                    raise
                bm.exit()

                # navigate auto-caches the code-entry snapshot on the mounted /renders
                # volume: it restores renders/vice_code_entry.vsf when present (skipping
                # the ~50s tape boot) and saves it after the code-check patches when
                # absent -- landscape-agnostic (the digits are typed after restore).
                navigate(
                    bm,
                    typed_digits,
                    log,
                    snapshot_container="/renders/" + CODE_ENTRY_SNAP,
                    snapshot_host=os.path.join(renders_host, CODE_ENTRY_SNAP),
                )
                bm.auto_resume = False  # in play: reads must not EXIT, or observing the machine advances it by host-timed frames; the world moves only in deliberate run windows
                st = gs.read_game_state(gs.ViceSource(bm))
                if st.player is None:
                    log(f"boot try {boot_try}: not in play (no player); restart")
                    continue
                entry_match = gs.verify_entry(bm, landscape, log)

                record = os.environ.get("NO_RECORD") != "1"
                if record:
                    log(f"-- starting AVI recording -> {video_host} --")
                    try:
                        bm.video_record(f"/renders/{video_name}")
                    except Exception as e:
                        log(f"  video_record failed: {e}")
                    time.sleep(1.0)  # sleep-ok: VICE AVI encoder start, not the CPU
                else:
                    log("-- NO_RECORD=1: skipping AVI (warp stays on) --")

                session = GameSession(
                    bm, landscape, t_start, entry_match, st, video_host
                )
                # FINALIZE the AVI even when the plan loop raises (an aim-exact crash,
                # a mid-run divergence): the recording of the ATTEMPT up to the failure
                # is the deliverable, so stop+flush it in a finally before the exception
                # propagates -- otherwise the container teardown kills VICE mid-write and
                # the AVI has no frame index (frames=0, "not RIFF/AVI").
                try:
                    play_fn(session)
                    time.sleep(1.5)  # sleep-ok: VICE AVI muxer drain, not the CPU
                finally:
                    log("-- stopping AVI recording (finalize) --")
                    try:
                        if record:
                            bm.video_stop()
                        time.sleep(1.5)  # sleep-ok: VICE AVI index flush, not the CPU
                    except Exception as e:
                        log(f"  video_stop failed: {e}")
                    result["wall_seconds"] = round(time.time() - t_start, 1)
                    bm.close()
            return result
        except Exception as e:
            import traceback

            log(f"boot try {boot_try}: container/boot error: {type(e).__name__}: {e}")
            if boot_try == 0:
                traceback.print_exc()
            if result.get("actions"):
                result["divergence"] = result.get("divergence") or f"mid-run drop: {e}"
                return result
            time.sleep(2)  # sleep-ok: container relaunch backoff, no machine to poll
    result["divergence"] = f"could not boot into play after {boot_tries} tries"
    return result


def validate_avi(path):
    """Sanity-check a recorded AVI: RIFF/AVI header + at least one video frame in the
    'movi' list. Returns (ok, size_bytes, n_frames, message)."""
    import struct

    if not os.path.exists(path):
        return False, 0, 0, "missing"
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read()
    if data[0:4] != b"RIFF" or data[8:12] != b"AVI ":
        return False, size, 0, "not RIFF/AVI"
    movi = data.find(b"movi")
    if movi == -1:
        return False, size, 0, "no movi list"
    n, p = 0, movi + 4
    while p + 8 <= len(data):
        cid = data[p : p + 4]
        sz = struct.unpack("<I", data[p + 4 : p + 8])[0]
        if cid == b"idx1":
            break
        if cid[2:4] in (b"dc", b"db"):
            n += 1
        p += 8 + sz + (sz & 1)
    return n > 0, size, n, "ok" if n > 0 else "no frames"


# ============================================================================
# object-array reads (pure; work off any object with a `mem_get(a, b) -> bytes`)
# ============================================================================
def _obj_arrays(bm):
    """Bulk-read the 64-slot object arrays (flags, x, y, type) in 4 socket calls."""
    return (
        bm.mem_get(A_FLAGS, A_FLAGS + 63),
        bm.mem_get(A_X, A_X + 63),
        bm.mem_get(A_Y, A_Y + 63),
        bm.mem_get(A_TYPE, A_TYPE + 63),
    )


def slots_at_tile(bm, x, y):
    """(slot, type) of every live object on tile (x, y)."""
    fl, xs, ys, ty = _obj_arrays(bm)
    return [
        (i, ty[i])
        for i in range(64)
        if not (fl[i] & 0x80) and xs[i] == x and ys[i] == y
    ]


def object_in_tile(bm, x, y):
    """(slot, type) of the TOPMOST object on tile (x, y) -- the stack head no other
    live object sits on ($40-$7F flags mean "stacked on object N") -- or None."""
    fl, xs, ys, ty = _obj_arrays(bm)
    here = [i for i in range(64) if not (fl[i] & 0x80) and xs[i] == x and ys[i] == y]
    if not here:
        return None
    supporting = {fl[i] & 0x3F for i in range(64) if 0x40 <= fl[i] <= 0x7F}
    top = [i for i in here if i not in supporting]
    slot = top[0] if top else max(here)
    return slot, ty[slot]


def energy(bm):
    """Player energy (6-bit)."""
    return bm.mem_get(A_ENERGY, A_ENERGY)[0] & 0x3F


def player_tile(bm):
    """The player object's (x, y) tile."""
    ps = bm.mem_get(A_SLOT, A_SLOT)[0]
    return bm.mem_get(A_X + ps, A_X + ps)[0], bm.mem_get(A_Y + ps, A_Y + ps)[0]


def read_state(bm):
    """A full :class:`GameState` from live memory."""
    return gs.read_game_state(gs.ViceSource(bm))


# ============================================================================
# native aim snap (choose a keyboard view that lands on a tile with LOS, from a
# RAM snapshot -- no live-CPU LOS probe, which would wedge the incremental plotter)
# ============================================================================
def _native(bm):
    m4k = bytes(bm.mem_get(0x0000, 0x0FFF))
    ps = m4k[A_SLOT]
    return State.from_mem(m4k), ps, m4k[A_ZH + ps], m4k[A_H + ps], m4k[A_V + ps]


def _grids(h0, v0):
    hgrid = [((h0 % 8) + 8 * k) & 0xFF for k in range(32)]
    band = list(range(0xCD, 0x100)) + list(range(0x00, 0x36))
    vgrid = [v for v in band if v % 4 == (v0 % 4)]
    return hgrid, vgrid


def snap_view(bm, tile, want_centre=False):
    """A keyboard view (h, v, cursor) whose native ray lands on `tile` with LOS: the
    centred-cursor h*v lattice first, then a small bounded cursor window as the only
    fallback (bounded so an unreachable tile costs seconds, never minutes). None if
    no reachable view sees the tile. Uses sentinel.los, bit-exact vs the ROM aim."""
    st, ps, eye, h0, v0 = _native(bm)
    hgrid, vgrid = _grids(h0, v0)
    cxs = [CX_CENTRE, 71, 89, 62, 98]
    cys = [CY_CENTRE, 86, 104, 77, 113]

    def sweep(cx_list, cy_list):
        best = None
        for cx in cx_list:
            for cy in cy_list:
                for h in hgrid:
                    for v in vgrid:
                        rx, ry, hit, centre = los.aim_target(
                            st,
                            h,
                            v,
                            cx,
                            cy,
                            ps,
                            eye_z=eye,
                            max_steps=640,
                            return_centre=True,
                        )
                        if (rx, ry) != tuple(tile) or not hit:
                            continue
                        if want_centre and centre >= 0x40:
                            continue
                        view = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
                        if best is None or centre < best[0]:
                            best = (centre, view)
                        if centre < 0x20:
                            return best
        return best

    hit = sweep([CX_CENTRE], [CY_CENTRE]) or sweep(cxs, cys)
    return hit[1] if hit else None


def _probe_once(bm):
    """One live sights-ray read: (tile hit, LOS, centre) + the (latch, h, v, cursor)
    signature used to reject a transient mid-pan snapshot."""
    m = live_image(bm)
    ps = m[A_SLOT]
    st = State.from_mem(bytes(m))
    rx, ry, hit, centre = los.aim_target(
        st,
        m[A_H + ps],
        m[A_V + ps],
        m[A_CURSOR_X],
        m[A_CURSOR_Y],
        ps,
        eye_z=m[A_ZH + ps],
        max_steps=4000,
        return_centre=True,
    )
    sig = (
        m[A_ACTION_LATCH] & 0x80,
        m[A_H + ps],
        m[A_V + ps],
        m[A_CURSOR_X],
        m[A_CURSOR_Y],
    )
    return (rx, ry, hit, centre), sig


def probe_tile(bm):
    """Where the live sights ray lands now (sentinel.los on a cheap RAM snapshot).
    Hardened: only accept a snapshot when $0CE4 bit7 is clear AND h/v/cursor are
    identical across two consecutive reads (reject transient mid-pan / queued-wrap
    state), else advance the machine two frames and retry. Returns (rx, ry, los,
    centre). Frame-stepping is load-bearing, not a nicety: the live player leaves the
    CPU halted, so a host sleep would never let the plot finish and clear $0CE4."""
    res, prev = _probe_once(bm)
    for _ in range(8):
        if prev[0] == 0:
            res2, sig2 = _probe_once(bm)
            if sig2 == prev:
                return res2
            res, prev = res2, sig2
        else:
            clock.run_frames(bm, 2)
            res, prev = _probe_once(bm)
    return res


# ============================================================================
# the foundation driver
# ============================================================================
class SentinelDriver:
    """Boot the game, enter a landscape, and run memory-verified operations over one
    connected VICE monitor. Construct via :meth:`boot` (launch + connect + tape load)
    or directly from an already-connected ``bm``."""

    def __init__(self, bm, container=None, log=print, renders=None):
        self.bm = bm
        self.container = container
        self.log = log
        self.renders = renders or os.path.join(boot.ROOT, "renders")
        self.kbd = kbd_aim.KbdDriver(bm, log)

    @classmethod
    def boot(cls, log=print, attempts=4, record_mount=None):
        """Launch asid-vice and boot the tape to the title screen (saving a reusable
        boot snapshot if none exists). Returns a ready driver; call :meth:`close`."""
        container, bm = boot.boot_loaded(
            log=log, attempts=attempts, record_mount=record_mount
        )
        return cls(bm, container=container, log=log, renders=record_mount)

    def enter_landscape(self, landscape):
        """Enter `landscape` (internal seed) by driving the real title menu (the proven
        live path); leaves the CPU in the interactive play loop. A code-entry snapshot
        under the renders dir skips the ~50s tape boot."""
        snap_host = os.path.join(self.renders, CODE_ENTRY_SNAP)
        navigate(
            self.bm,
            f"{landscape:04x}",
            log=self.log,
            snapshot_container="/renders/" + CODE_ENTRY_SNAP,
            snapshot_host=snap_host,
        )

    # ---- reads -------------------------------------------------------------
    def state(self):
        return read_state(self.bm)

    def energy(self):
        return energy(self.bm)

    def player_tile(self):
        return player_tile(self.bm)

    def object_in_tile(self, x, y):
        return object_in_tile(self.bm, x, y)

    def slots_at_tile(self, x, y):
        return slots_at_tile(self.bm, x, y)

    def won(self):
        """Whether the landscape is complete ($0CDE bit6)."""
        return bool(self.bm.mem_get(A_LANDSCAPE_DONE, A_LANDSCAPE_DONE)[0] & 0x40)

    # ---- aim ---------------------------------------------------------------
    def aim(self, tile, want_centre=False):
        """Drive the sights onto `tile` (coarse rotate sights-off, then fine cursor
        sights-on, via the canonical KbdDriver). Returns True on a confirmed landing.
        The view is snapped with the CPU HALTED so the enemies do not advance while
        we think."""
        with self.bm.halted():
            view = snap_view(self.bm, tile, want_centre)
        self.bm.exit()
        if view is None:
            self.log(f"    aim {tuple(tile)}: no keyboard view")
            return False
        if not self.kbd.sights_set(False):
            return False
        okh = self.kbd.coarse_h(view["h_angle"])
        okv = self.kbd.coarse_v(view["v_angle"])
        if okh != "ok" or okv != "ok":  # one retry (a wrap/overshoot self-corrects)
            okh = self.kbd.coarse_h(view["h_angle"])
            okv = self.kbd.coarse_v(view["v_angle"])
        if okh != "ok" or okv != "ok":
            self.log(f"    aim {tuple(tile)}: coarse pan miss ({okh}/{okv})")
            return False
        if not self.kbd.sights_set(True):
            return False
        return bool(self.kbd.fine_cursor(*view["cursor"]))

    # ---- operations (aim -> fire -> verify from memory; self-healing) ------
    def create(self, otype, tile, tries=3):
        """Build an object of `otype` on `tile`. Returns the new slot, or None."""
        key = _CREATE_KEY[otype]

        def typed():
            return {s for s, t in self.slots_at_tile(*tile) if t == otype}

        before = typed()
        if before:  # a prior attempt already placed one
            return max(before)
        for attempt in range(tries):
            if not self.aim(tile, want_centre=(attempt == 0)):
                continue
            e0 = self.energy()
            self.kbd.tap_action(key)
            new = typed() - before
            if new:
                slot = max(new)
                self.log(
                    f"    created type{otype} @ {tuple(tile)} slot{slot} "
                    f"energy {e0}->{self.energy()}"
                )
                return slot
        return None

    def absorb(self, tile, tries=3):
        """Absorb the topmost object on `tile`. Returns True (incl. nothing there)."""
        occ = self.object_in_tile(*tile)
        if occ is None:
            return True
        for attempt in range(tries):
            if not self.aim(tile, want_centre=(attempt == 0)):
                continue
            e0 = self.energy()
            self.kbd.tap_action(K_ABSORB)
            occ2 = self.object_in_tile(*tile)
            if occ2 is None or occ2[0] != occ[0]:
                self.log(
                    f"    absorbed slot{occ[0]} @ {tuple(tile)} "
                    f"energy {e0}->{self.energy()}"
                )
                return True
        return False

    def transfer(self, tile, tries=3):
        """Transfer the point of view into the robot on `tile`. Returns True."""
        for attempt in range(tries):
            if not self.aim(tile, want_centre=(attempt == 0)):
                continue
            self.kbd.tap_action(K_TRANSFER)
            if self.player_tile() == tuple(tile):
                self.log(f"    transferred to {tuple(tile)}")
                return True
        return False

    def hyperspace(self):
        """Fire hyperspace (panic escape / win when standing on the platform)."""
        return self.kbd.tap_action(K_HYPERSPACE)

    def close(self):
        """Stop the container if this driver owns one."""
        if self.container is not None:
            try:
                self.container.stop()
            except Exception:
                pass


def main(argv=None):
    """Smoke demo: boot, enter a landscape (argv[1], default 0), print the live
    state, then stop. Needs Docker + the tape image."""
    argv = sys.argv if argv is None else argv
    landscape = int(argv[1]) if len(argv) > 1 else 0
    drv = SentinelDriver.boot()
    try:
        drv.enter_landscape(landscape)
        px, py = drv.player_tile()
        print(f"landscape {landscape}: player ({px},{py}) energy {drv.energy()}")
        print(gs.dump(drv.state()))
    finally:
        drv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
