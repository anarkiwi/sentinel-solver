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

from vice_driver import BinMon, keys
from vice_driver.binmon import TAP_MODE_FIXED
from sentinel.state import State
from sentinel import los
from driver import boot, kbd_aim
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
    0x0000-0xFFFF request is 65536 bytes == 0 mod 2^16 and comes back empty."""
    return bytearray(bm.mem_get(0x0000, 0x7FFF)) + bytearray(bm.mem_get(0x8000, 0xFFFF))


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
    """The player types a 4-digit landscape number; the game reads the last two digits
    as a single BCD byte, whose value IS the internal seed (e.g. "0042" -> byte 0x42 ->
    seed 66). Decimal digits are numerically identical to hex nibbles, so parsing the
    last 2 characters as hex reproduces the BCD-to-binary step."""
    return int(typed_digits[-2:], 16)


def navigate(bm, typed_digits, log=print, snapshot_container=None, snapshot_host=None):
    """Boot under WARP to LANDSCAPE NUMBER, patch the code-checks, type the digits +
    dummy secret code, dismiss the preview, enter play -- the proven live-entry path.

    If snapshot_host exists, RESTORE the machine state saved there (skipping the ~50s tape
    boot) instead of booting; otherwise boot, then SAVE a snapshot at the code-entry screen
    (after the code-check patches, before the landscape digits) so the NEXT run is fast.
    The snapshot is landscape-agnostic -- the digits are always typed after restore."""

    def tap(name, hold=20, settle=0.4):
        robust(
            bm,
            log,
            lambda: bm.keymatrix_tap(
                [keys.lookup(name)], mode=TAP_MODE_FIXED, frames=hold
            ),
        )
        time.sleep(settle)

    def tap_text(t):
        for chord in keys.text_to_chords(t):
            ks = [keys.lookup(n) for n in chord]
            robust(
                bm,
                log,
                lambda ks=ks: bm.keymatrix_tap(ks, mode=TAP_MODE_FIXED, frames=20),
            )
            time.sleep(0.4)

    restored = False
    if snapshot_container and snapshot_host and os.path.exists(snapshot_host):
        try:
            log(f"restoring VICE snapshot {snapshot_host} (skipping tape boot)")
            robust(bm, log, lambda: boot.load_snapshot(bm, snapshot_container))
            try:
                bm.exit()  # resume the CPU after the restore leaves the monitor stopped
            except Exception:
                pass
            time.sleep(1.0)
            restored = True
        except Exception as e:
            log(f"  snapshot restore failed ({e}); falling back to full boot")

    if not restored:
        log("booting + loading (warp)...")
        for _ in range(50):
            time.sleep(1.0)
            robust(bm, log, lambda: bm.mem_get(0x00, 0x00))
        for _ in range(3):
            tap("SPACE", hold=30, settle=1.5)
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
    tap("RETURN", hold=30, settle=3.0)
    tap_text("00000000")
    tap("RETURN", hold=30, settle=8.0)
    time.sleep(3)
    tap("SPACE", hold=25, settle=1.2)  # dismiss the isometric preview
    time.sleep(4)


# ============================================================================
# container / connection (the one home for the asid-vice plumbing the runners share)
# ============================================================================
def bridge_ip(container_id, log=print):
    """The docker bridge IP of a started asid-vice container (host -p publishing is
    not reachable in this environment; the bridge IP is). None on failure."""
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


def free_stale_containers(log=print):
    """Remove any leftover asid-vice containers still holding port 6502."""
    try:
        ids = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "ancestor=asid-vice:latest"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.split()
        if ids:
            subprocess.run(
                ["docker", "rm", "-f", *ids], capture_output=True, timeout=30
            )
            time.sleep(2)
    except Exception as e:
        log(f"  container cleanup warning: {e}")


def connect_binmon(container, log=print):
    """Connect a BinMon to a started container (env BINMON_HOST/PORT override; else
    the container's bridge IP, else host loopback)."""
    time.sleep(2)
    host = os.environ.get("BINMON_HOST") or bridge_ip(container.container_id, log)
    if not host:
        host = "127.0.0.1"
    port = int(os.environ.get("BINMON_PORT", "6502"))
    log(f"  connecting binmon {host}:{port}")
    bm = BinMon(host, port)
    bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
    bm.exit()
    return bm


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
        if not (okh and okv):  # one retry (a wrap/overshoot self-corrects)
            okh = self.kbd.coarse_h(view["h_angle"])
            okv = self.kbd.coarse_v(view["v_angle"])
        if not (okh and okv):
            self.log(f"    aim {tuple(tile)}: coarse pan miss")
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
