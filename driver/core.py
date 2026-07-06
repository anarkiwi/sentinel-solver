#!/usr/bin/env python3
"""The foundation Sentinel game driver.

One class, :class:`SentinelDriver`, is the single entry point for driving the real
game in asid-vice: it boots the tape to the title screen, enters an arbitrary
landscape, and executes the game operations (aim at a tile, create an object,
absorb, transfer, hyperspace) with memory-verified results. It composes the
already-canonical pieces rather than re-deriving them, so there is ONE driver:

  * boot to title      -- :mod:`driver.boot` (ret/retry container + tape load,
                          reusable boot.vsf snapshot);
  * enter landscape N  -- :func:`generate_and_enter` (the ROM's own generate + enter,
                          bypassing the secret-code gate);
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

from vice_driver import BinMon
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
# ROM-native landscape entry (bypassing the obfuscated "SECRET ENTRY CODE?" gate)
#
# The code-check at $14DC-$14F2 computes the jump to play_landscape ($35A4) from the
# validation result + objects_flags, so naive patching crashes. Instead we mirror
# play_setup ($1A97) exactly, minus the preview/title plotting and minus the code gate:
#   $1149 reset_game_state
#   $33ED seed_prnd_from_landscape_number (X=lo, Y=hi)  -> stores $0CFD/$0CFE/$0C52
#   $2ACC generate_landscape, STOP at $2B21 (terrain-build end; the render tail desyncs
#         the prnd and changes the tree count)
#   $1420 set_palette_and_initialise_enemies (Sentinel + sentries + landscape palette)
#   $1450 initialise_player_and_trees -- called with $0C71 bit7 CLEAR so its $14AF
#         `BIT $0C71; BPL` takes the PREVIEW path (build validation table, clean RTS)
#         instead of the obfuscated leave-to-play jump. Byte-for-byte the same
#         object/prnd path the simulator uses, so the board matches.
#   set $141F = $7F (viewpoint_perspective; REQUIRED for in-play LOS geometry $13FF)
#   set $0C71 bit7 (play_game_after_generation -> in-play semantics)
#   ensure $0CDE = 0 (player_has_hyperspaced clear; play_landscape entry checks it)
#   JMP $35A4 (play_landscape) -- the real interactive loop.
# Each ROM routine is invoked JSR-style via a tiny `JSR addr ; JMP self` stub planted in
# free RAM; generate's early stop ($2B21) uses run_until_pc on $2B21 directly.
# ============================================================================
R_RESET = 0x1149  # reset_game_state
R_SEED = 0x33ED  # seed_prnd_from_landscape_number (X=lo, Y=hi)
R_GENERATE = 0x2ACC  # generate_landscape
GENERATE_END = 0x2B21  # terrain-build end (stop before preview-render tail)
R_INIT_ENEMIES = (
    0x1420  # set_palette_and_initialise_enemies (Sentinel/sentries+palette)
)
R_INIT_PLAYER = 0x1450  # initialise_player_and_trees
PLAY_LANDSCAPE = 0x35A4  # play_landscape (real first-person loop)

A_PLAY_FLAG = 0x0C71  # play_game_after_generation (bit7)
A_VIEWPOINT = 0x141F  # viewpoint_perspective: 0=preview, $7F=in-play ($13FF)
A_HYPERSPACED = 0x0CDE  # player_has_hyperspaced (bit7) / landscape-complete (bit6)

# scratch RAM for the JSR stub. The tape buffer / free zp-area $0334 is unused at the
# title/code screen; clear of the LOS stub ($02A0) so nothing live is clobbered.
STUB = 0x0334

# While the Sentinel waits for input at the "LANDSCAPE NUMBER?" / code screen it spins
# in its keyboard-matrix scanner ($8CF9, scan_keyboard_matrix) -- a reliable place to
# STOP the CPU before injecting (live PC samples sit in $8CF9-$8D68).
TITLE_HALT = 0x8CF9


def _halt(bm, addr=TITLE_HALT, timeout=8.0):
    """Bring the CPU to a known HALTED state so a subsequent registers_set(PC) sticks.
    `bm.halted()` only disables auto-resume -- it does NOT stop a running CPU, so a PC
    we set would be immediately overwritten by the live game loop. Stop the CPU with an
    EXEC checkpoint at `addr` (an address the current loop executes every iteration)."""
    bm.run_until_pc(addr, timeout=timeout)


def _stub_call(bm, addr, a=0, x=0, y=0, timeout=8.0, stop_pc=None):
    """JSR-call ROM `addr` (A/X/Y set) and run until it returns. Precondition: the CPU
    is ALREADY HALTED (the caller halts once via _halt; after each call the CPU sits at
    the `JMP self` guard, still halted). Plant `JSR addr ; JMP self` at STUB, set PC=STUB
    and the A/X/Y regs (sticks because the CPU is halted -- done with auto-resume off so
    no live frame runs between the pokes), then run_until_pc the JMP-self (the routine's
    RTS lands on it). `stop_pc` halts generate at $2B21 before its preview-render tail.
    """
    jsr = bytes([0x20, addr & 0xFF, (addr >> 8) & 0xFF])
    jmp_self = STUB + len(jsr)
    code = jsr + bytes([0x4C, jmp_self & 0xFF, (jmp_self >> 8) & 0xFF])
    with bm.halted():
        bm.mem_set(STUB, code)
        # SP near top of stack so the JSR has room; regs A/X/Y; FLAGS=$20 (clear D/I).
        bm.registers_set(
            {0: a & 0xFF, 1: x & 0xFF, 2: y & 0xFF, 3: STUB, 4: 0xFD, 5: 0x20}
        )
    target = stop_pc if stop_pc is not None else jmp_self
    bm.run_until_pc(target, timeout=timeout)


def generate_and_enter(bm, landscape, log=print, settle=1.0):
    """Run the play_setup mirror in live VICE for `landscape`, then JMP into
    play_landscape ($35A4). Returns when the interactive loop has been entered.
    Caller is responsible for having booted the tape far enough that the ROM +
    KERNAL are resident (the title / code screen is fine)."""
    lo, hi = landscape & 0xFF, (landscape >> 8) & 0xFF
    log(f"  gen_enter ls{landscape}: reset/seed/generate/enemies/player ...")
    # HALT the CPU ONCE at the title key-scan loop; thereafter each _stub_call leaves it
    # halted at the JMP-self guard, so no live frame runs between steps. All pokes use
    # auto-resume OFF (bm.halted()) so the halted state is preserved throughout.
    _halt(bm)
    with bm.halted():
        _stub_call(bm, R_RESET)
        _stub_call(bm, R_SEED, x=lo, y=hi)
        # play flag CLEAR while initialise_player_and_trees runs so $14AF takes the
        # preview path (build table, clean RTS) -- not the obfuscated leave-to-play jump.
        bm.mem_set(A_PLAY_FLAG, bytes([bm.mem_get(A_PLAY_FLAG, A_PLAY_FLAG)[0] & 0x7F]))
        _stub_call(bm, R_GENERATE, stop_pc=GENERATE_END, timeout=20.0)
        _stub_call(bm, R_INIT_ENEMIES)
        _stub_call(bm, R_INIT_PLAYER)
        # in-play state, then JMP into the real play loop.
        bm.mem_set(A_VIEWPOINT, bytes([0x7F]))  # $141F = $7F (in-play LOS geometry)
        bm.mem_set(
            A_PLAY_FLAG, bytes([bm.mem_get(A_PLAY_FLAG, A_PLAY_FLAG)[0] | 0x80])
        )  # play semantics
        bm.mem_set(A_HYPERSPACED, bytes([0x00]))  # not hyperspaced / not complete
        log(f"  gen_enter: JMP play_landscape ${PLAY_LANDSCAPE:04x}")
        bm.registers_set({3: PLAY_LANDSCAPE, 4: 0xFD, 5: 0x20})
    bm.exit()  # resume the CPU into the play loop
    time.sleep(settle)


# ============================================================================
# container / connection (consolidated from the copies in live_climb /
# sentinel_record_plan / record_win_0042)
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

    def __init__(self, bm, container=None, log=print):
        self.bm = bm
        self.container = container
        self.log = log
        self.kbd = kbd_aim.KbdDriver(bm, log)

    @classmethod
    def boot(cls, log=print, attempts=4, record_mount=None):
        """Launch asid-vice and boot the tape to the title screen (saving a reusable
        boot snapshot if none exists). Returns a ready driver; call :meth:`close`."""
        container, bm = boot.boot_loaded(
            log=log, attempts=attempts, record_mount=record_mount
        )
        return cls(bm, container=container, log=log)

    def enter_landscape(self, landscape, settle=1.0):
        """Generate + enter `landscape` via the ROM's own routines (bypasses the
        secret-code gate); leaves the CPU in the interactive play loop."""
        generate_and_enter(self.bm, landscape, log=self.log, settle=settle)

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
