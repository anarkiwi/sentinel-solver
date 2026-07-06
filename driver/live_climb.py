#!/usr/bin/env python3
"""Live, 100% event-driven boulder-climb test of The Sentinel in asid-vice.

Boots from the code-entry VICE snapshot (renders/vice_code_entry.vsf) to skip the
tape load, enters landscape 0 through the menu (record_win's navigate -- which
installs the real gameplay IRQ), patches update_enemies ($16B5 -> RTS) so the
Sentinel can never see/absorb/drain the player, then repeatedly:

  * pick a VISIBLE tile whose terrain height is >= the player's current terrain
    height (discounting any boulder the player stands on), preferring higher ground
    and never immediately bouncing back to the tile just left,
  * create a boulder there, create a synthoid on top of it, TRANSFER into it,
  * absorb the synthoid + boulder the player just left,
  * repeat, trying to gain height, until a create/absorb/transfer fails or no
    reachable candidate tile remains.

Everything that touches the keyboard is EVENT-DRIVEN: a key is pressed only while
the CPU is halted at a ROM checkpoint, the CPU is advanced to the next ROM
checkpoint, and internal game memory is read to decide the next step. There are NO
wall-clock waits or fixed-frame holds anywhere in the aim/action path -- pans are
anchored on the per-attempt commit PC ($365D) and stall detection is purely a
state count, cursor moves are one-STA-per-press, and actions latch on a single
gated input scan. (run_until_pc's timeout is only a socket-hang guard; no control
decision is ever made from it.)

The aim target is chosen by sentinel.los (a pure-Python port of the ROM's own
$1C10+$1CDD aim path, verified bit-exact 12,800/12,800 vs the live ROM) and the
keys are driven to that exact (h, v, cursor); the live LOS probe is deliberately
NOT used -- it hijacks the CPU PC mid-plot and wedges the incremental plotter.
Each action is confirmed by its memory effect (object created/absorbed, POV moved).

Env:
  RECORD=1        also record an AVI (off by default; encoding slows the socket)
  RUPC=N          run_until_pc socket-guard timeout seconds (default 60)
  MAX_STEPS=N     cap climb iterations (default 40)
  STOP_AFTER_ENTER=1   enter + patch, short dwell, then stop (harness smoke test)
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

from vice_driver import BinMon, DiskMount, ViceContainer, keys
from driver import sentinel_state as gs
from sentinel.state import State
from sentinel import los
from driver.sentinel_execute import Executor

TAP = os.path.join(ROOT, "sentinel-gold.tap")
RENDERS = os.path.join(ROOT, "renders")
SNAP_HOST = os.path.join(RENDERS, "vice_code_entry.vsf")
SNAP_CONTAINER = "/renders/vice_code_entry.vsf"

# ---- game addresses ----
A_SLOT = 0x000B
A_H = 0x09C0
A_V = 0x0140
A_CX = 0x0CC6
A_CY = 0x0CC7
A_SFLAG = 0x0C5F
A_ENERGY = 0x0C0A
A_ZH = 0x0940
A_X = 0x0900
A_Y = 0x0980
A_FLAGS = 0x0100
A_TYPE = 0x0A40
UPDATE_ENEMIES = 0x16B5
A_LANDSCAPE_DONE = 0x0CDE  # bit6 set == landscape complete (ROM $2198)

# ---- ROM checkpoints (all reached deterministically by the play loop) ----
PC_PAN_DONE = 0x365D  # after JSR pan_viewpoint in the foreground loop; every attempt
PC_H_COMMIT = 0x10EE
PC_V_COMMIT = 0x1135
PC_CX_INC, PC_CX_DEC = 0x997C, 0x9990
PC_CY_INC, PC_CY_DEC = 0x99B8, 0x99D2
PC_IRQ_SCAN = 0x9678  # IRQ-side JSR check_for_full_player_input (a gated full scan)
PC_IRQ_SCAN_DONE = 0x967B

CX_CENTRE, CY_CENTRE = 80, 95

# run_until_pc timeout is only a socket-hang guard (never a control decision). AVI
# encoding periodically delays the binmon socket, so keep it generous.
RUPC = float(os.environ.get("RUPC", "60"))

# action-latch codes ($0CE9 value the key produces); poll $0CE8..$0CEB to confirm.
ACTION_CODE = {
    "R": 0x00,
    "T": 0x02,
    "B": 0x03,
    "A": 0x20,
    "Q": 0x21,
    "H": 0x22,
    "U": 0x23,
}
K_ABSORB, K_TRANSFER, K_ROBOT, K_BOULDER = "A", "Q", "R", "B"

T_ROBOT, T_BOULDER = 0, 3


def _k(name):
    return keys.lookup(name)


# ===========================================================================
# event-driven keyboard driver (BinMon) -- mirrors the proven kbd_sim protocol
# ===========================================================================
class EventDriver:
    """Drive the real keys to an exact (h, v, cursor) and fire actions, entirely
    from ROM checkpoints + memory reads. No wall-clock control decisions."""

    def __init__(self, bm, log):
        self.bm = bm
        self.log = log

    def rd(self, a):
        return self.bm.mem_get(a, a)[0]

    def _ru(self, pc, tries=int(os.environ.get("RU_TRIES", "5"))):
        """run_until_pc with retries. AVI encoding can transiently block the binmon
        socket past the guard timeout even though the game keeps reaching `pc`; a
        plain re-wait recovers it. This is still purely event-driven -- we only ever
        wait for a real ROM checkpoint, never sleep a fixed time or decide on a clock.
        """
        last = None
        for _ in range(tries):
            try:
                self.bm.run_until_pc(pc, timeout=RUPC)
                return
            except Exception as e:  # socket timeout during an AVI flush
                last = e
                try:
                    self.bm.exit()  # ensure the CPU is running before re-waiting
                except Exception:
                    pass
        raise last

    def slot(self):
        return self.rd(A_SLOT)

    def hang(self):
        return self.rd(A_H + self.slot())

    def vang(self):
        return self.rd(A_V + self.slot())

    def cur(self):
        return self.rd(A_CX), self.rd(A_CY)

    # ---- sights toggle: one gated full scan == one edge-latched toggle ----
    def sights_set(self, on):
        for _ in range(8):
            if bool(self.rd(A_SFLAG) & 0x80) == on:
                return True
            with self.bm.halted():
                try:
                    self._ru(PC_IRQ_SCAN)
                    r, c = _k("SPACE")
                    self.bm.keymatrix_set([(r, c, 1)])
                    self._ru(PC_IRQ_SCAN_DONE)
                finally:
                    self.bm.keymatrix_release_all()
            self.bm.exit()
        return bool(self.rd(A_SFLAG) & 0x80) == on

    # ---- pan one axis to an exact value, anchored on $365D (per-attempt) ----
    def _pan(self, addr, want, dir_fn, stall_bail=20, max_attempts=260):
        want &= 0xFF
        if self.rd(addr) == want:
            return True
        stalled = 0
        held = None
        with self.bm.halted():
            try:
                for _ in range(max_attempts):
                    cur = self.rd(addr)
                    if cur == want:
                        break
                    key = dir_fn(cur)
                    if key != held:
                        self.bm.keymatrix_release_all()
                        r, c = _k(key)
                        self.bm.keymatrix_set([(r, c, 1)])
                        held = key
                    self._ru(PC_PAN_DONE)  # one attempt
                    new = self.rd(addr)
                    self.bm.advance_instructions(1)  # step off $365D
                    if new == cur:  # undone or clamped this attempt
                        stalled += 1
                        if stalled >= stall_bail:
                            break
                    else:
                        stalled = 0
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()
        return self.rd(addr) == want

    def coarse_h(self, want):
        addr = A_H + self.slot()
        dir_fn = lambda cur: "D" if ((want - cur) & 0xFF) <= 0x80 else "S"
        return self._pan(addr, want, dir_fn)

    @staticmethod
    def _pitch_lin(v):
        v &= 0xFF
        return v - 0xCD if v >= 0xCD else v + 0x33

    def coarse_v(self, want):
        addr = A_V + self.slot()
        dir_fn = lambda cur: (
            "L" if self._pitch_lin(want) > self._pitch_lin(cur) else "COMMA"
        )
        return self._pan(addr, want, dir_fn)

    # ---- cursor: exactly one committed pixel per press ----
    def _cursor_step(self, key, sta_pc):
        with self.bm.halted():
            try:
                r, c = _k(key)
                self.bm.keymatrix_set([(r, c, 1)])
                self._ru(sta_pc)
                self.bm.advance_instructions(1)  # execute the STA: 1px committed
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()

    def _cursor_axis(self, addr, want, key_inc, key_dec, inc_pc, dec_pc, max_moves=200):
        want &= 0xFF
        for _ in range(max_moves):
            cur = self.rd(addr)
            if cur == want:
                return True
            if want > cur:
                self._cursor_step(key_inc, inc_pc)
            else:
                self._cursor_step(key_dec, dec_pc)
        return self.rd(addr) == want

    def fine_cursor(self, cx, cy):
        okx = self._cursor_axis(A_CX, cx, "D", "S", PC_CX_INC, PC_CX_DEC)
        oky = self._cursor_axis(A_CY, cy, "L", "COMMA", PC_CY_INC, PC_CY_DEC)
        return okx and oky

    # ---- fire an action key exactly once, confirmed by the want-flag latch ----
    def tap_action(self, name, max_passes=60):
        want = ACTION_CODE[name]
        r, c = _k(name)
        latched = False
        with self.bm.halted():
            try:
                for _ in range(max_passes):
                    # one full IDLE scan first: update_game clears $0C51 and only an
                    # idle full scan re-arms it, else a u-turn latch is dropped at $1B2F.
                    self._ru(PC_IRQ_SCAN)
                    self.bm.advance_instructions(1)
                    self._ru(PC_IRQ_SCAN)  # idle scan ran
                    self.bm.keymatrix_set([(r, c, 1)])
                    self._ru(PC_IRQ_SCAN_DONE)
                    flags = self.bm.mem_get(0x0CE8, 0x0CEB)
                    self.bm.keymatrix_release_all()
                    if want in flags:
                        latched = True
                        self._ru(PC_IRQ_SCAN)  # consumed
                        break
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()
        return latched

    def fire_hyperspace(self, tries=8):
        """Press H to hyperspace off the platform, confirmed by the ROM's own
        landscape-complete flag ($0CDE bit6) -- NOT by an action-latch code (H does
        not latch into $0CE8 like create/absorb). Bounded + flag-checked at the top of
        each pass so we never press again after the win transition (which would leave
        the input-scan loop and hang)."""
        r, c = _k("H")
        with self.bm.halted():
            try:
                for _ in range(tries):
                    if self.rd(A_LANDSCAPE_DONE) & 0x40:
                        break
                    self._ru(PC_IRQ_SCAN)
                    self.bm.advance_instructions(1)
                    self._ru(PC_IRQ_SCAN)  # idle scan re-arms the action gate
                    self.bm.keymatrix_set([(r, c, 1)])
                    self._ru(PC_IRQ_SCAN_DONE)
                    self.bm.keymatrix_release_all()
                    if self.rd(A_LANDSCAPE_DONE) & 0x40:
                        break
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()
        return bool(self.rd(A_LANDSCAPE_DONE) & 0x40)


# ===========================================================================
# aiming: snap a keyboard view via sentinel.los on live RAM (sentinel.los is bit-exact
# vs the ROM's $1C10+$1CDD -- verified 12,800/12,800), then DRIVE the live keys to
# that exact (h, v, cursor). We deliberately do NOT run the live LOS probe here: it
# forcibly redirects the CPU PC to an injected stub, which -- if done while the
# world is mid-plot ($0CE4 bit7 set) -- corrupts the incremental plotter's state
# and wedges it (never reaching the next input scan). Trusting native==ROM and
# verifying the ACTION's memory effect afterwards avoids ever hijacking the CPU.
#
# PERFORMANCE: kbd_aim.snap_keyboard_view falls back to a 1.38M-point cursor grid
# (~28 min in pure Python) whenever a tile can't be hit with the centred cursor --
# ruinous when probing candidates. But aim_target_native(h, v, cx, cy) depends only
# on the VIEW, not the target tile, so ONE centre-cursor sweep of the h*v lattice
# (~864 marches, ~1 s) yields the tile every reachable view lands on. reachable_map
# does that sweep once per POV (answering reachability for ALL candidates by dict
# lookup); fast_snap targets one tile with the centre sweep plus a small bounded
# cursor window (not the full 41x39 grid) as the only fallback.
# ===========================================================================
def _native(bm):
    m4k = bytes(bm.mem_get(0x0000, 0x0FFF))
    ps = m4k[A_SLOT]
    return State.from_mem(m4k), ps, m4k[A_ZH + ps], m4k[A_H + ps], m4k[A_V + ps]


def _grids(h0, v0):
    hgrid = [((h0 % 8) + 8 * k) & 0xFF for k in range(32)]
    band = list(range(0xCD, 0x100)) + list(range(0x00, 0x36))
    vgrid = [v for v in band if v % 4 == (v0 % 4)]
    return hgrid, vgrid


def reachable_map(bm, want_centre=False):
    """{(tx,ty): view} for every tile hit with LOS from the current POV using the
    CENTRED cursor -- one h*v lattice sweep (~1 s), reused for all candidates."""
    st, ps, eye, h0, v0 = _native(bm)
    hgrid, vgrid = _grids(h0, v0)
    best = {}
    for h in hgrid:
        for v in vgrid:
            rx, ry, los_hit, centre = los.aim_target(
                st,
                h,
                v,
                CX_CENTRE,
                CY_CENTRE,
                ps,
                eye_z=eye,
                max_steps=640,
                return_centre=True,
            )
            if not los_hit or (want_centre and centre >= 0x40):
                continue
            k = (rx, ry)
            if k not in best or centre < best[k][0]:
                best[k] = (
                    centre,
                    {"h_angle": h, "v_angle": v, "cursor": [CX_CENTRE, CY_CENTRE]},
                )
    return {k: v[1] for k, v in best.items()}


def fast_snap(bm, tile, want_centre=False):
    """A view whose native ray lands on `tile` with LOS: the centre-cursor h*v sweep
    first, then a SMALL bounded cursor window (not kbd_aim's full 41x39 grid) as the
    only fallback. Bounded so an unreachable tile costs ~seconds, never ~28 minutes."""
    st, ps, eye, h0, v0 = _native(bm)
    hgrid, vgrid = _grids(h0, v0)
    # bounded cursor offsets: centre first, then +-1 and +-2 lattice steps (9px grid)
    cxs = [CX_CENTRE, 71, 89, 62, 98]
    cys = [CY_CENTRE, 86, 104, 77, 113]

    def sweep(cx_list, cy_list):
        best = None
        for cx in cx_list:
            for cy in cy_list:
                for h in hgrid:
                    for v in vgrid:
                        rx, ry, los_hit, centre = los.aim_target(
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
                        if (rx, ry) != tile or not los_hit:
                            continue
                        if want_centre and centre >= 0x40:
                            continue
                        view = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
                        if best is None or centre < best[0]:
                            best = (centre, view)
                        if centre < 0x20:
                            return best
        return best

    hit = sweep([CX_CENTRE], [CY_CENTRE])  # phase 1 (fast, common case)
    if hit is None:
        hit = sweep(cxs, cys)  # bounded phase 2
    return hit[1] if hit else None


def aim_at(drv, ex, tile, want_centre=False):
    """Drive the sights onto `tile`. Returns (tx, ty, True) when the live view was
    driven to a snapped keyboard view that sentinel.los lands on `tile` with LOS, or
    None if no view exists / the coarse pan could not reach it."""
    # Compute the snapped view with the CPU HALTED: fast_snap is ~0.8 s of pure
    # Python, and if the CPU keeps running (esp. under warp) the enemies advance
    # thousands of frames while we think -- the Sentinel rotates onto the player
    # and drains/downgrades the board out from under the plan. Freezing across the
    # decision is what keeps a live Sentinel from diverging the replay.
    with drv.bm.halted():
        view = fast_snap(drv.bm, tile, want_centre)
    drv.bm.exit()
    if view is None:
        drv.log(f"    aim {tile}: no keyboard view (bounded snap)")
        return None
    if not drv.sights_set(False):
        return None
    okh = drv.coarse_h(view["h_angle"])
    okv = drv.coarse_v(view["v_angle"])
    if not (okh and okv):
        okh = drv.coarse_h(view["h_angle"])
        okv = drv.coarse_v(view["v_angle"])
    if not (okh and okv):
        drv.log(f"    aim {tile}: coarse pan miss (h ok={okh} v ok={okv})")
        return None
    if not drv.sights_set(True):
        return None
    okc = drv.fine_cursor(*view["cursor"])
    if not okc:
        drv.log(f"    aim {tile}: cursor drive miss")
        return None
    return (tile[0], tile[1], True)


# ===========================================================================
# climb planning from live state
# ===========================================================================
def read_state(bm):
    return gs.read_game_state(gs.ViceSource(bm))


def terrain_height(bm, x, y):
    """Terrain-only height (whole units) at (x,y): the tile nibble high bits,
    or the resolved state height when an object occupies the tile byte."""
    t = bm.mem_get(0x0400 + (y * 32 + x), 0x0400 + (y * 32 + x))[0]
    if t < 0xC0:
        return t >> 4
    st = read_state(bm)
    return st.height[y][x]


def player_terrain_tile(bm):
    ps = bm.mem_get(A_SLOT, A_SLOT)[0]
    px = bm.mem_get(A_X + ps, A_X + ps)[0]
    py = bm.mem_get(A_Y + ps, A_Y + ps)[0]
    return px, py, terrain_height(bm, px, py)


def _obj_arrays(bm):
    """Bulk-read the 64-slot object arrays (flags, x, y, type) in 4 socket calls."""
    fl = bm.mem_get(A_FLAGS, A_FLAGS + 63)
    xs = bm.mem_get(A_X, A_X + 63)
    ys = bm.mem_get(A_Y, A_Y + 63)
    ty = bm.mem_get(A_TYPE, A_TYPE + 63)
    return fl, xs, ys, ty


def slots_at_tile(bm, x, y):
    """List of (slot, type) of all live objects occupying tile (x,y)."""
    fl, xs, ys, ty = _obj_arrays(bm)
    return [
        (i, ty[i])
        for i in range(64)
        if not (fl[i] & 0x80) and xs[i] == x and ys[i] == y
    ]


def object_in_tile(bm, x, y):
    """Return (slot, type) of the TOPMOST object on tile (x,y) (the stack head:
    the live object at the tile that no other live object is stacked on), or None.
    Stacking is recorded in objects_flags: a value $40-$7F means "on object N"."""
    fl, xs, ys, ty = _obj_arrays(bm)
    here = [i for i in range(64) if not (fl[i] & 0x80) and xs[i] == x and ys[i] == y]
    if not here:
        return None
    supporting = {fl[i] & 0x3F for i in range(64) if 0x40 <= fl[i] <= 0x7F}
    top = [i for i in here if i not in supporting]
    slot = top[0] if top else max(here)
    return slot, ty[slot]


def candidate_tiles(bm, radius=6, exclude=()):
    """Visible-candidate tiles: terrain height >= player's terrain height, not the
    player's own tile (or any `exclude` tile), empty. Ordered to TRY TO GO HIGHER:
    strictly-higher terrain first (highest first), then equal-height tiles, nearest
    first within each. Excluding the tile we just came from stops a two-tile
    A<->B leapfrog and forces the climb to seek higher ground until none remains."""
    px, py, ph = player_terrain_tile(bm)
    exclude = set(exclude) | {(px, py)}
    higher, equal = [], []
    for ty in range(max(0, py - radius), min(32, py + radius + 1)):
        for tx in range(max(0, px - radius), min(32, px + radius + 1)):
            if (tx, ty) in exclude:
                continue
            th = terrain_height(bm, tx, ty)
            if th < ph:
                continue
            if object_in_tile(bm, tx, ty) is not None:
                continue
            d = abs(tx - px) + abs(ty - py)
            (higher if th > ph else equal).append((-th, d, (tx, ty)))
    higher.sort()
    equal.sort()
    return [c[2] for c in higher] + [c[2] for c in equal]


# ===========================================================================
# live actions (aim -> fire -> verify from memory; self-healing retries)
# ===========================================================================
def energy(bm):
    return bm.mem_get(A_ENERGY, A_ENERGY)[0] & 0x3F


def do_create(drv, ex, tile, otype, key, tries=3):
    """Aim at `tile` and create an object of `otype` there. Verify a NEW object of
    that exact type appears at the tile (the create stacks on whatever is there).
    Returns the created slot or None."""

    def typed_slots():
        return {s for s, t in slots_at_tile(drv.bm, *tile) if t == otype}

    before = typed_slots()
    if before:  # already one of this type here (e.g. a prior partial attempt)
        return max(before)
    for attempt in range(tries):
        probe = aim_at(drv, ex, tile, want_centre=(attempt == 0))
        if probe is None or (probe[0], probe[1]) != tile or not probe[2]:
            drv.log(f"    create {otype}@{tile}: aim miss {probe} (try {attempt})")
            continue
        e0 = energy(drv.bm)
        drv.tap_action(key)
        new = typed_slots() - before
        if new:
            slot = max(new)
            drv.log(
                f"    created type{otype} @ {tile} slot{slot} "
                f"energy {e0}->{energy(drv.bm)}"
            )
            return slot
        drv.log(
            f"    create {otype}@{tile}: no new type{otype} object "
            f"(energy {e0}->{energy(drv.bm)})"
        )
    return None


def do_transfer(drv, ex, tile, tries=3):
    """Transfer POV into the robot on `tile`. Verify the player slot moved there."""
    for attempt in range(tries):
        probe = aim_at(drv, ex, tile, want_centre=(attempt == 0))
        if probe is None or (probe[0], probe[1]) != tile or not probe[2]:
            drv.log(f"    transfer@{tile}: aim miss {probe} (try {attempt})")
            continue
        drv.tap_action(K_TRANSFER)
        ps = drv.slot()
        px = drv.bm.mem_get(A_X + ps, A_X + ps)[0]
        py = drv.bm.mem_get(A_Y + ps, A_Y + ps)[0]
        if (px, py) == tile:
            drv.log(f"    transferred to slot{ps} @ {tile}")
            return True
        drv.log(f"    transfer@{tile}: player still slot{ps} @ ({px},{py})")
    return False


def do_absorb(drv, ex, tile, tries=3):
    """Absorb the topmost object on `tile`. Verify the object is gone + energy up."""
    occ = object_in_tile(drv.bm, *tile)
    if occ is None:
        return True  # nothing to absorb
    for attempt in range(tries):
        probe = aim_at(drv, ex, tile, want_centre=(attempt == 0))
        if probe is None or (probe[0], probe[1]) != tile or not probe[2]:
            drv.log(f"    absorb@{tile}: aim miss {probe} (try {attempt})")
            continue
        e0 = energy(drv.bm)
        drv.tap_action(K_ABSORB)
        occ2 = object_in_tile(drv.bm, *tile)
        if occ2 is None or occ2[0] != occ[0]:
            drv.log(f"    absorbed slot{occ[0]} @ {tile} energy {e0}->{energy(drv.bm)}")
            return True
        drv.log(
            f"    absorb@{tile}: object still present (energy {e0}->{energy(drv.bm)})"
        )
    return False


# ===========================================================================
# container / connection
# ===========================================================================
def _bridge_ip(container_id, log):
    import subprocess

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
    except Exception as e:
        log(f"  bridge-ip lookup failed: {e}")
        return None


def _free_containers(log):
    import subprocess

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


def connect(container, log):
    time.sleep(2)
    host = os.environ.get("BINMON_HOST") or _bridge_ip(container.container_id, log)
    if not host:
        host = "127.0.0.1"
    port = int(os.environ.get("BINMON_PORT", "6502"))
    log(f"  connecting binmon {host}:{port}")
    bm = BinMon(host, port)
    bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
    bm.exit()
    return bm


def enter_landscape0(bm, log):
    """Restore the code-entry snapshot (skip tape boot) and enter landscape 0 the
    normal way -- through the menu -- which installs the real gameplay IRQ. This is
    record_win's proven navigate() path (menu typing to reach play); the keyboard
    aim/action loop below is the fully event-driven part."""
    import record_win_0042 as rw

    log("entering landscape 0 via menu (record_win navigate, from snapshot)")
    rw.navigate(bm, "0000", log, SNAP_CONTAINER, SNAP_HOST)
    # patch update_enemies -> RTS: the Sentinel/sentries can never see/absorb the
    # player. FREEZE=0 leaves the Sentinel fully live (real-game solve).
    if os.environ.get("FREEZE", "1") == "1":
        bm.mem_set(UPDATE_ENEMIES, bytes([0x60]))
        log(f"patched update_enemies ${UPDATE_ENEMIES:04x} = RTS (enemies frozen)")
    else:
        log("FREEZE=0: Sentinel LEFT ENABLED (real-game solve)")


# ===========================================================================
# main climb loop
# ===========================================================================
def climb(drv, ex, log, max_steps):
    px, py, ph = player_terrain_tile(drv.bm)
    log(f"START: player @ ({px},{py}) terrain h={ph} energy={energy(drv.bm)}")
    prev_tile = None  # the tile we came from last step (excluded to avoid A<->B)
    for step in range(max_steps):
        try:
            old_px, old_py, old_ph = player_terrain_tile(drv.bm)
            old_stack = (old_px, old_py)
            cands = candidate_tiles(drv.bm, exclude=(prev_tile,) if prev_tile else ())
            log(
                f"[step {step}] on ({old_px},{old_py}) h={old_ph} "
                f"energy={energy(drv.bm)} {len(cands)} candidate tiles"
            )
            if energy(drv.bm) < 5:
                log(f"  energy {energy(drv.bm)} < 5: cannot afford boulder+synthoid")
                return "out_of_energy"
            made_progress = False
            # ONE centre-cursor sweep from this POV answers reachability for every
            # candidate (aim_target_native depends only on the view, not the target),
            # so selection is a dict lookup -- not a ~28-min snap per unreachable tile.
            rmap = reachable_map(drv.bm)
            chosen = next((tile for tile in cands if tile in rmap), None)
            # rmap answers CENTRE-cursor reachability at every (h,v). A tile that no
            # centre view lands on may still be hit by an off-centre fine-cursor nudge
            # (fast_snap phase-2). That path costs ~20 s per miss, so only fall back to
            # it when the cheap lookup found nothing -- i.e. right where we'd otherwise
            # stop -- and only on the few best candidates.
            if chosen is None:
                for tile in cands[:4]:
                    if fast_snap(drv.bm, tile) is not None:
                        chosen = tile
                        log(f"  (off-centre fallback reached {tile})")
                        break
            for tile in [chosen] if chosen else []:
                log(f"  chosen tile {tile} (terrain h={terrain_height(drv.bm, *tile)})")
                bslot = do_create(drv, ex, tile, T_BOULDER, K_BOULDER)
                if bslot is None:
                    log("  boulder create FAILED -> stop")
                    return "create_boulder_failed"
                rslot = do_create(drv, ex, tile, T_ROBOT, K_ROBOT)
                if rslot is None:
                    log("  synthoid create FAILED -> stop")
                    return "create_synthoid_failed"
                if not do_transfer(drv, ex, tile):
                    log("  transfer FAILED -> stop")
                    return "transfer_failed"
                # absorb the old stack (synthoid then boulder) from the new vantage
                if old_stack != tile:
                    occ = object_in_tile(drv.bm, *old_stack)
                    while occ is not None:
                        if not do_absorb(drv, ex, old_stack):
                            log(f"  absorb old stack @ {old_stack} FAILED -> stop")
                            return "absorb_failed"
                        nxt = object_in_tile(drv.bm, *old_stack)
                        if nxt is not None and nxt[0] == occ[0]:
                            break
                        occ = nxt
                made_progress = True
                prev_tile = old_stack  # don't immediately bounce back here next step
                npx, npy, nph = player_terrain_tile(drv.bm)
                log(
                    f"  -> now on ({npx},{npy}) terrain h={nph} energy={energy(drv.bm)}"
                )
                break
            if not made_progress:
                log("  no reachable candidate tile -> stop")
                return "no_reachable_tile"
        except Exception as e:
            # any driver stall (socket timeout, checkpoint miss) -> stop the climb
            # cleanly so the recording can be finalized; leave no key held / CPU halted.
            log(f"  driver stall/error: {type(e).__name__}: {e} -> stop cleanly")
            try:
                drv.bm.keymatrix_release_all()
            except Exception:
                pass
            try:
                drv.bm.exit()
                pc = drv.bm.registers_get()[3]
                plotting = drv.rd(0x0CE4)
                pan = drv.rd(0x0009)
                sflag = drv.rd(A_SFLAG)
                ce9 = drv.rd(0x0CE9)
                log(
                    f"  STALL STATE: PC=${pc:04x} $0CE4(plotting)=${plotting:02x} "
                    f"$0009(pan)=${pan:02x} sights=${sflag:02x} $0CE9=${ce9:02x}"
                )
            except Exception as e2:
                log(f"  (stall-state read failed: {e2})")
            try:
                drv.bm.exit()
            except Exception:
                pass
            return "driver_stall"
    return "max_steps"


CREATE_KEY = {0: K_ROBOT, 3: K_BOULDER, 2: "T"}


def solve_from_plan(drv, ex, log, steps):
    """Execute a native-won plan (out/kbd_search_*.json) with the proven event-driven
    primitives, then hyperspace off the platform for the real ROM win ($0CDE bit6)."""
    px, py, ph = player_terrain_tile(drv.bm)
    log(
        f"START: player @ ({px},{py}) terrain h={ph} energy={energy(drv.bm)} "
        f"({len(steps)} plan steps)"
    )
    for i, stp in enumerate(steps):
        verb, tile = stp["verb"], tuple(stp["target"])
        otype = stp.get("otype")
        e0 = energy(drv.bm)
        log(f"[{i:2}] {verb} {tile} otype={otype} energy={e0}")
        if verb == "create":
            key = CREATE_KEY.get(otype, K_ROBOT)
            if do_create(drv, ex, tile, otype, key) is None:
                return f"create_failed@step{i}"
        elif verb == "absorb":
            if not do_absorb(drv, ex, tile):
                return f"absorb_failed@step{i}"
        elif verb == "transfer":
            if not do_transfer(drv, ex, tile):
                return f"transfer_failed@step{i}"
        else:
            return f"unknown_verb@step{i}:{verb}"
    # endgame: player should be standing on the platform tile; hyperspace = the win.
    ps = drv.slot()
    px = drv.bm.mem_get(A_X + ps, A_X + ps)[0]
    py = drv.bm.mem_get(A_Y + ps, A_Y + ps)[0]
    log(
        f"plan done: player slot{ps} @ ({px},{py}) energy={energy(drv.bm)}; hyperspacing"
    )
    won = drv.fire_hyperspace()
    done = drv.rd(A_LANDSCAPE_DONE)
    log(f"$0CDE = {done:#04x} (bit6 = landscape complete)")
    return "WON" if won else f"no_win_flag(${done:02x})"


def main():
    def log(m):
        print(m, flush=True)

    if not os.path.exists(TAP):
        raise FileNotFoundError(f"{TAP} missing")
    if not os.path.exists(SNAP_HOST):
        raise FileNotFoundError(f"{SNAP_HOST} missing (code-entry snapshot)")
    os.makedirs(RENDERS, exist_ok=True)
    max_steps = int(os.environ.get("MAX_STEPS", "40"))
    record = os.environ.get("RECORD") == "1"  # recording off by default
    stop_after_enter = os.environ.get("STOP_AFTER_ENTER") == "1"
    video_name = "live_climb.avi"
    video_host = os.path.join(RENDERS, video_name)
    if os.path.exists(video_host):
        try:
            os.remove(video_host)
        except OSError:
            pass

    _free_containers(log)
    container = ViceContainer(
        autostart="/work/sentinel.tap",
        mounts=[
            DiskMount(TAP, "/work/sentinel.tap", read_only=True),
            DiskMount(RENDERS, "/renders", read_only=False),
        ],
        warp=True,
        silent=True,
    )
    outcome = "error"
    with container:
        bm = connect(container, log)
        try:
            enter_landscape0(bm, log)
            st = read_state(bm)
            if st.player is None:
                log("NOT IN PLAY (no player) -- aborting")
                return
            log(
                f"IN PLAY: slot {st.player_slot} @ ({st.player.x},{st.player.y}) "
                f"energy {st.player_energy} objs {len(st.objects)}"
            )
            if record:
                log(f"-- recording -> {video_host} --")
                bm.video_record(f"/renders/{video_name}")
                time.sleep(0.5)
            ex = Executor(bm, log)
            if stop_after_enter:
                log("STOP_AFTER_ENTER: short dwell then stop")
                time.sleep(2.0)
                outcome = "entered_only"
            elif os.environ.get("PLAN"):
                import json as _json

                with open(os.environ["PLAN"]) as pf:
                    steps = _json.load(pf)["steps"]
                log(f"executing plan {os.environ['PLAN']} ({len(steps)} steps)")
                outcome = solve_from_plan(EventDriver(bm, log), ex, log, steps)
            else:
                outcome = climb(EventDriver(bm, log), ex, log, max_steps)
            log(f"CLIMB OUTCOME: {outcome}")
        finally:
            try:
                if record:
                    time.sleep(1.0)
                    bm.video_stop()
                    time.sleep(1.0)
                    log(f"-- recording stopped -> {video_host} --")
            except Exception as e:
                log(f"  video_stop failed: {e}")
            try:
                bm.close()
            except Exception:
                pass
    log(f"DONE: outcome={outcome} video={'(none)' if not record else video_host}")


if __name__ == "__main__":
    main()
