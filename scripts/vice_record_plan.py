#!/usr/bin/env python3
"""Open-loop KEYBOARD-DRIVEN replay of a native-snapped Sentinel plan inside asid-vice,
recorded to a real AVI, with the genuine ROM win flag ($0CDE bit6) verified.

This is the authoritative real-ROM execution: the native snap (snap_kbd_plan.py) only
PREDICTS the win; this run PROVES it. Every input that changes game state is a real
keystroke (S/D/L/COMMA pan + sights cursor, U u-turn, R/B robot/boulder, Q transfer,
A absorb, H hyperspace, SPACE sights). We READ memory only to (a) confirm an aim
reached its snapped view and (b) verify each action's state delta -- reads never
change game state. If VICE ever diverges from the native prediction we ABORT and
report it (that would be a sentinel.los bug), rather than silently retrying.

Boot/load/menu nav run under WARP (not the recorded gameplay). Recording wraps the
in-game actions at true speed so the AVI is watchable.

Confirmed live key map (decoded from the ROM key tables $138D/$139C via keynum=col*8+row,
verified by reading the latched action code $0C61):
  R=create robot($00)  B=create boulder($03)  Q=transfer($21)  A=absorb($20)
  H=hyperspace($22)    U=u-turn($23, EOR $80) SPACE=toggle sights ($0C5F bit7)
  sights OFF: S/D pan objects_h_angle -8/+8 ; L/COMMA pan objects_v_angle +4/-4
  sights ON : S/D move sights cursor x -5/+5 ; L/COMMA move cursor y +5/-5
  pan/cursor are plot-gated: a press registers only when $0CE4 bit7 (world-being-
  plotted) is clear, so we wait on that flag between presses (cheap read, not a stop).

ls42 needs a SECRET ENTRY CODE; we patch the two ROM code-checks in live RAM so any
code is accepted (the GAMEPLAY is unaffected and fully authentic):
  $14DF 29 1E (AND #$1E) -> A9 1E (LDA #$1E): first_secret_code_check always equal.
  $2565 D0 71 (BNE reject) -> EA EA (NOP): second_secret_code_check per-digit no-reject.
  $2570 F0 66 (BEQ reject) -> EA EA (NOP): tamper branch no-reject.
"""

import os, sys, time, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "vice-driver")))

from vice_driver import BinMon, DiskMount, ViceContainer, keys
from vice_driver.binmon import TAP_MODE_FIXED
from vice_driver.display import parse_display_response, parse_palette_response

import vice_state as gs

TAP = os.path.join(ROOT, "sentinel-gold.tap")

# ---- addresses (see module docstring) ----------------------------------------
A_PLAYER_SLOT = 0x000B
A_OBJ_H = 0x09C0  # + slot
A_OBJ_V = 0x0140  # + slot
A_OBJ_X = 0x0900
A_OBJ_Y = 0x0980
A_OBJ_Z = 0x0940
A_OBJ_TYPE = 0x0A40
A_OBJ_FLAGS = 0x0100
A_SIGHTS_X = 0x0CC6
A_SIGHTS_Y = 0x0CC7
A_SIGHTS_FLAG = 0x0C5F  # bit7 set == sights active
A_ENERGY = 0x0C0A
A_TARGET_X = 0x0024  # action-time LOS-marched target tile
A_TARGET_Y = 0x0026
A_LANDSCAPE_DONE = 0x0CDE  # bit6 set == landscape complete
A_PLAT_X = 0x0C19
A_PLAT_Y = 0x0C1A
A_ACTION_CODE = 0x0C61
A_PLOTTING = 0x0CE4  # bit7 set == world being plotted (pan/cursor gate)

# secret-code-check patches
PATCHES = [
    (0x14DF, bytes([0xA9, 0x1E])),
    (0x2565, bytes([0xEA, 0xEA])),
    (0x2570, bytes([0xEA, 0xEA])),
]

# action keys (live-confirmed)
K_ROBOT, K_BOULDER, K_TREE = "R", "B", "T"
K_TRANSFER, K_ABSORB, K_HYPERSPACE, K_UTURN = "Q", "A", "H", "U"
K_SPACE = "SPACE"
# sights/pan keys
K_LEFT, K_RIGHT, K_DOWN, K_UP = "S", "D", "L", "COMMA"

SIGHTS_CX, SIGHTS_CY = 0x50, 0x5F


class Driver:
    def __init__(self, bm, log):
        self.bm = bm
        self.log = log
        self.ps = self.rd(A_PLAYER_SLOT)

    def rd(self, a):
        return robust(self.bm, self.log, lambda: self.bm.mem_get(a, a)[0])

    def hang(self):
        return self.rd(A_OBJ_H + self.rd(A_PLAYER_SLOT))

    def vang(self):
        return self.rd(A_OBJ_V + self.rd(A_PLAYER_SLOT))

    def state(self):
        return gs.read_game_state(gs.ViceSource(self.bm))

    # ---- plot-gated key press: hold the key, then wait for the world re-plot to
    # finish ($0CE4 bit7 clears) so the NEXT press is not swallowed. bounded. ----
    def press(self, name, hold=16, max_wait=2.5):
        self.bm.keymatrix_tap([keys.lookup(name)], mode=TAP_MODE_FIXED, frames=hold)
        t0 = time.time()
        time.sleep(0.04)
        while time.time() - t0 < max_wait and (self.rd(A_PLOTTING) & 0x80):
            time.sleep(0.02)
        time.sleep(0.06)

    def tap(self, name, hold=14, settle=0.35):
        self.bm.keymatrix_tap([keys.lookup(name)], mode=TAP_MODE_FIXED, frames=hold)
        time.sleep(settle)

    def sights(self, on):
        cur = bool(self.rd(A_SIGHTS_FLAG) & 0x80)
        if cur != on:
            self.tap(K_SPACE, hold=18, settle=0.5)
        return bool(self.rd(A_SIGHTS_FLAG) & 0x80)

    # ---- pan objects_h_angle to an absolute target via S(-8)/D(+8), preferring the
    # shorter signed direction; U-turn (EOR $80) shortcuts ~half the circle. -------
    def pan_h_to(self, target, max_presses=48):
        target &= 0xFF
        stuck = 0
        for _ in range(max_presses):
            cur = self.hang()
            if cur == target:
                return True
            diff = (target - cur) & 0xFF
            if 0x60 <= diff <= 0xA0:  # ~half turn away: u-turn (EOR $80)
                self.press(K_UTURN)
                continue
            signed = diff if diff < 128 else diff - 256
            before = cur
            self.press(K_RIGHT if signed > 0 else K_LEFT)
            if self.hang() == before:  # press swallowed (D is auto-repeat flaky)
                stuck += 1
                if stuck > 3:
                    # fall back to the reliable S key (always -8) toward target
                    self.press(K_LEFT)
                    stuck = 0
            else:
                stuck = 0
        return self.hang() == target

    def pan_v_to(self, target, max_presses=48):
        target &= 0xFF
        for _ in range(max_presses):
            cur = self.vang()
            if cur == target:
                return True
            diff = (target - cur) & 0xFF
            signed = diff if diff < 128 else diff - 256
            self.press(K_DOWN if signed > 0 else K_UP)  # L=+4 (down), COMMA=-4 (up)
        return self.vang() == target

    # ---- move the sights cursor to an absolute (cx,cy) via S/D (x -+5) and
    # L/COMMA (y +-5) with sights ON. ----------------------------------------------
    def cursor_to(self, cx, cy, max_presses=48):
        for _ in range(max_presses):
            x = self.rd(A_SIGHTS_X)
            if x == cx:
                break
            self.press(K_RIGHT if cx > x else K_LEFT)
        for _ in range(max_presses):
            y = self.rd(A_SIGHTS_Y)
            if y == cy:
                break
            self.press(K_DOWN if cy > y else K_UP)
        return (self.rd(A_SIGHTS_X), self.rd(A_SIGHTS_Y))

    def aim(self, view):
        """Drive obj_h/obj_v (sights OFF) then sights cursor (sights ON) to the
        snapped view's absolute values, all via real keys. Returns (ok, achieved)."""
        th, tv = view["h_angle"] & 0xFF, view["v_angle"] & 0xFF
        cx, cy = view["cursor"]
        self.sights(False)
        self.pan_h_to(th)
        self.pan_v_to(tv)
        self.sights(True)
        gx, gy = (self.rd(A_SIGHTS_X), self.rd(A_SIGHTS_Y))
        if (cx, cy) != (gx, gy):
            gx, gy = self.cursor_to(cx, cy)
        ach = {"h": self.hang(), "v": self.vang(), "cur": (gx, gy)}
        ok = self.hang() == th and self.vang() == tv and (gx, gy) == (cx, cy)
        return ok, ach


def _reconnect(bm, log):
    """Reconnect the binary-monitor socket to the SAME (still-running) container after
    an intermittent drop, so a flaky warp boot does not cost a full container restart.
    """
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
    """Run a binmon op, reconnecting on a dropped socket (the asid-vice:latest monitor
    flakes during warp boot). Reads/keystrokes are idempotent enough to retry."""
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
            _reconnect(bm, log)
    return fn()  # last attempt: let it raise


def navigate(bm, landscape, log):
    """Boot under WARP to the LANDSCAPE NUMBER prompt, patch the secret-code checks,
    type the number + a dummy code, dismiss the preview, enter play. Tolerant of the
    intermittent monitor-socket drop during the warp load (reconnects in place)."""

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

    log("booting + loading (warp)...")
    # poll a cheap read each second during the warp load; reconnect on a drop so the
    # boot survives the flake without restarting the container.
    for _ in range(50):
        time.sleep(1.0)
        robust(bm, log, lambda: bm.mem_get(0x00, 0x00))
    for _ in range(3):
        tap(K_SPACE, hold=30, settle=1.5)
    log("patching secret-code checks ($14DF/$2565/$2570) so ls plays with any code")
    for addr, data in PATCHES:
        robust(bm, log, lambda a=addr, d=data: bm.mem_set(a, d))
    tap_text(f"{landscape:04d}")
    tap("RETURN", hold=30, settle=3.0)
    tap_text("00000000")  # dummy secret code (accepted via the patch)
    tap("RETURN", hold=30, settle=8.0)
    time.sleep(3)
    tap(K_SPACE, hold=25, settle=1.2)  # dismiss the isometric preview
    time.sleep(4)


def objects_at(state, x, y):
    return [o for o in state.objects if o.x == x and o.y == y]


def _free_port_6502(log):
    """Force-remove any lingering asid-vice container so port 6502 is free for the
    next boot attempt (a BrokenPipe on the monitor can leave a container running)."""
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
                ["docker", "rm", "-f", *ids], capture_output=True, text=True, timeout=30
            )
            time.sleep(2)
    except Exception as e:
        log(f"  port-cleanup warning: {e}")


def _bridge_ip(container_id, log):
    """Docker-bridge IP of the container. In this environment the published
    loopback port (127.0.0.1:6502) is refused; the container's bridge IP is
    directly reachable (same path live_climb.connect uses)."""
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


def run(landscape, plan_path, max_seconds, log):
    with open(plan_path) as f:
        plan = json.load(f)
    if not plan.get("won"):
        log(
            f"REFUSING: snapped plan {plan_path} is not a native win (won={plan.get('won')})"
        )
        return {"won": False, "note": "plan not validated", "landscape": landscape}
    steps = plan["steps"]

    renders_host = os.path.join(ROOT, "renders")
    os.makedirs(renders_host, exist_ok=True)
    video_host = os.path.join(renders_host, f"solver_run_{landscape:04d}.avi")
    if os.path.exists(video_host):
        try:
            os.remove(video_host)
        except OSError:
            pass

    result = {
        "won": False,
        "landscape": landscape,
        "video": video_host,
        "actions": [],
        "divergence": None,
    }
    # The asid-vice:latest binary monitor can intermittently drop the socket during
    # the warp tape boot (observed ~12-34s in). Boot is NOT the recorded gameplay, so
    # we retry the whole container+boot up to BOOT_TRIES times until we are safely in
    # play, then run the (single, authoritative) recorded session.
    BOOT_TRIES = 8
    for boot_try in range(BOOT_TRIES):
        _free_port_6502(log)  # ensure no stale container holds the monitor port
        container = ViceContainer(
            autostart="/work/sentinel.tap",
            mounts=[
                DiskMount(TAP, "/work/sentinel.tap", read_only=True),
                DiskMount(renders_host, "/renders", read_only=False),
            ],
            warp=True,
            silent=True,
        )
        t_start = time.time()
        try:
            with container:
                time.sleep(2)
                host = (
                    os.environ.get("BINMON_HOST")
                    or _bridge_ip(container.container_id, log)
                    or "127.0.0.1"
                )
                log(f"  connecting binmon {host}:6502")
                bm = BinMon(host, 6502)
                bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
                bm.exit()
                pal = parse_palette_response(bm.palette_get())

                def grab(tag):
                    try:
                        snap = parse_display_response(bm.display_get())
                        snap.save_png(os.path.join(renders_host, f"rec_{tag}.png"), pal)
                    except Exception as e:
                        log(f"  grab {tag} failed: {e}")

                navigate(bm, landscape, log)
                d = Driver(bm, log)
                st = d.state()
                if st.player is None:
                    log(
                        f"boot try {boot_try}: not in play (no player); restarting container"
                    )
                    continue
                plat = (d.rd(A_PLAT_X), d.rd(A_PLAT_Y))
                log(
                    f"IN PLAY: player slot {st.player_slot} @ ({st.player.x},{st.player.y}) "
                    f"energy {st.player_energy} objs {len(st.objects)} platform {plat} "
                    f"h0=${d.hang():02x} v0=${d.vang():02x}"
                )
                grab("play_start")

                # ---- START REAL VIDEO RECORDING (true speed, watchable) ----------
                log(f"-- starting AVI recording -> {video_host} --")
                try:
                    bm.video_record(f"/renders/solver_run_{landscape:04d}.avi")
                except Exception as e:
                    log(f"  video_record failed: {e}")
                time.sleep(1.0)

                won = execute(
                    d, bm, steps, plat, grab, log, result, t_start, max_seconds
                )
                result["won"] = won
                grab("final")
                time.sleep(1.5)

                log("-- stopping AVI recording (finalize) --")
                try:
                    bm.video_stop()
                    time.sleep(1.5)
                except Exception as e:
                    log(f"  video_stop failed: {e}")
                result["wall_seconds"] = round(time.time() - t_start, 1)
                bm.close()
            return result  # in-play run completed (won or diverged) -> done
        except Exception as e:
            import traceback

            log(f"boot try {boot_try}: container/boot error: {type(e).__name__}: {e}")
            if boot_try == 0:
                traceback.print_exc()
            # if we already started gameplay (have actions logged), do NOT retry --
            # that would be a real divergence, not a boot flake.
            if result["actions"]:
                result["divergence"] = result.get("divergence") or f"mid-run drop: {e}"
                return result
            time.sleep(2)
    result["divergence"] = f"could not boot into play after {BOOT_TRIES} tries"
    return result


def execute(d, bm, steps, plat, grab, log, result, t_start, max_seconds):
    """Walk the snapped steps with real keystrokes, verifying each action's live
    delta. Abort+report on any divergence from the native prediction."""
    for i, stp in enumerate(steps):
        if time.time() - t_start > max_seconds:
            log(f"TIME BUDGET ({max_seconds}s) exceeded at step {i}; aborting")
            result["divergence"] = f"timeout at step {i}"
            return False
        verb, tile, otype = stp["verb"], tuple(stp["target"]), stp["otype"]
        view = stp.get("view")
        before = d.state()
        e0 = before.player_energy
        objs0 = len(objects_at(before, *tile))
        slot0 = before.player_slot

        # --- AIM (only create/absorb carry a view; transfer/blind-create do not) ---
        aim_info = None
        if view is not None:
            ok, ach = d.aim(view)
            aim_info = ach
            tgt = (d.rd(A_TARGET_X), d.rd(A_TARGET_Y))
            if not ok:
                log(
                    f"[{i:2}] {verb} {tile}: AIM FAILED ach={ach} want h=${view['h_angle']:02x} "
                    f"v=${view['v_angle']:02x} cur={view['cursor']}"
                )
                result["divergence"] = (
                    f"step {i} aim could not reach snapped view "
                    f"(got {ach}, want {view})"
                )
                grab(f"aimfail_{i:02d}")
                return False
            log(
                f"[{i:2}] {verb} {tile}: aimed h=${ach['h']:02x} v=${ach['v']:02x} "
                f"cur={ach['cur']} (LOS probe tile ${tgt[0]:02x},${tgt[1]:02x})"
            )

        # --- ACTION KEY ---
        key = {"transfer": K_TRANSFER, "absorb": K_ABSORB}.get(verb)
        if verb == "create":
            key = {0: K_ROBOT, 2: K_TREE, 3: K_BOULDER}[otype]
        if key is None:
            log(f"[{i:2}] unknown verb {verb}; aborting")
            result["divergence"] = f"step {i} unknown verb {verb}"
            return False
        # create/absorb gate requires sights ON ($12D9); blind creates too.
        if verb in ("create", "absorb"):
            d.sights(True)
        d.tap(key, hold=16, settle=0.5)

        after = d.state()
        e1 = after.player_energy
        objs1 = len(objects_at(after, *tile))
        slot1 = after.player_slot

        # --- VERIFY the live delta matches the native prediction ---
        ok, msg = verify(
            verb, otype, tile, before, after, objs0, objs1, slot0, slot1, e0, e1
        )
        result["actions"].append(
            {
                "step": i,
                "verb": verb,
                "tile": list(tile),
                "otype": otype,
                "ok": ok,
                "msg": msg,
                "energy": [e0, e1],
                "aim": aim_info,
            }
        )
        log(f"[{i:2}] {verb:8} {tile} otype={otype}: {'OK ' if ok else 'FAIL'} {msg}")
        if not ok:
            result["divergence"] = f"step {i} {verb} {tile}: {msg}"
            grab(f"verifyfail_{i:02d}")
            return False
        if i % 4 == 0:
            grab(f"step{i:02d}")

    # ---- FINAL HYPERSPACE: the genuine win. The last plan step (transfer onto the
    # platform robot) put the player on the platform tile; H -> do_hyperspace $2156
    # sets $0CDE bit6 because player tile == platform tile $0C19/$0C1A. ------------
    p = d.state().player
    pcur = (p.x, p.y)
    done0 = d.rd(A_LANDSCAPE_DONE)
    log(
        f"-- FINAL HYPERSPACE (H) from player tile {pcur} platform {plat}; "
        f"$0CDE before=${done0:02x} --"
    )
    if pcur != plat:
        log(f"   WARNING: player tile {pcur} != platform {plat} before hyperspace")
    won = False
    for attempt in range(4):
        d.tap(K_HYPERSPACE, hold=20, settle=0.7)
        done1 = d.rd(A_LANDSCAPE_DONE)
        log(
            f"   H attempt {attempt}: $0CDE=${done1:02x} bit6={'SET' if done1 & 0x40 else 'clear'}"
        )
        grab(f"hyperspace_{attempt}")
        if done1 & 0x40:
            won = True
            break
    result["landscape_done"] = d.rd(A_LANDSCAPE_DONE)
    return won


# ROM energy deltas ($214F, masked AND #$3F): create pays the cost; absorb refunds it.
_CREATE_COST = {0: 3, 2: 1, 3: 2}
_ABSORB_REFUND = {0: 3, 1: 3, 2: 1, 3: 2, 5: 4}


def verify(verb, otype, tile, before, after, objs0, objs1, slot0, slot1, e0, e1):
    # D6: require the EXACT on-tile object delta AND energy delta; flag any other global
    # object-count change (wrong-tile, meanie, held-key extras) as divergence.
    dtot = len(after.objects) - len(before.objects)
    if verb == "create":
        if objs1 != objs0 + 1:
            return (
                False,
                f"create wrong-tile/none on {tile} (objs {objs0}->{objs1}); energy {e0}->{e1}",
            )
        if dtot != 1:
            return (
                False,
                f"create changed global object count by {dtot}; energy {e0}->{e1}",
            )
        exp = (e0 - _CREATE_COST.get(otype, 3)) & 0x3F
        if e1 != exp:
            return False, f"create energy {e0}->{e1} != expected {exp}"
        return (
            True,
            f"object created on {tile} (objs {objs0}->{objs1}); energy {e0}->{e1}",
        )
    if verb == "transfer":
        moved = (slot1 != slot0) or (
            after.player and (after.player.x, after.player.y) == tile
        )
        if moved:
            return (
                True,
                f"player_slot {slot0}->{slot1}, now at ({after.player.x},{after.player.y})",
            )
        return False, f"transfer did not move player (slot {slot0}->{slot1})"
    if verb == "absorb":
        if objs1 != objs0 - 1:
            return (
                False,
                f"absorb wrong-tile/none on {tile} (objs {objs0}->{objs1}); energy {e0}->{e1}",
            )
        if dtot != -1:
            return (
                False,
                f"absorb changed global object count by {dtot}; energy {e0}->{e1}",
            )
        exp = (e0 + _ABSORB_REFUND.get(otype, 3)) & 0x3F
        if e1 != exp:
            return False, f"absorb energy {e0}->{e1} != expected {exp}"
        return (
            True,
            f"object absorbed on {tile} (objs {objs0}->{objs1}); energy {e0}->{e1}",
        )
    return False, "?"


def validate_avi(path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landscape", type=int, default=42)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--max-seconds", type=int, default=600)
    args = ap.parse_args()
    plan_path = args.plan or os.path.join(
        ROOT, "out", f"kbd_snapped_{args.landscape:04d}.json"
    )

    def log(m):
        print(m, flush=True)

    log(f"=== VICE keyboard record: ls{args.landscape:04d} plan={plan_path} ===")
    result = run(args.landscape, plan_path, args.max_seconds, log)

    vid = result.get("video")
    if vid:
        ok, size, nfr, msg = validate_avi(vid)
        result["video_valid"] = ok
        result["video_size"] = size
        result["video_frames"] = nfr
        import subprocess

        ft = ""
        try:
            ft = subprocess.run(
                ["file", "-b", vid], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            pass
        log(
            f"AVI: {vid}\n   valid={ok} size={size}B ({size/1024:.1f} KiB) frames={nfr} ({msg})\n   file: {ft}"
        )

    done = result.get("landscape_done", 0)
    log("\n=== RESULT ===")
    log(f"  landscape    : {args.landscape:04d}")
    log(f"  $0CDE flag   : {done:#04x} (bit6 = landscape complete)")
    log(f"  WIN VERIFIED : {'PASS' if result.get('won') else 'FAIL'}")
    if result.get("divergence"):
        log(f"  DIVERGENCE   : {result['divergence']}")
    log(f"  wall seconds : {result.get('wall_seconds')}")
    log(
        f"  video        : {vid} valid={result.get('video_valid')} "
        f"{result.get('video_size',0)}B frames={result.get('video_frames',0)}"
    )
    return 0 if result.get("won") else 1


if __name__ == "__main__":
    sys.exit(main())
