#!/usr/bin/env python3
"""Keyboard-driven REAL-ROM replay of the ROM-validated ls42 (internal seed 66) win
inside asid-vice, recorded to a real AVI via the native ZMBV video opcode, with the
genuine ROM win flag $0CDE bit6 verified.

NO PIXELS: aim and verification are entirely from MEMORY reads. Aiming uses the
authentic keyboard sights-cursor path (kbd_aim.KbdDriver): drive the view angles
sights-off (S/D/L/COMMA + U-turn) then the sights cursor sights-on, closed-loop on
the live native-LOS probe (native_los on a RAM snapshot) until the target tile is hit
with LOS. The action is then a real keystroke (R/B robot/boulder, Q transfer, A absorb,
H hyperspace), fired via tap_action which polls the game's own $0CE9 action latch.
Every state-changing input is a real key; reads never change state. The AVI is captured
by the native video_record/video_stop opcode.

Plan: out/kbd_greedy_0066.json -- ROM-validated steps (count == len(steps)). The player
TYPES "0042" (BCD 0x42 = internal seed 66).
"""

import os, sys, time, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from vice_driver import BinMon, DiskMount, ViceContainer, keys
from vice_driver.binmon import TAP_MODE_FIXED

import game_state as gs
from vice_execute import Executor, CREATE_KEY, K_ABSORB, K_TRANSFER, K_HYPERSPACE
import kbd_aim
from native_los import NativeState, aim_target_native

TAP = os.path.join(ROOT, "sentinel-gold.tap")

A_PLAYER_SLOT = 0x000B
A_ENERGY = 0x0C0A
A_LANDSCAPE_DONE = 0x0CDE
A_PLAT_X = 0x0C19
A_PLAT_Y = 0x0C1A

# secret-code-check patches (accept any code)
PATCHES = [
    (0x14DF, bytes([0xA9, 0x1E])),
    (0x2565, bytes([0xEA, 0xEA])),
    (0x2570, bytes([0xEA, 0xEA])),
]


def _reconnect(bm, log):
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
    return fn()


def navigate(bm, typed_digits, log):
    """Boot under WARP to LANDSCAPE NUMBER, patch the code-checks, type the digits +
    dummy secret code, dismiss the preview, enter play."""

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
    for _ in range(50):
        time.sleep(1.0)
        robust(bm, log, lambda: bm.mem_get(0x00, 0x00))
    for _ in range(3):
        tap("SPACE", hold=30, settle=1.5)
    log("patching secret-code checks ($14DF/$2565/$2570)")
    for addr, data in PATCHES:
        robust(bm, log, lambda a=addr, d=data: bm.mem_set(a, d))
    log(f"typing landscape digits {typed_digits!r}")
    tap_text(typed_digits)
    tap("RETURN", hold=30, settle=3.0)
    tap_text("00000000")
    tap("RETURN", hold=30, settle=8.0)
    time.sleep(3)
    tap("SPACE", hold=25, settle=1.2)  # dismiss the isometric preview
    time.sleep(4)


def objects_at(state, x, y):
    return [o for o in state.objects if o.x == x and o.y == y]


def _bridge_ip(container_id, log):
    """Look up a started container's docker BRIDGE IP. Host -p port publishing is not
    reachable in this environment (127.0.0.1:6502 fails); the container's bridge IP is.
    """
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
        if out:
            return out
    except Exception as e:
        log(f"  bridge-ip lookup failed: {e}")
    return None


def _free_port_6502(log):
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


def verify_entry(bm, log):
    """Confirm the live state matches game_state.Py65Source.from_landscape(66) 16/16."""
    try:
        ref = gs.read_game_state(gs.Py65Source.from_landscape(66))
    except Exception as e:
        log(f"  (entry ref unavailable: {e})")
        return None
    live = gs.read_game_state(gs.ViceSource(bm))
    ref_objs = sorted((o.x, o.y, o.type) for o in ref.objects)
    live_objs = sorted((o.x, o.y, o.type) for o in live.objects)
    matched = sum(1 for o in ref_objs if o in live_objs)
    log(
        f"ENTRY MATCH: {matched}/{len(ref_objs)} objects vs from_landscape(66) "
        f"(live has {len(live_objs)})"
    )
    return matched, len(ref_objs)


def run(typed_digits, plan_path, max_seconds, log):
    if not os.path.exists(TAP):
        raise FileNotFoundError(
            f"{TAP} missing: place the game tape image there (not distributed)"
        )
    with open(plan_path) as f:
        plan = json.load(f)
    if not plan.get("won"):
        log(f"REFUSING: plan {plan_path} is not a win (won={plan.get('won')})")
        return {"won": False, "note": "plan not validated"}
    steps = plan["steps"]
    log(f"loaded {len(steps)} steps from {plan_path}")

    renders_host = os.path.join(ROOT, "renders")
    os.makedirs(renders_host, exist_ok=True)
    video_host = os.path.join(renders_host, "solver_run_0042.avi")
    if os.path.exists(video_host):
        try:
            os.remove(video_host)
        except OSError:
            pass

    result = {
        "won": False,
        "video": video_host,
        "actions": [],
        "divergence": None,
        "energy_curve": [],
    }
    BOOT_TRIES = 8
    for boot_try in range(BOOT_TRIES):
        _free_port_6502(log)
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
                # Host -p publishing is broken here (127.0.0.1:6502 unreachable); connect
                # via the started container's docker bridge IP. BINMON_HOST env overrides.
                time.sleep(2)
                bm_host = os.environ.get("BINMON_HOST") or _bridge_ip(
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

                navigate(bm, typed_digits, log)
                st = gs.read_game_state(gs.ViceSource(bm))
                if st.player is None:
                    log(f"boot try {boot_try}: not in play (no player); restart")
                    continue
                result["entry_match"] = verify_entry(bm, log)
                ex = Executor(bm, log)
                plat = (ex.rd(A_PLAT_X), ex.rd(A_PLAT_Y))
                log(
                    f"IN PLAY: slot {st.player_slot} @ ({st.player.x},{st.player.y}) "
                    f"energy {st.player_energy} objs {len(st.objects)} platform {plat}"
                )

                record = os.environ.get("NO_RECORD") != "1"
                if record:
                    log(f"-- starting AVI recording -> {video_host} --")
                    try:
                        bm.video_record("/renders/solver_run_0042.avi")
                    except Exception as e:
                        log(f"  video_record failed: {e}")
                    time.sleep(1.0)
                else:
                    log("-- NO_RECORD=1: skipping AVI (warp stays on) --")

                won = execute(ex, steps, plat, log, result, t_start, max_seconds)
                result["won"] = won
                time.sleep(1.5)

                log("-- stopping AVI recording (finalize) --")
                try:
                    if record:
                        bm.video_stop()
                    time.sleep(1.5)
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
            if result["actions"]:
                result["divergence"] = result.get("divergence") or f"mid-run drop: {e}"
                return result
            time.sleep(2)
    result["divergence"] = f"could not boot into play after {BOOT_TRIES} tries"
    return result


def execute(ex, steps, plat, log, result, t_start, max_seconds):
    """Walk the ROM-validated steps with the authentic keyboard sights-cursor aim
    (kbd_aim.KbdDriver) + real action keys, verifying each step's memory delta and
    watching the energy curve."""
    drv = kbd_aim.KbdDriver(ex.bm, log)

    def _probe_once():
        m = bytearray(ex.bm.mem_get(0x0000, 0x0FFF))
        ps = m[0x000B]
        st = NativeState.from_mem(bytes(m))
        rx, ry, los, centre = aim_target_native(
            st,
            m[0x09C0 + ps],
            m[0x0140 + ps],
            m[0x0CC6],
            m[0x0CC7],
            ps,
            eye_z=m[0x0940 + ps],
            max_steps=4000,
            return_centre=True,
        )
        sig = (m[0x0CE4] & 0x80, m[0x09C0 + ps], m[0x0140 + ps], m[0x0CC6], m[0x0CC7])
        return (rx, ry, los, centre), sig

    def probe_tile():
        """Where the live sights ray lands now (native_los on a cheap RAM snapshot).
        Hardened (D2): only accept a snapshot when $0CE4 bit7 is clear AND h/v/cursor are
        identical across two consecutive reads (reject transient mid-pan / queued-wrap
        state), else wait 50ms and retry. Returns (rx, ry, los, centre)."""
        res, prev = _probe_once()
        for _ in range(8):
            if prev[0] == 0:
                res2, sig2 = _probe_once()
                if sig2 == prev:
                    return res2
                res, prev = res2, sig2
            else:
                time.sleep(0.05)
                res, prev = _probe_once()
        return res

    for i, stp in enumerate(steps):
        if time.time() - t_start > max_seconds:
            log(f"TIME BUDGET ({max_seconds}s) exceeded at step {i}; aborting")
            result["divergence"] = f"timeout at step {i}"
            return False
        verb, tile, otype = stp["verb"], tuple(stp["target"]), stp["otype"]
        _tx, _ty = tile
        plan_view = stp.get("view")

        before = ex.state()
        e0 = before.player_energy
        objs0 = len(objects_at(before, *tile))
        slot0 = before.player_slot
        result["energy_curve"].append({"step": i, "verb": verb, "energy_before": e0})

        # D7: proactive drain watch. The plan carries the ROM-validated expected energy
        # AFTER each step ("plan_energy"); if the live budget before a CREATE has already
        # fallen below its cost, enemies drained us during aiming -- flag it explicitly
        # rather than letting the create silently energy-block.
        if verb == "create" and e0 < otype_cost(otype):
            log(
                f"[{i:2}] create {tile}: DRAINED -- energy {e0} < cost {otype_cost(otype)} "
                f"(plan expected ~{stp.get('plan_energy')})"
            )
            result["energy_block"] = {
                "step": i,
                "tile": list(tile),
                "otype": otype,
                "energy_before": e0,
                "plan_energy": stp.get("plan_energy"),
            }
            result["divergence"] = f"step {i} create {tile}: drained (energy {e0})"
            return False

        # --- KEYBOARD AIM: DRIVE the persisted view. Each create/absorb step carries a
        # ROM-validated, keyboard-lattice view (h%8==0, v%4==1 in the pan band) computed by
        # inverting the aim transform (scripts/aim_invert.py). We do NOT recompute via
        # snap_keyboard_view (broken/slow); we drive the real keys to the persisted angles
        # sights-off, then the persisted cursor sights-on, and CONFIRM the live LOS ray.
        # blind synthoid-creates (view=null) build on the boulder we already aimed at;
        # transfers need no aim. ---
        aim_info = None
        if verb in ("create", "absorb") and plan_view is not None:
            view = plan_view
            if not drv.sights_set(False):
                log(f"[{i:2}] {verb} {tile}: sights would not turn OFF")
                result["divergence"] = f"step {i} sights off failed"
                return False
            okh = drv.coarse_h(view["h_angle"])
            okv = drv.coarse_v(view["v_angle"])
            if not (okh and okv):
                okh = drv.coarse_h(view["h_angle"])
                okv = drv.coarse_v(view["v_angle"])
            if not drv.sights_on():
                log(f"[{i:2}] {verb} {tile}: sights would not turn ON")
                result["divergence"] = f"step {i} sights on failed"
                return False
            okc = drv.fine_cursor(
                *view["cursor"]
            )  # sights-on re-centred it; drive persisted
            rx, ry, los, centre = probe_tile()
            ach = {"h": drv.hang(), "v": drv.vang(), "cur": drv.cur()}
            aim_info = {
                "ach": ach,
                "want": view,
                "probe": (rx, ry, los, centre),
                "ok": {"h": okh, "v": okv, "cur": okc},
            }
            log(
                f"[{i:2}] {verb} {tile}: drove view h=${view['h_angle']:02x} "
                f"v=${view['v_angle']:02x} cur={view['cursor']} -> ach h=${ach['h']:02x} "
                f"v=${ach['v']:02x} cur={ach['cur']} probe=({rx},{ry}) los={los} "
                f"centre=${centre:02x}"
            )
            # The native_los probe is ADVISORY only: the arbiter is the real ROM's
            # object-count/energy delta (verify() below). The persisted view is
            # ROM-validated; drive it and let the game decide. Only note a probe miss.
            drove_ok = (
                drv.hang() == view["h_angle"]
                and drv.vang() == view["v_angle"]
                and drv.cur() == tuple(view["cursor"])
            )
            if (rx, ry) != tile or not los or not drove_ok:
                log(
                    f"[{i:2}] {verb} {tile}: (advisory) probe ({rx},{ry}) los={los} "
                    f"drove_ok={drove_ok}; firing anyway, verify() decides"
                )

        # --- ACTION KEY (deterministic, scan-consumed) ---
        if verb == "create":
            key = CREATE_KEY[otype]
        elif verb == "transfer":
            key = K_TRANSFER
        elif verb == "absorb":
            key = K_ABSORB
        else:
            log(f"[{i:2}] unknown verb {verb}; abort")
            result["divergence"] = f"step {i} unknown verb {verb}"
            return False
        # consider_player_action ($12D9) requires sights active for create/absorb AND
        # transfer. Fire the key EXACTLY ONCE (tap_action is single-fire; NEVER re-fire on a
        # false-negative latch -- a second create/absorb would stack an extra object). The
        # object-count/energy/slot delta in verify() is the real arbiter of success.
        if verb in ("create", "absorb", "transfer"):
            drv.sights_on()
        latched = drv.tap_action(key)
        if not latched:
            log(
                f"[{i:2}] {verb} {tile}: action key {key} latch not observed; verify() decides"
            )

        after = ex.state()
        e1 = after.player_energy
        objs1 = len(objects_at(after, *tile))
        slot1 = after.player_slot

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
            # BEST-EFFORT ABSORBS: trail/fuel absorbs (otype != 5, the Sentinel) are energy
            # recovery, exactly as the ROM validator treats them -- a miss is NOT fatal. The
            # plan carries an energy margin; if a missed refund later starves a build, that
            # CREATE will energy-block below and abort with a clear message. Skip and go on.
            if verb == "absorb" and otype != 5:
                log(
                    f"    (best-effort absorb miss at step {i}; continuing -- "
                    f"energy {e1})"
                )
                continue
            # If a CREATE failed with energy at/near 0, flag it as an energy-budget block.
            if verb == "create" and e0 <= otype_cost(otype):
                result["energy_block"] = {
                    "step": i,
                    "tile": list(tile),
                    "otype": otype,
                    "energy_before": e0,
                }
                log(
                    f"    >>> ENERGY BLOCK: build at step {i} needs more energy than "
                    f"the {e0} available"
                )
            result["divergence"] = f"step {i} {verb} {tile}: {msg}"
            return False

    # ---- FINAL HYPERSPACE ----
    p = ex.state().player
    pcur = (p.x, p.y)
    done0 = ex.rd(A_LANDSCAPE_DONE)
    log(f"-- FINAL HYPERSPACE (H) from {pcur} platform {plat}; $0CDE=${done0:02x} --")
    if pcur != plat:
        log(f"   WARNING: player tile {pcur} != platform {plat}")
    won = False
    for attempt in range(4):
        drv.tap_action(K_HYPERSPACE)
        done1 = ex.rd(A_LANDSCAPE_DONE)
        log(
            f"   H attempt {attempt}: $0CDE=${done1:02x} "
            f"bit6={'SET' if done1 & 0x40 else 'clear'}"
        )
        if done1 & 0x40:
            won = True
            break
    result["landscape_done"] = ex.rd(A_LANDSCAPE_DONE)
    return won


def otype_cost(otype):
    # ROM object energy costs: robot/synthoid 3, tree 1, boulder 2.
    return _CREATE_COST.get(otype, 3)


# ROM energy deltas ($214F, masked AND #$3F): create pays the cost; absorb refunds it.
_CREATE_COST = {0: 3, 2: 1, 3: 2}
_ABSORB_REFUND = {0: 3, 1: 3, 2: 1, 3: 2, 5: 4}


def verify(verb, otype, tile, before, after, objs0, objs1, slot0, slot1, e0, e1):
    # D6: require the EXACT on-tile object delta and the EXACT energy delta, and flag any
    # other global object-count change (wrong-tile landing, meanie spawn, held-key extra
    # creates) as divergence -- the loose "grew/gone OR global-count" test hid all three.
    dtot = len(after.objects) - len(before.objects)
    if verb == "create":
        if objs1 != objs0 + 1:
            return (
                False,
                f"create wrong-tile/none on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}",
            )
        if dtot != 1:
            return (
                False,
                f"create changed global object count by {dtot} (meanie/extra?); E {e0}->{e1}",
            )
        exp = (e0 - _CREATE_COST.get(otype, 3)) & 0x3F
        if e1 != exp:
            return (
                False,
                f"create energy {e0}->{e1} != expected {exp} (cost {_CREATE_COST.get(otype, 3)})",
            )
        return True, f"object created on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}"
    if verb == "transfer":
        moved = (slot1 != slot0) or (
            after.player and (after.player.x, after.player.y) == tile
        )
        if moved:
            return (
                True,
                f"slot {slot0}->{slot1}, now ({after.player.x},{after.player.y})",
            )
        return False, f"transfer did not move player (slot {slot0}->{slot1})"
    if verb == "absorb":
        if objs1 != objs0 - 1:
            return (
                False,
                f"absorb wrong-tile/none on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}",
            )
        if dtot != -1:
            return False, f"absorb changed global object count by {dtot}; E {e0}->{e1}"
        exp = (e0 + _ABSORB_REFUND.get(otype, 3)) & 0x3F
        if e1 != exp:
            return (
                False,
                f"absorb energy {e0}->{e1} != expected {exp} (refund {_ABSORB_REFUND.get(otype, 3)})",
            )
        return True, f"object absorbed on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}"
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
    ap.add_argument("--digits", default="0042")
    ap.add_argument("--plan", default=os.path.join(ROOT, "out", "kbd_greedy_0066.json"))
    ap.add_argument("--max-seconds", type=int, default=1500)
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    log(f"=== VICE keyboard record: type {args.digits!r} plan={args.plan} ===")
    result = run(args.digits, args.plan, args.max_seconds, log)

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
            f"AVI: {vid}\n   valid={ok} size={size}B ({size/1024:.1f} KiB) "
            f"frames={nfr} ({msg})\n   file: {ft}"
        )

    done = result.get("landscape_done", 0)
    n_ok = sum(1 for a in result.get("actions", []) if a["ok"])
    log("\n=== RESULT ===")
    log(f"  entry match  : {result.get('entry_match')}")
    log(f"  steps OK     : {n_ok}/{len(result.get('actions', []))} keyboard steps")
    if result.get("energy_block"):
        log(f"  ENERGY BLOCK : {result['energy_block']}")
    log(f"  $0CDE flag   : {done:#04x} (bit6 = landscape complete)")
    log(f"  WIN VERIFIED : {'PASS' if result.get('won') else 'FAIL'}")
    if result.get("divergence"):
        log(f"  DIVERGENCE   : {result['divergence']}")
    log(f"  wall seconds : {result.get('wall_seconds')}")
    log(
        f"  video        : {vid} valid={result.get('video_valid')} "
        f"{result.get('video_size',0)}B frames={result.get('video_frames',0)}"
    )
    # dump the energy curve for the report
    ec = result.get("energy_curve", [])
    if ec:
        log(
            "  energy curve : "
            + " ".join(f"{e['step']}:{e['energy_before']}" for e in ec)
        )
    return 0 if result.get("won") else 1


if __name__ == "__main__":
    sys.exit(main())
