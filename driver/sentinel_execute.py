#!/usr/bin/env python3
"""Live-execution constants and a thin Executor (key mapping, memory reads, live
GameState over BinMon) used by the plan runner."""

from driver import sentinel_state as gs
from driver import clock, core, kbd_prog
from driver.kbd_prog import V_OKH, V_OKV
from sentinel import memmap as mm
from sentinel import aim
from sentinel.state import State
from sentinel.memmap import T_BOULDER, T_ROBOT, T_TREE

# action keys (vice_driver.keys names), decoded from the game's key-number table
# $138D + action-code table $139C and confirmed live:
#   A = absorb ($20), Q = transfer ($21), R = create robot ($00),
#   T = create tree ($02), B = create boulder ($03), H = hyperspace ($22).
K_ABSORB = "A"
K_TRANSFER = "Q"
K_CREATE_ROBOT = "R"
K_CREATE_TREE = "T"
K_CREATE_BOULDER = "B"
K_HYPERSPACE = "H"

CREATE_KEY = {
    T_ROBOT: K_CREATE_ROBOT,
    T_TREE: K_CREATE_TREE,
    T_BOULDER: K_CREATE_BOULDER,
}
_VERB_KEY = {"transfer": K_TRANSFER, "absorb": K_ABSORB}


def otype_cost(otype):
    """The ROM energy a create of ``otype`` spends / an absorb of it refunds
    (energy_in_objects $214F, via sentinel.memmap)."""
    return mm.ENERGY_IN_OBJECTS.get(otype, 3)


def verify(verb, otype, tile, before, after, objs0, objs1, slot0, slot1, e0, e1):
    """Arbitrate whether a fired action really did what the plan step intended, from
    the live memory delta: the EXACT on-tile object-count change AND the EXACT energy
    delta, flagging any other global object-count change (wrong-tile landing, meanie
    spawn, held-key extra creates) as a divergence. Returns ``(ok, message)``."""
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
        exp = (e0 - otype_cost(otype)) & 0x3F
        if e1 != exp:
            return (
                False,
                f"create energy {e0}->{e1} != expected {exp} (cost {otype_cost(otype)})",
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
        exp = (e0 + otype_cost(otype)) & 0x3F
        if e1 != exp:
            return (
                False,
                f"absorb energy {e0}->{e1} != expected {exp} (refund {otype_cost(otype)})",
            )
        return True, f"object absorbed on {tile} (objs {objs0}->{objs1}); E {e0}->{e1}"
    return False, "?"


def classify_outcome(verb, otype, ok, primary_ok):
    """Map a fired step's verify() result to a caller outcome, priority-ordered.

    ``primary_ok`` (the on-tile object created/absorbed, or the player moved) is
    checked BEFORE the best-effort-absorb-miss shortcut: a tree/fuel absorb whose
    object WAS removed but coincided with a Sentinel discharge (a fresh tree at a
    random tile) nets the global object delta to 0, failing verify()'s dtot check.
    That is a world-divergence to resync, not a miss to retry -- re-firing an absorb
    would aim at a tile whose object is already gone. Only a genuine miss (object not
    removed) reaches the ``best_effort_miss`` fuel-recovery path.
    """
    if ok:
        return "ok"
    if primary_ok:
        return "diverge"
    if verb == "absorb" and otype != 5:  # Sentinel is otype 5; its absorb is not fuel
        return "best_effort_miss"
    return "fail"


class Executor:
    def __init__(self, bm, log):
        self.bm = bm
        self.log = log

    def rd(self, a):
        return self.bm.mem_get(a, a)[0]

    def frames(self):
        """Exact, wrap-free elapsed-frame count ($9630 checkpoint hit_count); delta two
        calls to time a span exactly."""
        return clock.frames(self.bm)

    def state(self):
        return gs.read_game_state(gs.ViceSource(self.bm))

    def platform(self):
        """The Sentinel's platform tile (x, y)."""
        return (self.rd(mm.PLATFORM_X), self.rd(mm.PLATFORM_Y))

    def landscape_done(self):
        """The raw landscape-complete byte ($0CDE); bit6 is the win flag."""
        return self.rd(mm.LANDSCAPE_COMPLETE)

    def won(self):
        """Whether the landscape is complete ($0CDE bit6 set)."""
        return bool(self.landscape_done() & 0x40)


def _live_centre_view(bm, tile, log, label, verb):
    """Resolve a deferred aim (view None) against CURRENT live memory via the SHARED
    aim proposer (``sentinel.aim.propose`` at the player's TRUE eye) on a RAM
    snapshot, so callers never diverge on how a tile is aimed. Returns the view dict
    (cursor as a list) or None when no keyboard aim lands on ``tile``."""
    mem = core.live_image(bm)
    view = aim.propose(State.from_mem(mem), tile, eye_z=None)
    if view is None:
        log(
            f"[{label}] {verb} {tile}: no live keyboard view (no LOS); "
            "firing blind, verify() decides"
        )
    return view


def perform_step(ex, drv, label, stp, log, result):
    """Fire ONE plan step (verb/otype/target/view) against the live game via a real
    keystroke, verify the memory delta, and report the outcome. Used by the live
    replanning loop. Returns one of:
      "ok"               -- verified success
      "diverge"          -- on-tile effect landed but the world moved (resync + replan)
      "best_effort_miss" -- a non-Sentinel absorb missed (fuel recovery; non-fatal)
      "drained"          -- energy already below a create's cost before firing (no keys sent)
      "aim_miss"         -- the aim never reached the requested view (nothing fired)
      "fail"             -- verify() rejected the step (wrong-tile/count/energy delta)
    """
    verb, tile, otype = stp["verb"], tuple(stp["target"]), stp["otype"]
    plan_view = stp.get("view")

    before = ex.state()
    e0 = before.player_energy
    objs0 = len(before.objects_at(*tile))
    slot0 = before.player_slot
    result["energy_curve"].append({"step": label, "verb": verb, "energy_before": e0})

    # D7: proactive drain watch -- if the live budget before a CREATE has already
    # fallen below its cost, enemies drained us during aiming; flag it explicitly
    # rather than letting the create silently energy-block.
    if verb == "create" and e0 < otype_cost(otype):
        log(
            f"[{label}] create {tile}: DRAINED -- energy {e0} < cost {otype_cost(otype)}"
        )
        result["energy_block"] = {
            "step": label,
            "tile": list(tile),
            "otype": otype,
            "energy_before": e0,
        }
        return "drained"

    # A view of None means the aim was deferred (an on-boulder synthoid re-aims after
    # the boulder just landed; an absorb whose coarse candidate sweep didn't resolve
    # one). Resolve a live centre-aimed view here (sentinel.los against CURRENT
    # memory, e.g. post-boulder) before firing.
    if verb in ("create", "absorb") and plan_view is None:
        plan_view = _live_centre_view(ex.bm, tile, log, label, verb)

    # --- KEYBOARD AIM: DRIVE the given view. Create/absorb steps carry a keyboard-
    # lattice view (h%8==0, v%4==1 in the pan band); drive the real keys to those
    # angles sights-off, then the cursor sights-on, and CONFIRM the live LOS ray. ---
    key = CREATE_KEY[otype] if verb == "create" else _VERB_KEY.get(verb)
    min_energy = stp.get("min_energy") if verb == "create" else None
    fire = None if key is None else (key, min_energy)
    fired = None
    aim_info = None
    if verb in ("create", "absorb") and plan_view is not None:
        view = plan_view
        # AIM is pre-action and idempotent (drives the cursor to ABSOLUTE angles, reads
        # probes -- no game action fires). A best-effort fuel absorb at an extreme angle
        # can leave the monitor checkpoint desynced (CPU stopped / PC never recurs), and
        # the raw driver calls do not catch that -- an unhandled socket TimeoutError would
        # otherwise crash the whole run into the boot-retry. Reconnect + re-aim instead.
        okh = okv = okc = False
        probe = (0, 0, False, 0)
        ach = {"h": 0, "v": 0, "cur": (0, 0)}
        want_bearing = (view["h_angle"], view["v_angle"])
        # REUSE the live aim across consecutive same-bearing steps (e.g. a stacked build).
        # A create/absorb leaves sights ON and the bearing untouched -- SPACE is the only
        # sights toggle ($11B3) and the action keys never call initialise_sights -- so when
        # the committed bearing already equals this view's, keep sights ON and drive ONLY
        # the cursor. That skips the sights OFF->ON toggle whose initialise_sights ($134C)
        # recenters the cursor to ($50,$5F) and forces a full re-drive down, holding the
        # player exposed while enemies rotate. The native-LOS probe GATES the fast path: it
        # fires only if the live ray still lands on `tile`, else it falls back to a full
        # re-aim -- so reuse can never fire on the wrong tile.
        reuse = drv.sights_live_on() and drv.committed_bearing() == want_bearing

        def _aim_dropped(e):
            nonlocal reuse
            reuse = False  # a dropped pass may leave the bearing/cursor half-driven
            drv.clear_bearing()
            log(
                f"[{label}] {verb} {tile}: aim monitor drop "
                f"({type(e).__name__}); reconnecting + re-aiming"
            )

        for _aim_try in range(3):
            with core.drop_guard(ex.bm, log, _aim_dropped):
                if reuse:
                    okc = drv.cursor_probe(*view["cursor"])
                    probe = drv.probe_ray()
                    if okc and probe[:2] == tile and probe[2]:
                        okh = okv = True
                        ach = {
                            "h": want_bearing[0],
                            "v": want_bearing[1],
                            "cur": drv.cur(),
                        }
                        break
                    reuse = False  # cursor-only did not land -> full re-aim this pass
                # Compiler.aim reads h/v sights-OFF ($10B7 leaves them off-lattice on).
                merged = None if drv.advisory else fire
                got = drv.step(kbd_prog.view_key(view), merged)
                if got.halt == kbd_prog.H_FAIL_OFF:
                    log(f"[{label}] {verb} {tile}: sights would not turn OFF")
                    return "fail"
                okh, okv = (kbd_prog.STATUS[got[k]] for k in (V_OKH, V_OKV))
                if got.halt == kbd_prog.H_HYPERSPACE:
                    log(f"[{label}] {verb} {tile}: HYPERSPACED mid-aim; aborting aim")
                    return "aim_hyperspace"
                if got.halt == kbd_prog.H_FAIL_ON:
                    log(f"[{label}] {verb} {tile}: sights would not turn ON")
                    return "fail"
                okc = bool(got[kbd_prog.V_OKC])
                probe = drv.probe_ray() if drv.advisory else None
                ach = {
                    "h": got[kbd_prog.V_ACH_H],
                    "v": got[kbd_prog.V_ACH_V],
                    "cur": (got[kbd_prog.V_CX], got[kbd_prog.V_CY]),
                }
                fired = got if merged else None
                break
        else:
            log(f"[{label}] {verb} {tile}: aim never stabilised; skipping step")
            return "fail"
        aim_info = {
            "ach": ach,
            "want": view,
            "probe": probe,
            "ok": {"h": okh, "v": okv, "cur": okc},
        }
        if probe is not None:
            log(
                f"[{label}] {verb} {tile}: drove view h=${view['h_angle']:02x} "
                f"v=${view['v_angle']:02x} cur={view['cursor']} -> ach h=${ach['h']:02x} "
                f"v=${ach['v']:02x} cur={ach['cur']} probe=({probe[0]},{probe[1]}) "
                f"los={probe[2]} centre=${probe[3]:02x}"
            )
        # The sentinel.los probe is ADVISORY only: the arbiter is the real ROM's
        # object-count/energy delta (verify() below). Drive the view and let the game
        # decide; only note a probe miss. drove_ok comes from the already-read ach (no
        # extra monitor round-trips -- fewer socket ops = fewer flaky-idle drops).
        drove_ok = (
            ach["h"] == view["h_angle"]
            and ach["v"] == view["v_angle"]
            and ach["cur"] == tuple(view["cursor"])
        )
        # GUARD: never fire when the aim did NOT reach the requested view. The angles read
        # back (ach) not matching the request means a pan clamped or could not converge, so
        # the sights are pointing somewhere other than `tile` -- firing here is exactly the
        # "acted on the wrong tile" failure that drained energy and desynced the model. Skip
        # the action and report a miss so the loop resyncs and re-plans (re-snaps a view, or
        # picks a different foothold) instead. The native-LOS probe stays ADVISORY: with the
        # angles correct it can still disagree with the ROM, so a probe-only mismatch is NOT
        # a reason to withhold the action (that would skip valid actions forever).
        if not drove_ok:
            drv.clear_bearing()  # aim did not converge -> bearing is unknown for reuse
            log(
                f"[{label}] {verb} {tile}: aim did NOT reach view (want h=${view['h_angle']:02x} "
                f"v=${view['v_angle']:02x} cur={view['cursor']}, got h=${ach['h']:02x} "
                f"v=${ach['v']:02x} cur={ach['cur']}); NOT firing -- resync + re-plan"
            )
            return "aim_miss"
        drv.set_bearing(
            ach["h"], ach["v"]
        )  # committed: a same-bearing next step can reuse
        if probe is not None and (probe[:2] != tile or not probe[2]):
            log(
                f"[{label}] {verb} {tile}: (advisory) probe ({probe[0]},{probe[1]}) "
                f"los={probe[2]} but angles reached; firing, verify() decides"
            )

    if key is None:
        log(f"[{label}] unknown verb {verb}; abort")
        return "fail"
    # Compiler.fire: budget gate, sights on ($12D9), ONE tap; verify() arbitrates.
    if fired is None:

        def _fire_dropped(e):
            log(f"[{label}] {verb} {tile}: fire monitor drop ({type(e).__name__})")

        for _s_try in range(3):
            with core.drop_guard(ex.bm, log, _fire_dropped):
                fired = drv.step(None, fire)
                break
        else:
            return "fail"
    if fired.halt == kbd_prog.H_DRAINED:
        log(
            f"[{label}] create {tile}: DRAINED mid-aim -- energy "
            f"{fired[kbd_prog.V_ENERGY]} < floor {min_energy}; not firing"
        )
        return "drained"
    settle_frames = fired[kbd_prog.V_F1] - fired[kbd_prog.V_F0]
    latched = bool(fired[kbd_prog.V_LATCHED])
    result.setdefault("settle_audit", []).append([label, verb, settle_frames])
    if not latched:
        log(
            f"[{label}] {verb} {tile}: action key {key} latch not observed; verify() decides"
        )

    after = ex.state()
    e1 = after.player_energy
    objs1 = len(after.objects_at(*tile))
    slot1 = after.player_slot
    if (
        slot1 != slot0
    ):  # slot changed (transfer): the per-slot committed bearing is stale
        drv.clear_bearing()

    ok, msg = verify(
        verb, otype, tile, before, after, objs0, objs1, slot0, slot1, e0, e1
    )
    result["actions"].append(
        {
            "step": label,
            "verb": verb,
            "tile": list(tile),
            "otype": otype,
            "ok": ok,
            "msg": msg,
            "energy": [e0, e1],
            "aim": aim_info,
        }
    )
    log(f"[{label}] {verb:8} {tile} otype={otype}: {'OK ' if ok else 'FAIL'} {msg}")
    primary_ok = (
        (verb == "create" and objs1 == objs0 + 1)
        or (verb == "absorb" and objs1 == objs0 - 1)
        or (
            verb == "transfer"
            and (
                slot1 != slot0
                or bool(after.player and (after.player.x, after.player.y) == tile)
            )
        )
    )
    outcome = classify_outcome(verb, otype, ok, primary_ok)
    if outcome == "diverge":
        log(
            f"    (world-divergence at {label}: aim landed on {tile}, "
            f"state diverged [{msg}]; resync + replan)"
        )
    elif outcome == "best_effort_miss":
        log(f"    (best-effort absorb miss at {label}; continuing -- energy {e1})")
    elif outcome == "fail" and verb == "create" and e0 <= otype_cost(otype):
        result["energy_block"] = {
            "step": label,
            "tile": list(tile),
            "otype": otype,
            "energy_before": e0,
        }
        log(
            f"    >>> ENERGY BLOCK: build at {label} needs more energy than the {e0} available"
        )
    return outcome


def fire_hyperspace(ex, drv, plat, log, result):
    """Final hyperspace attempt from the platform tile; verified by the ROM's own
    landscape-complete flag ($0CDE bit6)."""
    p = ex.state().player
    pcur = (p.x, p.y)
    done0 = ex.landscape_done()
    log(f"-- FINAL HYPERSPACE (H) from {pcur} platform {plat}; $0CDE=${done0:02x} --")
    if pcur != plat:
        log(f"   WARNING: player tile {pcur} != platform {plat}")
    won = False
    for attempt in range(4):
        done1, stopped = drv.hyperspace_attempt()
        if stopped:
            log(f"   H {attempt}: frame marker stopped (BinmonError)")
        log(
            f"   H attempt {attempt}: $0CDE=${done1:02x} "
            f"bit6={'SET' if done1 & 0x40 else 'clear'}"
        )
        if done1 & 0x40:
            won = True
            break
    result["landscape_done"] = ex.landscape_done()
    return won
