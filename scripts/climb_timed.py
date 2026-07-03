#!/usr/bin/env python3
"""Time-accurate (enemy-rotation-aware) keyboard-win planner for The Sentinel.

This is the additive, TIMED successor to `native_game.plan`: it keeps the exact
same native climb mechanics (centre-aim, stack-only-on-boulder/platform, eye-
strictly-above-to-absorb, hyperspace-from-platform win) but adds the TIME
DIMENSION so the player is never caught in a rotating enemy's sightline (or a
meanie-spawn trigger) while it dwells building / transferring.

It returns a plan in EXACTLY the step format `validate_kbd_plan` consumes, so the
timed plan can be replayed through the real ROM to the genuine `$0CDE` win flag.

----------------------------------------------------------------------------
DESIGN (all native -- no py65 in the search loop)
----------------------------------------------------------------------------
* Search state: (tile, z_height, tick).  The enemy phase is a DETERMINISTIC
  function of tick (enemies only rotate on a fixed cadence -- enemy_dynamics
  step_enemies), so we cache phase-by-tick and never store it in the key.  This
  keeps the A* state small while still being fully time-resolved.

* Action tick-cost: each player action (build a stepping-stone boulder, hop a
  synthoid, transfer up, absorb) advances the enemy phase by TICKS_PER_ACTION
  game ticks.  We reuse the value the project's own minimax solver settled on
  (solver_exact.TICKS_PER_ACTION = 8; see the "Action-lattice bounds" note in
  `solver_exact.py`).
  A real keyboard create/transfer/absorb spans several game rounds; the exact
  count is executor-timing-dependent and NOT pinned by the ROM, so this is a
  documented fixed estimate (same one the rest of the project uses).

* Safety gate (the MIN layer / minimax framing): a move is allowed
  only if the player's DWELL tile is not seen by ANY enemy, treated WORST-CASE
  over a +/- ENEMY_PHASE_BAND (=4) tick band (enemy_dynamics.exposed_within_window),
  AND the dwell tile is not a meanie-spawn trigger (enemy_dynamics.meanie_safe).
  Both the tile we build/transfer FROM (we dwell there during the create+transfer)
  and the tile we land ON are gated.  enemy_sees uses the bit-exact terrain LOS +
  the time-accurate angular FOV (rotation step +-$14 = +-28.125 deg, FOV +-$0A =
  +-14.0625 deg; addresses in enemy_dynamics.py).

* WAIT moves: when no safe foothold is reachable NOW, the A* may advance `tick`
  in place (WAIT_TICKS per wait) to let the enemies rotate away.  This is the
  look-ahead that makes the safe next foothold non-local/timed.

The enemy queries take a `game_state.GameState`; terrain is static during a solve
(boulders we build are local footholds, not enemies), and enemies never move
(they only rotate), so ONE GameState built from the initial RAM serves every
enemy query.  The dwell-tile coordinates are all that vary.
"""

import sys, os, json, time, heapq

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, ".."))

import native_game as ng
from native_game import Game, terrain_z, visibility_sweep, cheb
import game_state as gs
import enemy_dynamics as ed

# --- timing model (cited from the project's established minimax solver) -------
# solver_exact.TICKS_PER_ACTION (8) and solver_exact.ENEMY_PHASE_BAND (4); see the
# "Action-lattice bounds" / "Minimax structure" notes in `solver_exact.py`.  A WAIT
# advances the phase one band-width so a single wait can change which enemies see
# a tile.
TICKS_PER_ACTION = 8
ENEMY_PHASE_BAND = 4
WAIT_TICKS = 8  # one wait == one action-worth of enemy rotation
MAX_WAITS = 24  # cap consecutive look-ahead waits (finiteness)


class TimedClimb:
    """Wraps a native_game.Game with the time-accurate enemy phase machinery.

    Terrain + enemies are read ONCE into a GameState (enemies only rotate; the
    terrain height field is constant during the climb), so every per-tick enemy
    query reuses it.  Phase-by-tick is cached: phase(tick) is deterministic."""

    def __init__(self, g: Game):
        self.g = g
        self.state = gs.read_game_state(gs.Py65Source(bytes(g.mem)))
        self.phase0 = ed.init_phase_from_ram(self.state, g.mem)
        self._phase_cache = {0: self.phase0}
        # safety cache keyed on (tile, tick) -- the worst-case exposure verdict.
        self._safe_cache = {}

    # ---- phase as a pure function of tick (cached, monotone) ----
    def phase_at(self, tick: int):
        if tick in self._phase_cache:
            return self._phase_cache[tick]
        # find the highest cached tick <= tick and step forward
        base = max(t for t in self._phase_cache if t <= tick)
        ph = self._phase_cache[base]
        for t in range(base, tick):
            ph = ed.step_enemies(self.state, ph)
            self._phase_cache[t + 1] = ph
        return self._phase_cache[tick]

    # ---- the safety gate (worst-case over the band + meanie trigger) ----
    def tile_safe(self, tile, tick, object_top=ed.ROBOT_EYE, check_meanie=True):
        """True if dwelling on `tile` is safe across [tick, tick+ENEMY_PHASE_BAND]:
        no enemy sees it at any tick in that band (worst-case, the MIN layer) AND it
        is not a meanie-spawn trigger.  Pure native.

        `check_meanie` may be cleared for the player's FORCED start tile: the player
        cannot choose where it starts, so the meanie-spawn gate (which is a static,
        phase-blind geometric capability -- enemy_dynamics models it with the
        conservative `can_see`, not the time-accurate FOV) cannot be satisfied by
        waiting and would otherwise dead-lock a landscape whose start tile is
        meanie-flagged.  The time-accurate SIGHTLINE gate still applies to the start
        tile; only the unavoidable static meanie capability is waived there."""
        key = (tile, tick, round(object_top, 3), check_meanie)
        v = self._safe_cache.get(key)
        if v is not None:
            return v
        ph = self.phase_at(tick)
        x, y = tile
        exposed = ed.exposed_within_window(
            self.state, ph, x, y, ticks=ENEMY_PHASE_BAND, object_top=object_top
        )
        safe = (not exposed) and (
            not check_meanie or ed.meanie_safe(self.state, (x, y))
        )
        self._safe_cache[key] = safe
        return safe


def _find_climb_path_timed(
    g,
    target_z,
    tc,
    log=lambda *a: None,
    max_nodes=20000,
    sweep_counter=None,
    gate_meanie=True,
):
    """A* over (tile, z_height, tick) that travels to the platform AND ascends to
    >= target_z while NEVER dwelling on a tile an enemy sees (worst-case band).

    Mirrors native_game._find_climb_path but adds the time dimension:
      * each step costs TICKS_PER_ACTION enemy ticks (advance tick),
      * a WAIT action advances tick in place (WAIT_TICKS) when nothing safe is
        reachable now,
      * a move is rejected unless BOTH the dwell-during-build tile (current tile)
        and the landing tile are tile_safe at the relevant ticks.
    Returns a list of (tile, use_boulder, view, n_waits_before) moves or None.

    OCCLUSION FIX (place-and-re-sweep, plan/validate/blocklist).  Like
    native_game._find_climb_path, the bare timed A* sweeps over the base terrain and
    cannot see the boulders/synthoids/abandoned shells it drops, so the real $1B46
    gate later refuses an occluded build.  We wrap the timed A* in the same outer loop:
    plan a route, replay it accumulating its TRUE objects and re-sweeping each foothold
    with them materialized (native_game._validate_path_occlusion), and if a foothold is
    occluded blocklist that edge and replan.  The enemy-exposure / meanie / WAIT model
    is unchanged."""
    start_tile = g.player_xy()  # the forced start tile (meanie gate waived)
    ps = g.player
    base = bytes(g.mem)
    ox0, oy0 = g.mem[ng.OBJ_X + ps], g.mem[ng.OBJ_Y + ps]
    goal = lambda s: cheb(s[0], g.plat) <= 1 and s[1] >= target_z
    hh = lambda s: cheb(s[0], g.plat) + max(0, target_z - s[1])
    cache = {}

    def sweep_from(tile, z, efrac):
        c = cache.get((tile, z, efrac))
        if c is not None:
            return c
        # bare-terrain sweep with the observer placed at `tile` and the correct stack
        # eye_frac (occlusion correctness is enforced by the outer blocklist loop's
        # place-and-re-sweep validation; the frac keeps the bare A* footholds faithful).
        sw = ng.sweep_with_placed(base, ps, z, tile, frozenset(), eye_frac=efrac)
        if sweep_counter is not None:
            sweep_counter[0] += 1
        cache[(tile, z, efrac)] = sw
        return sw

    def timed_astar(blocked):
        start = (start_tile, int(g.eye), int(g.mem[ng.OBJ_ZF + ps]), 0)
        openq = [(hh(start), 0.0, start)]
        gc = {start: 0.0}
        came = {start: (None, None)}
        found = None
        nodes = 0
        while openq:
            _f, c, s = heapq.heappop(openq)
            if goal(s):
                found = s
                break
            if c > gc.get(s, 1e18) or nodes > max_nodes:
                continue
            nodes += 1
            tile, z, efrac, tick = s
            dwell_safe = tc.tile_safe(
                tile, tick, check_meanie=(gate_meanie and tile != start_tile)
            )
            # WAIT: advance tick in place (look-ahead).
            wait_state = (tile, z, efrac, tick + WAIT_TICKS)
            if tick < MAX_WAITS * WAIT_TICKS:
                nc = c + 0.05
                if nc < gc.get(wait_state, 1e18):
                    gc[wait_state] = nc
                    came[wait_state] = (s, ("WAIT",))
                    heapq.heappush(openq, (nc + hh(wait_state), nc, wait_state))
            if not dwell_safe:
                continue
            land_tick = tick + TICKS_PER_ACTION
            for T2, view in sweep_from(tile, z, efrac).items():
                if T2 == tile:
                    continue
                tz2 = terrain_z(g.mem, *T2)
                if tz2 is None:
                    continue
                if not tc.tile_safe(T2, land_tick, check_meanie=gate_meanie):
                    continue
                d = cheb(T2, tile)
                hop_cost = 1.0 + 0.12 * max(0, d - 2)
                moves = [(False, hop_cost)]
                if d <= 1 and tz2 <= z:
                    moves.append((True, 1.3))
                for ub, cost in moves:
                    if ((tile, z, T2, ub)) in blocked:
                        continue
                    _np, neye, nfrac = ng._move_placed(
                        frozenset(), tile, z, T2, ub, tz2, z_frac=efrac
                    )
                    ns = (T2, neye, nfrac, land_tick)
                    nc = c + cost
                    if nc < gc.get(ns, 1e18):
                        gc[ns] = nc
                        came[ns] = (s, (T2, ub, view))
                        heapq.heappush(openq, (nc + hh(ns), nc, ns))
        if found is None:
            return None, nodes
        raw = []
        s = found
        while came[s][1] is not None:
            raw.append(came[s][1])
            s = came[s][0]
        raw.reverse()
        path = []
        pending_waits = 0
        for mv in raw:
            if mv[0] == "WAIT":
                pending_waits += 1
                continue
            T2, ub, view = mv
            path.append((T2, ub, view, pending_waits))
            pending_waits = 0
        return path, nodes

    # ---- plan / validate-occlusion / blocklist / replan ----
    blocked = set()
    for it in range(60):
        timed_path, nodes = timed_astar(blocked)
        if timed_path is None:
            g.mem[ng.OBJ_X + ps], g.mem[ng.OBJ_Y + ps] = ox0, oy0
            log(f"  timed A*: NO PATH after {it} replans ({len(blocked)} blocked)")
            return None
        # validate occlusion against the route's TRUE accumulated objects (drop waits;
        # geometry only).  _validate_path_occlusion expects (T2, ub, view, from_eye);
        # the timed path carries n_waits in slot 3, so adapt the move tuples.
        geo = [(T2, ub, view, None) for (T2, ub, view, _w) in timed_path]
        ok, bad, refreshed = ng._validate_path_occlusion(g, geo, sweep_counter)
        if ok:
            # re-attach the wait counts to the (occlusion-refreshed) views.
            out = [
                (geo_mv[0], geo_mv[1], geo_mv[2], tp[3])
                for geo_mv, tp in zip(refreshed, timed_path)
            ]
            g.mem[ng.OBJ_X + ps], g.mem[ng.OBJ_Y + ps] = ox0, oy0
            log(
                f"  timed A*: {len(out)} moves, {nodes} nodes, {it} replans, "
                f"{len(blocked)} blocked, {len(cache)} sweeps"
            )
            return out
        blocked.add(bad)
    g.mem[ng.OBJ_X + ps], g.mem[ng.OBJ_Y + ps] = ox0, oy0
    log("  timed A*: blocklist-replan budget exhausted")
    return None


def plan_timed(landscape, verbose=True, top_energy=True, gate_meanie=True):
    """Timed climb-and-win planner.  Same return shape as native_game.plan (a Game
    with .steps and .native_won), but the route is chosen so the player is never
    seen by a rotating enemy while it dwells (worst-case band + meanie gate).

    The emitted .steps carry an extra 'tick' field (informational; validate_kbd_plan
    ignores unknown fields) so the replay/verification can check exposure per tick."""
    t0 = time.time()
    g = Game(landscape)
    if top_energy:
        g.energy = 63
    tc = TimedClimb(g)
    log = lambda *a: verbose and print(*a)
    log(
        f"ls{landscape}: start {g.player_xy()} eye {g.eye} platform {g.plat} "
        f"plat_ground {g.plat_ground} enemies {len(tc.phase0.enemies)}"
    )
    n_sweeps = [0]
    cur_tick = [0]

    def annotate(tick):
        # tag the most recent step with the enemy tick it dwells at + a safety note.
        if g.steps:
            x, y = g.player_xy()
            seers = ed.enemies_seeing(tc.state, tc.phase_at(tick), x, y)
            g.steps[-1]["tick"] = tick
            g.steps[-1]["seen_by"] = seers

    target_z = (g.plat_ground or 8) + 1
    path = _find_climb_path_timed(
        g, target_z, tc, log=log, sweep_counter=n_sweeps, gate_meanie=gate_meanie
    )
    if path is None:
        log("  NO TIMED PATH found")
    else:
        for tile, use_b, view, n_waits in path:
            cur_tick[0] += n_waits * WAIT_TICKS
            prev_slot, prev_tile = g.player, g.player_xy()
            if use_b:
                g.create(3, tile, view, f"climb boulder (waited {n_waits})")
                sslot = g.create(0, tile, None, "climb synthoid")
            else:
                sslot = g.create(0, tile, view, f"hop synthoid (waited {n_waits})")
            g.transfer(sslot, "step")
            cur_tick[0] += TICKS_PER_ACTION
            annotate(cur_tick[0])
            sw2 = visibility_sweep(g.mem, g.player, int(g.eye))
            n_sweeps[0] += 1
            if (
                prev_tile in sw2
                and g.mem[ng.OBJ_TYPE + prev_slot] == 0
                and prev_tile not in g.col
            ):
                g.absorb(prev_slot, sw2[prev_tile], "reabsorb prior synthoid")
                cur_tick[0] += TICKS_PER_ACTION
            log(
                f"  {'step' if use_b else 'hop'} -> {tile} eye {g.eye} "
                f"(d={cheb(g.player_xy(), g.plat)}) tick {cur_tick[0]} "
                f"seen_by {ed.enemies_seeing(tc.state, tc.phase_at(cur_tick[0]), *g.player_xy())}"
            )

    # ---- final win: absorb the Sentinel from above its base tile, put a synthoid
    # on the platform and transfer onto it (hyperspace win).  The vantage tile we
    # stand on for the absorb was already gated safe by the A*.
    won = False
    if g.plat_ground is not None:
        sw = visibility_sweep(g.mem, g.player, int(g.eye))
        n_sweeps[0] += 1
        if (
            int(g.eye) > g.plat_ground
            and cheb(g.player_xy(), g.plat) <= 1
            and g.sentinel_slot is not None
        ):
            g.absorb(g.sentinel_slot, sw.get(g.plat), "absorb Sentinel")
            cur_tick[0] += TICKS_PER_ACTION
            annotate(cur_tick[0])
            if g.feasible(0, g.plat):
                g.transfer(
                    g.create(0, g.plat, None, "platform synthoid"),
                    "hyperspace onto platform (WIN)",
                )
                cur_tick[0] += TICKS_PER_ACTION
                won = True
                log(f"  WIN: synthoid on platform {g.plat} + transfer")
    log(
        f"=== timed plan {'WON' if won else 'INCOMPLETE'} in {time.time()-t0:.2f}s, "
        f"{len(g.steps)} steps, {n_sweeps[0]} sweeps, final tick {cur_tick[0]} ==="
    )
    g.native_won = won
    g.tc = tc
    g.final_tick = cur_tick[0]
    return g


def check_exposure(g):
    """Audit a planned Game: for every step that has a 'tick', confirm the dwell
    tile is NOT seen by any enemy at that tick.  Returns (clean, violations)."""
    tc = g.tc
    violations = []
    for i, st in enumerate(g.steps):
        if "tick" not in st:
            continue
        tile = tuple(st["player_tile"])
        tick = st["tick"]
        seers = ed.enemies_seeing(tc.state, tc.phase_at(tick), *tile)
        if seers:
            violations.append((i, st["verb"], tile, tick, seers))
    return (not violations), violations


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gate_meanie = "--no-meanie" not in sys.argv
    g = plan_timed(ls, gate_meanie=gate_meanie)
    clean, viol = check_exposure(g)
    print(f"exposure audit: {'CLEAN' if clean else 'VIOLATIONS'} ({len(viol)})")
    for v in viol:
        print("  ", v)
    out = {
        "landscape": ls,
        "native_won": g.native_won,
        "final_player": list(g.player_xy()),
        "eye": g.eye,
        "energy": g.energy,
        "final_tick": g.final_tick,
        "steps": g.steps,
    }
    json.dump(out, open(f"out/kbd_timed_{ls:04d}.json", "w"), indent=0)
    print(
        "FINAL",
        g.player_xy(),
        "eye",
        g.eye,
        "energy",
        g.energy,
        "steps",
        len(g.steps),
        "native_won",
        g.native_won,
    )
