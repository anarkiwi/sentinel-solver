# Planner implementation plan (the "how")

Mechanically executable task plan for the design in
[`docs/planner-redesign-proposal.md`](planner-redesign-proposal.md) (the "why").
Each task is ~1 PR. Do not write speculative code beyond a task's scope. Repo
conventions apply: numpy-first / numba where the surrounding code is, black + pylint
(no unused imports/vars), `pytest -n auto`, golden-fixture validation, **60 s hard CPU
budget per script**.

The new solver lives in `solver/` as a set of small modules and **replaces**
`solver/climb_search.py` once at parity (T2.5 / T5.1 swap the runner imports; a final
cleanup task deletes the old file). `sentinel/` is the transition function and is not
modified except where a task explicitly adds a helper.

## Verified API reference (checked against current code)

Node/macros call exactly these (signatures confirmed in the cited files):

- `sentinel.game.Game.new(landscape_number)` / `.clone()` / `.step_enemies()` /
  `.player_xy()` / `.platform_xy()` / `.energy` (property) / `.won()` — `sentinel/game.py`.
- `sentinel.enemies.step(state)` (one round: `tick_cooldowns` + `update_enemies`;
  statefully advances rotation, drain, downgrades, **meanie lifecycle**, discharge) /
  `enemies.enemy_slots(state)` / `enemies.FOV_SCAN` (=0x14) — `sentinel/enemies.py`.
- `sentinel.actions.create(state, otype, tile) -> slot|None` /
  `absorb(state, slot) -> bool` / `transfer(state, slot) -> bool` /
  `can_create(state, otype, tile) -> bool` / `can_absorb(state, slot) -> bool` /
  `on_platform(state) -> bool` / `won(state) -> bool` / `player_dead(state) -> bool` /
  `win(state, tile=None) -> bool` — `sentinel/actions.py`.
- `sentinel.threat.ticks_until_seen(state, x, y, horizon=256, object_top=ROBOT_EYE) -> int`
  / `is_exposed(state, x, y, ...)` / `exposed_tiles(state, tiles)` /
  `gaze_distance(state, tiles) -> {tile:0..128}` / `meanie_safe(state, tile) -> bool` /
  `drain_over_window(state, ticks) -> int` — `sentinel/threat.py`.
- `sentinel.los.aim_target(state, h_angle, v_angle, cur_x, cur_y, player_slot, eye_z=None, max_steps=20000, return_centre=False) -> (tx,ty,los[,centre])`
  / `visible_tiles(state, slot=None, eye_z=None, azimuth_step=8, max_steps=2000) -> {(tx,ty):view}`
  / `sweep_with_centres(state, slot, eye_z, max_steps=200) -> (views, centres)`
  / `sees_tile(state, tile, slot=None, eye_z=None, azimuth_step=8, max_steps=2000) -> bool`
  / `centre_view(state, tile, slot=None, eye_z=None, azimuth_step=8, max_steps=2000) -> view|None`
  / constants `SIGHTS_CX=0x50`, `SIGHTS_CY=0x5F`, `AZIMUTH_STEP=8`, `PITCH_BAND` —
  `sentinel/los.py`. A `view` is `{"h_angle":int,"v_angle":int,"cursor":[cx,cy]}`.
- `sentinel.aimcost.bearing_to(ex,ey,tx,ty) -> h|None` / `angle_dist(a,b) -> 0..128` /
  `h_steps(h0,h1)` / `v_steps(v0,v1)` / `h_press_count(h0,h1) -> (n_uturn,n_step)` /
  `bearing_rounds(h0,h1,rounds_per_step,rounds_per_uturn) -> float` /
  `pan_steps(h0,v0,h1,v1) -> int` — `sentinel/aimcost.py`.
- `sentinel.actioncost.action_rounds(mem, verb, view, stacked=False) -> float` /
  `is_stacked(mem, tile) -> bool` / `SETTLE={"absorb":190,"create":290,"transfer":300}` /
  `STACK_CREATE=285` — `sentinel/actioncost.py`.
- `sentinel.state.State.clone()` / `.energy` / `.player` / `.player_xy()` /
  `.platform_xy` / `.free_slots()` / `.occupied_slots()` / `.is_empty(slot)` /
  `.slot_of_type(otype)` / `.obj_x/obj_y/obj_z_height/obj_z_frac/obj_h_angle/obj_type/obj_flags`
  (indexable views) — `sentinel/state.py`.
- `solver.plan_game.PlanGame(landscape)` / `.from_mem(mem,...)` / `.clone()` /
  `.create(otype, tile, view, note="") -> slot|None` / `.absorb(slot, view, note="")` /
  `.transfer(slot, note="")` / `.feasible(otype, tile) -> bool` / `.top_of(tile) -> float|None`
  / `.player_xy()` / attributes `.state .mem .col .eye .steps .free .player .energy
  .plat .plat_ground .sentinel_slot .native_won` — `solver/plan_game.py`.
- Module helpers `solver.plan_game.terrain_z(mem_or_state,x,y) -> nibble|None` /
  `cheb(a,b)` / `visibility_sweep(mem,slot,eye_z,max_steps,coarse)` /
  `sees_tile(mem,tile,slot,eye_z,max_steps)` / `centre_view_for(mem,tile,slot,eye_z,...)`.
- `sentinel.memmap` addresses used by the closed-set key:
  `ENEMIES_UPDATE_COOLDOWN=0x0C30`, `OBJECTS_H_ANGLE=0x09C0`, `T_MEANIE=4`,
  `T_SENTINEL=5`, `ENEMY_TYPES=(1,5)`, `ENERGY_IN_OBJECTS`, `PRND_STATE=0x0C7B`,
  `NUM_SLOTS=64`.
- Runner `scripts/run_plan_simulated.py`: `execute_step(world, stp, heading, budget, log)`,
  `advance(world, rounds, budget)`, `make_world(landscape)`, resync via
  `PlanGame.from_mem(world.mem, ...)`.

---

## Task table

| ID | Title | Files | Deps | Gate |
|----|-------|-------|------|------|
| **P-1** | Stateful meanie + two-probe exposure byte (IN PROGRESS, external) | `sentinel/enemies.py`, `sentinel/relative.py`, `sentinel/tests/golden_meanie.json`, `golden_exposure.json` | — | golden 0-divergence vs ROM |
| T0.1 | Search node + closed-set key | `solver/search_node.py` (new) | P-1 | unit: clone_apply == direct sentinel ops |
| T0.2 | Gaze timeline oracle | `solver/gaze.py` (new) | P-1 | agrees with `ticks_until_seen` t=0, all ls0 tiles |
| T0.3 | Move cost model | `solver/cost.py` (new) | — | matches `execute_step` settle for scripted seq |
| T1.1 | Down-look launch enumeration + terminal test | `solver/launch.py` (new) | T0.1 | ls0 set ⊇ human launch tile, ∌ dead-end corner |
| T1.2 | Managed-exposure feasibility over window | `solver/cost.py` | T0.1,T0.3 | energy delta over window == sim exactly |
| T2.1 | Climb macros (hop + boulder-step) | `solver/macros.py` (new) | T0.1,T0.3,T1.2 | unit: macro reproduces `_apply` eye/energy |
| T2.2 | Refuel macro (visibility-deferred) | `solver/macros.py` | T2.1 | refuel emitted only when energy-blocked |
| T2.3 | Endgame macro (down-look, drive-through) | `solver/macros.py` | T1.1,T2.1 | from launch state, endgame wins in sim |
| T2.4 | Weighted-A\* driver + heuristic + failure contract | `solver/astar_planner.py` (new) | T2.1,T2.2,T2.3 | `plan(0)` won, ≤ ~12 macro steps |
| T2.5 | Wire `run_plan_simulated` to offline-plan + replan | `scripts/run_plan_simulated.py` | T2.4 | `run_plan_simulated.py 0` WON, **zero drain** |
| T3.1 | Exposed-but-survivable climb variants | `solver/macros.py` | T2.4 | no-hidden-route map WON, energy>0 all windows |
| T3.2 | Decoy macro (gaze pin) | `solver/macros.py` | T3.1 | sim shows pinned enemy stops precessing over window |
| T3.3 | Multi-sentry gap intersection + greedy batch fit | `solver/gaze.py`, `solver/macros.py` | T3.1 | batch fits intersection; drain within buffer |
| T4.1 | ARA\* anytime wrapper | `solver/astar_planner.py` | T2.4 | dense map: feasible plan within T_BUDGET |
| T4.2 | Deterministic UCT escalation tier | `solver/uct.py` (new) | T2.4,T4.1 | high-branch node: UCT returns feasible plan |
| T4.3 | Multiprocess offline solve | `solver/astar_planner.py` | T2.4 | solve within 60 s (or authorized exception) |
| T4.4 | Loud-failure contract + unsolvable test | `solver/astar_planner.py`, `solver/tests/test_failure.py` | T2.4 | unsolvable → structured failure, nonzero exit, no false win |
| T4.5 | (optional) CP-SAT inner scheduler | `solver/schedule_cpsat.py` (new) | T3.3 | parity with greedy; solves a greedy-miss case |
| T5.1 | Wire `run_plan_live`; missed-aim-is-crash | `scripts/run_plan_live.py` | T2.5 | recorded live ls0 win, no missed aim |
| T5.2 | Live multi-enemy win | (config/validation) | T5.1,T3.x | recorded live win, multi-enemy landscape |
| T6.0 | Delete `climb_search.py`, update imports/tests | `solver/climb_search.py`, callers | T2.5,T5.1 | `pytest -n auto` green |

Global config defaults (one place, `solver/astar_planner.py` module constants, all
env-overridable): `W_ASTAR=1.5`, `MAX_H_PER_MOVE=2.0`, `MIN_MOVE_ROUNDS=290.0`,
`T_BUCKET=64`, `NEXT_COST_FLOOR=3`, `HORIZON=4000`, `NODE_BUDGET=20000`,
`T_BUDGET_S=45.0`, `BEAM=8`, `BRANCH_HIGH=24`, `SAFETY_HORIZON=256`.

---

## Phase -1 — Meanie substrate (prerequisite, external, in progress)

**P-1.** A separate subagent is making the meanie lifecycle + the two-probe exposure
byte bit-exact and ROM-validated. **Do not re-specify its internals.** The planner
depends only on these post-conditions, which every downstream task assumes:

- `sentinel.enemies.step(state)` **statefully** advances any meanie (spawn from a
  tree, rotate/move toward the player, forced hyperspace, expiry) as a side effect on
  `state`, so a node that clones a `Game`/`PlanGame` and calls `step` sees meanies
  evolve exactly as the live game. No separate meanie call is needed by the planner.
- Meanie objects occupy normal object slots with `obj_type == mm.T_MEANIE (4)`.
- Forced hyperspace / drain-death are observable via `actions.player_dead(state)` and
  `actions.on_platform(state)` (already true today).
- The full/partial exposure classification driving spawn is correct (fixes the
  `docs/simulator.md` two-probe approximation).

**Gate (owned by P-1):** new `golden_meanie.json` + `golden_exposure.json`,
0-divergence vs the py65 ROM oracle over hundreds of rounds on meanie-spawning
landscapes. Downstream tasks may begin against the current `enemies.step` and tighten
their meanie assertions once P-1 lands.

---

## Phase 0 — Complete-state search substrate

### T0.1 — Search node + closed-set key — `solver/search_node.py` (new)

Node wraps a `PlanGame` (reuses its `col`/`eye`/`steps` tracking and bit-exact
`create`/`absorb`/`transfer`) plus the world tick and sights heading.

```python
from dataclasses import dataclass
from typing import Optional
from solver.plan_game import PlanGame
from sentinel import enemies, memmap as mm

@dataclass
class Node:
    g: PlanGame                 # holds sentinel State incl PRNG; .clone() branches
    t: int                      # world tick: number of enemies.step applied since start
    vh: int                     # sights bearing 0..255 (current heading)
    vv: int                     # sights pitch (v_angle)
    cost: float                 # g-cost: cumulative enemy-rounds
    parent: Optional["Node"] = None
    macro: Optional[dict] = None  # record of the macro that produced this node

def eye(n: Node) -> float: return n.g.eye
def energy(n: Node) -> int: return n.g.state.energy
def tile(n: Node) -> tuple: return n.g.player_xy()

def step_world(g: PlanGame, rounds: int) -> None:
    """Advance the real enemy world `rounds` rounds on g.state (drain/rotate/meanie)."""
    for _ in range(int(rounds)):
        enemies.step(g.state)

T_BUCKET = 64

def enemy_phase_hash(state) -> tuple:
    ph = tuple(sorted(
        (int(state.obj_type[e]), int(state.obj_h_angle[e]) >> 3,
         1 if state.mem[mm.ENEMIES_UPDATE_COOLDOWN + e] >= 2 else 0)
        for e in enemies.enemy_slots(state)))
    meanies = tuple(sorted(s for s in range(mm.NUM_SLOTS)
                           if not state.is_empty(s) and state.obj_type[s] == mm.T_MEANIE))
    return ph + (meanies,)

def node_key(n: Node) -> tuple:
    st = n.g.state
    return (n.g.player_xy(), round(n.g.eye, 3), st.energy,
            n.t // T_BUCKET, enemy_phase_hash(st))
```

Rationale for the key: two nodes with the same tile/eye/energy, the same coarse tick
bucket, and the same coarse enemy phase (bearing bucketed to 8 units, whether each
enemy is mid-cooldown, and which meanies exist) continue identically enough to dedup.

**Gate:** `solver/tests/test_search_node.py` — build `Node` from `PlanGame(0)`, clone,
apply a `create`+`transfer` two ways (via the node's `PlanGame` and via raw
`sentinel.actions` on a parallel `State`); assert equal energy, player tile, eye, and
`node_key` stability across `.clone()`.

### T0.2 — Gaze timeline oracle — `solver/gaze.py` (new)

Precompute idle enemy facings for the horizon; expose safe-window queries used as an
**admissible heuristic and pruning filter** (not correctness — the true transition
governs once exposure/objects perturb enemies).

```python
class GazeTimeline:
    def __init__(self, state, horizon=4000):
        # forward-sim a passive clone; record facings[e][t] and positions.
        # captures pre-existing-scenery drain events (real enemies.step).
        ...
    def seen_at(self, x, y, t) -> bool:
        # some enemy e: aimcost.angle_dist(bearing(e->tile), facings[e][t]) <= FOV_HALF(10)
        # AND static terrain-LOS enemy->tile (precomputed per (e,tile), cached).
        ...
    def safe_windows(self, x, y) -> list[tuple[int,int]]:
        # complement of seen intervals over [0, horizon].
        ...
    def is_safe(self, x, y, t0, t1) -> bool:
        # [t0, t1] lies wholly inside one safe window.
        ...
    def ticks_until_seen(self, x, y, t_from) -> int:
        # first t >= t_from with seen_at true; horizon if none.
        ...
```

Implementation notes: `FOV_HALF = 10` (= `enemies.FOV_SCAN // 2`). Static
enemy→tile terrain-LOS: reuse `sentinel.relative.can_see_object` against a phantom
robot (copy the ~12-line placement from `threat._place_phantom`, which is module-
private) with the facing gate dropped for the *terrain* factor, then intersect with
the per-tick cone. numpy-first: store `facings` as an `(n_enemy, horizon)` uint8 array.

**Gate:** `solver/tests/test_gaze.py` — for `Game.new(0)`, for every tile,
`GazeTimeline.ticks_until_seen(x,y,0)` equals `sentinel.threat.ticks_until_seen(
state, x, y, horizon=256)` (min over `min(., 256)`), 0 mismatches.

### T0.3 — Move cost model — `solver/cost.py` (new)

Elapsed enemy-rounds per aim/action, from the keyboard geometry (mirrors the verified
`climb_search._move_cost` / `_pan_rounds`, using only `aimcost` + `actioncost`).

```python
from sentinel import aimcost as ac, actioncost, memmap as mm

ROUNDS_PER_H_STEP = 16.0
ROUNDS_PER_V_STEP = 8.0
ROUNDS_PER_UTURN  = 16.0

def aim_rounds(h0, v0, view) -> float:
    if not view or view.get("h_angle") is None: return 0.0
    r = ac.bearing_rounds(h0, view["h_angle"], ROUNDS_PER_H_STEP, ROUNDS_PER_UTURN)
    if view.get("v_angle") is not None and v0 is not None:
        r += ac.v_steps(v0, view["v_angle"]) * ROUNDS_PER_V_STEP
    return r

def move_rounds(g, T2, use_boulder, n_boulders, view, vh, vv) -> tuple[float,int,int]:
    """(rounds, end_h, end_v) for a climb macro: aim+build+transfer+look-back reabsorb.
    Prices each fired verb by actioncost.SETTLE; a stacked create adds STACK_CREATE."""
    prev = g.player_xy()
    r = aim_rounds(vh, vv, view)
    if use_boulder:
        n = max(1, n_boulders)
        r += ROUNDS_PER_H_STEP + ROUNDS_PER_V_STEP  # recentre on-boulder synthoid
        first_stacked = T2 in g.col
        settle = actioncost.SETTLE["create"] + (actioncost.STACK_CREATE if first_stacked else 0.0)
        settle += n * (actioncost.SETTLE["create"] + actioncost.STACK_CREATE)  # bldrs 2..n + synth
    else:
        settle = actioncost.SETTLE["create"]
    settle += actioncost.SETTLE["transfer"]
    end_h, end_v = (view.get("h_angle"), view.get("v_angle")) if view else (vh, vv)
    back_h = ac.bearing_to(T2[0], T2[1], prev[0], prev[1])
    if back_h is not None and end_h is not None:
        r += ac.bearing_rounds(end_h, back_h, ROUNDS_PER_H_STEP, ROUNDS_PER_UTURN)
        end_h = back_h
        settle += actioncost.SETTLE["absorb"]
    return r + settle, end_h, end_v
```

**Gate:** `solver/tests/test_cost.py` — for a scripted `create`→`transfer`→`absorb`
sequence, `move_rounds` total equals the sum of `actioncost.action_rounds(mem, verb,
view, stacked=...)` that `scripts/run_plan_simulated.execute_step` would apply
(within ±1 rounding), for both a hop and a 1-boulder step on ls0.

---

## Phase 1 — Managed exposure feasibility + down-look launch

### T1.1 — Down-look launch enumeration + terminal test — `solver/launch.py` (new)

Fix the symmetric-LOS bug: the endgame shot is the **player looking down** at the
platform, legal where the reverse up-shot is blocked (`$1D2E`).

```python
def launch_tiles(state, plat, plat_ground) -> set[tuple[int,int]]:
    """Every tile from which a robot standing there (eye = its terrain height) has
    down-look LOS to the platform tile. Phantom-place a robot at each tile with
    terrain_z >= plat_ground, aim at plat via los.sees_tile(clone, plat, slot,
    eye_z=tile_eye). Inline the phantom placement (see threat._place_phantom)."""

def down_look_los(g, plat) -> bool:
    """LOS from the player's CURRENT tile/eye down to plat. Ceil the observer when the
    eye carries a fraction (a fractional eye above plat_ground sees down onto it):
    seye = int(eye)+1 if eye>int(eye) else int(eye); los.sees_tile(g.state, plat,
    g.player, eye_z=seye, max_steps=200)."""

def endgame_ready(g, plat, plat_ground) -> bool:
    return g.eye > plat_ground and down_look_los(g, plat)
```

**Recovered human ls0 win (pinned from the log `out/play_20260704_171942.jsonl`,
blob in commit `f74bbaa`).** Full climb ladder (tile @ eye), each landing tile's bare
terrain one unit higher than the last:

    (8,17)@5.875 → (9,7)@6.375 → (10,23)@7.375 → (9,3)@8.375 → (2,10)@9.375 → win

- **Launch tile = `(2,10)` at eye 9.375** (standing on a boulder, `zf=0x60`): the tile
  the endgame absorb was fired from.
- **Platform tile = `(12,4)`** (ground height 9): the player's final winning tile —
  Chebyshev-**10** from the launch tile ("launch from afar"), and the same tile the
  current native solver ends its win on (`FINAL (12,4) eye 9.875`).

**Gate:** `solver/tests/test_launch.py` — on ls0, `launch_tiles(state, (12,4), 9)`
**contains** `(2,10)` (the human launch tile) and **excludes** the tallest dead-end
corner the current `_launch_tiles` picks (assert that coordinate absent). Also assert
`down_look_los` is True from `(2,10)` at eye 9.375 to `(12,4)` and — the actual bug —
that the *symmetric* platform-vantage reverse sweep does **not** see `(2,10)` (proving
the down-look asymmetry is what recovers the tile).

### T1.2 — Managed-exposure feasibility over the window — `solver/cost.py`

The timed-race math (README strategy 1) as a hard feasibility test via the **true
transition** — no soft penalty, no blanket veto.

```python
NEXT_COST_FLOOR = 3  # keep enough energy for one synthoid after the window

def survivable(g_after_actions, from_tile, window_rounds) -> tuple[bool, int]:
    """g_after_actions has the macro's builds applied but the world NOT yet advanced.
    Step the real world `window_rounds` (drain/rotate/meanie), then test survival.
    Returns (ok, energy_after). ok iff not player_dead AND energy_after > 0 AND
    energy_after >= NEXT_COST_FLOOR."""
    e0 = g_after_actions.state.energy
    for _ in range(int(window_rounds)):
        enemies.step(g_after_actions.state)
    if actions.player_dead(g_after_actions.state):
        return (False, g_after_actions.state.energy)
    ea = g_after_actions.state.energy
    return (ea > 0 and ea >= NEXT_COST_FLOOR, ea)
```

The inequality, written out: a macro spanning `W` rounds from `from_tile` is feasible
iff `energy_after = energy_before - drain(W) >= NEXT_COST_FLOOR` and the player is not
dead, where `drain(W)` is produced by the real `enemies.step` loop (0 for a hidden
window). Cheapest common case (hidden): `drain(W)=0`, always passes.

**Gate:** `solver/tests/test_survivable.py` — construct an ls0 state seen by the
Sentinel; assert `survivable`'s `energy_after` equals `energy_before -
sentinel.threat.drain_over_window(state, W)` exactly (both drive `enemies.step`), and
that a state with insufficient buffer returns `ok=False`.

---

## Phase 2 — Macro A\* wins ls0 (never-seen as the emergent cheapest plan)

### T2.1 — Climb macros — `solver/macros.py` (new)

Each expander returns a list of child `Node`s. Climb footholds come from a keyboard
LOS sweep; each buildable tile yields a hop and (if it climbs) a boulder-step, priced
and feasibility-gated.

```python
def expand_climb(n, gaze) -> list[Node]:
    g, cur = n.g, n.g.player_xy()
    views, centres = los.sweep_with_centres(g.state, g.player, int(g.eye), max_steps=200)
    out = []
    for T2, view in views.items():
        if T2 == cur or plan_game.terrain_z(g.mem, *T2) is None and T2 not in g.col:
            pass  # allow boulder-topped columns; skip non-buildable tops
        top = _top_type(g, T2)                       # None|3 ok to build on
        if top is not None and top != mm.T_BOULDER: continue
        for use_b, n_b in _foothold_options(g, T2, view, centres):   # hop, then bounded batch
            he = _foothold_eye(g, T2, use_b, n_b)
            if he is None or he < g.eye - 1e-9: continue             # non-regression
            child = _apply_climb(n, T2, use_b, n_b, view, gaze)
            if child is not None: out.append(child)
    return out
```

`_apply_climb` is the concrete macro body (mirrors the verified `climb_search._apply`,
on a cloned `PlanGame`, then advances the world and runs the T1.2 feasibility test):

```python
def _apply_climb(n, T2, use_b, n_b, view, gaze) -> Optional[Node]:
    g = n.g.clone()
    W, end_h, end_v = cost.move_rounds(g, T2, use_b, n_b, view, n.vh, n.vv)
    # legality gate 1: aim/LOS/energy feasible up front
    if not _buildable(g, T2, use_b, n_b): return None
    prev_slot, prev_tile = g.player, g.player_xy()
    if use_b:
        for i in range(max(1, n_b)):
            g.create(mm.T_BOULDER, T2, view if i == 0 else None, "climb boulder")
        s = g.create(mm.T_ROBOT, T2, None, "climb synthoid")
    else:
        s = g.create(mm.T_ROBOT, T2, view, "hop synthoid")
    if s is None: return None
    g.transfer(s, "step")
    # look-back reabsorb of the departed shell if now below eye and in LOS
    sw = plan_game.visibility_sweep(g.mem, g.player, int(g.eye), max_steps=200)
    if (prev_tile in sw and g.state.obj_type[prev_slot] == mm.T_ROBOT and
            _full_top(g, prev_slot) <= g.eye + 1e-9):
        g.absorb(prev_slot, sw[prev_tile], "reabsorb prior shell")
    # legality gate 2: managed-exposure survivability over the whole window
    ok, _ea = cost.survivable(g, prev_tile, W)
    if not ok: return None
    return Node(g=g, t=n.t + int(round(W)), vh=end_h, vv=end_v,
                cost=n.cost + W, parent=n, macro={"kind":"climb","tile":list(T2),
                "use_boulder":use_b,"n_boulders":n_b})
```

Helpers `_top_type`, `_foothold_eye`, `_boulder_batch` are ported verbatim from
`climb_search.py` (they are validated mechanics). `_buildable` wraps
`actions.can_create` + the on-boulder `centre_view` check
(`climb_search._boulder_centre_feasible`). Batch size default: largest `n` fitting the
build cap `eye + ROBOT_EYE_FUDGE(2)` that energy affords, ≤ 4.

**Gate:** `solver/tests/test_macros_climb.py` — a single `expand_climb` child on ls0
reproduces the eye and energy that the ported `climb_search._apply` produces for the
same foothold (parity assertion against the old function while it still exists).

### T2.2 — Refuel macro (visibility-deferred) — `solver/macros.py`

Emit refuel children only when energy would otherwise block the cheapest climb child;
prefer fuel poorly visible from height.

```python
def expand_refuel(n, gaze, need=NEXT_COST_FLOOR + 2) -> list[Node]:
    if n.g.energy >= need: return []           # defer: only refuel when constrained
    # absorb below-eye, in-LOS fuel (trees/boulders/synthoids/sentries), one per tile,
    # topmost-first; skip broadly-visible fuel (visible from many sweep tiles) unless it
    # is the only affordable source. Advance world by the aim+absorb window; survivable().
    ...
```

**Gate:** `solver/tests/test_macros_refuel.py` — with energy ≥ `need`, `expand_refuel`
returns `[]`; with energy below `need` and a reachable tree, it returns a child whose
energy increased and whose window was survivable.

### T2.3 — Endgame macro — `solver/macros.py`

```python
def endgame_child(n, plat, plat_ground) -> Optional[Node]:
    if not launch.endgame_ready(n.g, plat, plat_ground): return None
    g = n.g.clone()
    seye = int(g.eye) + (1 if g.eye > int(g.eye) else 0)
    sw = plan_game.visibility_sweep(g.mem, g.player, seye, max_steps=200)
    if plat not in sw: return None
    sent = g.state.slot_of_type(mm.T_SENTINEL)
    if sent is not None: g.absorb(sent, sw.get(plat), "absorb Sentinel")   # drive-through
    if not g.feasible(mm.T_ROBOT, plat): return None
    s = g.create(mm.T_ROBOT, plat, sw.get(plat), "platform synthoid")
    g.transfer(s, "hyperspace onto platform (WIN)")
    if actions.on_platform(g.state) and not actions.player_dead(g.state):
        return Node(g=g, t=n.t, vh=n.vh, vv=n.vv, cost=n.cost, parent=n,
                    macro={"kind":"endgame"})
    return None
```

**Gate:** `solver/tests/test_endgame.py` — from a hand-constructed launch-ready ls0
node (eye > plat_ground, down-LOS to plat), `endgame_child` returns a node with
`actions.on_platform(node.g.state) is True`.

### T2.4 — Weighted-A\* driver + heuristic + failure contract — `solver/astar_planner.py` (new)

```python
W_ASTAR = 1.5; MAX_H_PER_MOVE = 2.0; MIN_MOVE_ROUNDS = 290.0
BEAM = 8; NODE_BUDGET = 20000; HORIZON = 4000

def heuristic(n, plat_ground) -> float:
    dh = max(0.0, (plat_ground + 1) - n.g.eye)
    return math.ceil(dh / MAX_H_PER_MOVE) * MIN_MOVE_ROUNDS   # admissible lower bound

@dataclass
class PlanResult:
    won: bool
    steps: list            # flattened PlanGame.steps of the winning path
    failure: Optional[dict]  # {"reason","blocker","detail"} when not won
    stats: dict            # nodes, wall_s, peak_eye

def plan(landscape_or_game, cfg=None) -> PlanResult:
    g0 = PlanGame(landscape_or_game) if isinstance(landscape_or_game, int) else landscape_or_game
    gaze = GazeTimeline(g0.state, horizon=HORIZON)
    start = Node(g=g0, t=0, vh=_facing_h(g0), vv=_facing_v(g0), cost=0.0)
    openq = [(heuristic(start, g0.plat_ground), 0, start)]   # (f, tiebreak, node)
    closed, best_eye, nodes = {}, g0.eye, 0
    while openq and nodes < NODE_BUDGET:
        f, _, n = heappop(openq)
        end = endgame_child(n, g0.plat, g0.plat_ground)
        if end is not None:
            return PlanResult(True, end.g.steps, None, {"nodes":nodes,...})
        k = node_key(n)
        if k in closed and closed[k] <= n.cost: continue
        closed[k] = n.cost
        best_eye = max(best_eye, n.g.eye)
        children = expand_climb(n, gaze) + expand_refuel(n, gaze)
        children = [c for c in children if energy(c) > 0 and c.t < HORIZON]
        children.sort(key=lambda c: (-c.g.eye, c.cost))       # best-first by height/round
        for c in children[:BEAM]:
            heappush(openq, (c.cost + W_ASTAR*heuristic(c, g0.plat_ground), next(counter), c))
        nodes += 1
    return PlanResult(False, [], _diagnose(best_eye, g0, nodes), {"nodes":nodes,...})
```

`_diagnose` picks the tightest blocker: `no_launch_los` (best_eye above plat_ground but
never down-LOS), `energy_deficit` (climb stalled below plat_ground with no affordable
refuel — report the deficit), `no_safe_window` (all height-gaining children failed
`survivable`), or `budget_exhausted` (NODE_BUDGET hit). Dominance pruning: fold into
`closed` (a state re-reached at higher cost is skipped; extend to strict dominance —
same tile/eye, `energy>=` and `t<=` — as a follow-up if the frontier is large).

**Gate:** `solver/tests/test_plan_ls0.py` — `plan(0).won is True`, path has ≤ ~12
macro steps, runs < 60 s (`pytest` timing), and the energy trace along `steps` is pure
build/absorb (no negative deltas from drain) — i.e. the chosen plan is the hidden one.

### T2.5 — Wire `run_plan_simulated` to offline-plan + replan — `scripts/run_plan_simulated.py`

Replace the `climb_search.search_iterate` decision loop with: `plan()` offline from the
resynced world → execute the plan's `steps` via the existing `execute_step` (which
advances the world by the real `actioncost` rounds) → on any step outcome that diverges
(create landed on a downgraded object, transfer failed, energy short), **resync and
re-`plan()`** from the true `world` state. Keep `execute_step`, `advance`,
`make_world`, and the `actions.player_dead` / `actions.won` checks unchanged.

**Gate (the headline gate):** `python3 scripts/run_plan_simulated.py 0` prints **WON**
and the per-step energy log shows **zero drain** (every energy change is a build cost or
an absorb gain), within the 60 s budget.

---

## Phase 3 — Exposure-required landscapes + decoy

### T3.1 — Exposed-but-survivable climb variants — `solver/macros.py`

`expand_climb` already admits exposed windows *if* `survivable` passes (T1.2). This task
adds the variants that a hidden-only search would skip: allow a foothold whose window is
seen when `survivable` holds, and rank exposed children by `energy_after` (a bigger
surviving buffer first). Add a `prefer_end_drain` bonus for a burst that absorbs its
drainer within the window (ends the drain).

**Gate:** `solver/tests/test_exposed_route.py` — on a chosen multi-sentry landscape (or
a constructed one) with **no fully-hidden route**, `plan().won is True`, and the energy
trace shows energy > 0 across every exposed window (assert min energy over the plan ≥ 1).

### T3.2 — Decoy macro (gaze pin) — `solver/macros.py`

```python
def expand_decoy(n, gaze) -> list[Node]:
    """Place a drainable object (tree/boulder/synthoid) in an enemy E's current view so
    E locks onto it and STOPS precessing (enemies.step rotates only when idle), pinning
    E's gaze away from the player's next work tile for the pinned window. Verify the pin
    through the true transition: after building the decoy and stepping the world, assert
    E's facing is unchanged over the window and E's targeted object is the decoy."""
```

Enumerate decoy tiles from the LOS sweep that fall inside E's current cone
(`threat.gaze_distance`) and are cheap to build; child validity requires the true
transition to confirm the pin (E's `obj_h_angle` static over the window and
`ENEMIES_TARGETED_OBJECT+E` == decoy slot). Emitted only when a subsequent work tile is
unsafe without the pin.

**Measured pin duration (ls0, `enemies.step`).** A drainable object fully in an enemy's
scan cone pins it hard: with a synthoid held in view the enemy's facing stayed constant
for **~591 rounds** before a single rotation (whole run: 1 rotation in 800 rounds),
versus an **idle** enemy rotating every ~200 rounds. The pin lasts as long as the decoy
stays a live, visible target — it is bounded by decoy *longevity* (drain downgrades it
robot→boulder→tree→gone), **not** a short cooldown. So a synthoid/boulder decoy sustains
the pin for many hundreds of rounds — comfortably longer than any single build window
(~200–300 rounds, `actioncost`). Sizing default: assume a decoy holds the gaze for the
decoy's full drain-down lifetime and re-confirm the actual window through the true
transition per plan (the pin is not a fixed constant). *(Measured against HEAD before the
in-flight meanie-substrate fix; re-verify after — the drain/target mechanic is not
meanie-specific, so the ≫window pin is not expected to change.)*

**Gate:** `solver/tests/test_decoy.py` — build a decoy in an enemy's view; step the
world `W` rounds (W ≥ a full build window, e.g. 300); assert the enemy's `obj_h_angle` is
unchanged (pinned) and its targeted-object slot is the decoy, and that a work tile in the
pinned direction is now `survivable` where it was not before.

### T3.3 — Multi-sentry gap intersection + greedy batch fit — `solver/gaze.py`, `solver/macros.py`

Add `GazeTimeline.safe_windows_intersection(tiles_or_tile, enemies="all")` = the
intersection over enemies of each enemy's gaze gaps for the build tile; and a greedy
scheduler for a boulder batch:

```python
def fit_batch(gaze, from_tile, t0, action_windows) -> Optional[int]:
    """Sequentially place each action's [t, t+w) inside one intersected safe window,
    starting at t0 (choose t0 just after the gaze sweeps past for max slack). Return the
    end tick, or None if the batch overruns."""
```

`expand_climb` uses `fit_batch` to size the largest surviving batch on multi-sentry
maps (single-sentry keeps the simple T2.1 sizing).

**Gate:** `solver/tests/test_batch_fit.py` — on a two-sentry landscape, a build batch
scheduled by `fit_batch` has every action window inside the intersection (assert via
`gaze.is_safe` per action), and the whole macro is `survivable`.

---

## Phase 4 — Escalation, budget, failure

### T4.1 — ARA\* anytime wrapper — `solver/astar_planner.py`

When `plan()` hits `NODE_BUDGET` without a goal, restart as ARA\*: run weighted A\* at
`W=3.0`, emit the first feasible plan, then re-run decrementing `W` by 0.5 (down to
1.0) reusing the inconsistent-set, until `T_BUDGET_S=45` elapses; return the best plan
found. `plan(anytime=True)` selects this path.

**Gate:** `solver/tests/test_ara.py` — on a landscape where plain weighted A\* exhausts
`NODE_BUDGET`, `plan(anytime=True).won is True` within `T_BUDGET_S`.

### T4.2 — Deterministic UCT escalation tier — `solver/uct.py` (new)

Engage per-node when `expand_*` yields `> BRANCH_HIGH=24` children (dense exposure/decoy
combinatorics). UCT over macros on the exact sim with **guided greedy rollouts**
(rollout policy = the T2.4 best-first `-eye, cost` order, no randomness), `ITER=2000`,
UCB1 `c=1.4`. Returns the first macro of the best line back to `plan()`.

```python
def uct_choice(n, gaze, plat, plat_ground, iters=2000, c=1.4) -> Optional[Node]:
    ...
```

**Gate:** `solver/tests/test_uct.py` — on a constructed high-branch node, `uct_choice`
returns a child that leads (via greedy rollout) to `endgame_ready`; deterministic across
two runs (same seed-free result).

### T4.3 — Multiprocess offline solve — `solver/astar_planner.py`

Shard the root expansion across processes (`concurrent.futures.ProcessPoolExecutor`,
`os.cpu_count()`): each worker runs `plan()` seeded to expand a disjoint subset of the
root's climb children first; the coordinator returns the first/cheapest win. Keeps each
worker within the 60 s budget.

**Gate:** `solver/tests/test_multiproc.py` — a landscape that single-process solves in
> 40 s solves in < 60 s wall across workers; result identical `won` to single-process.

### T4.4 — Loud-failure contract + unsolvable test — `solver/astar_planner.py`, `solver/tests/test_failure.py`

`PlanResult.failure = {"reason": <enum>, "blocker": <str>, "detail": <dict>}` with
`reason ∈ {no_launch_los, energy_deficit, no_safe_window, budget_exhausted, dead_end}`.
The runners must, on `won is False`, **log the failure loudly and exit nonzero** — never
print a partial "progress" success or silently continue.

**Gate:** `solver/tests/test_failure.py` — construct an unsolvable landscape (e.g. a
platform with no tile that can ever gain down-LOS within energy); `plan().won is False`,
`failure["reason"]` is the correct enum, and a runner smoke-test exits nonzero with the
blocker in its output. Assert **no false win** is ever reported.

### T4.5 — (optional) CP-SAT inner scheduler — `solver/schedule_cpsat.py` (new)

Behind `SCHED_BACKEND=cpsat`. Model the batch/decoy scheduling as CP-SAT interval vars
with `NoOverlap` and per-enemy forbidden windows (from `gaze`), energy-over-window as a
resource; return start offsets or INFEASIBLE. Hand-off boundary: `gaze` supplies the
windows and `cost` the durations; CP-SAT only orders/places actions — it never touches
geometry. Invoked only when `>1` enemy AND the greedy `fit_batch` fails AND branching >
`BRANCH_HIGH`.

**Gate:** `solver/tests/test_cpsat.py` — on solvable cases CP-SAT matches greedy;
construct one case greedy misses (a schedule needing reordering) that CP-SAT solves.

---

## Phase 5 — Live

### T5.1 — Wire `run_plan_live`; missed-aim-is-a-crash — `scripts/run_plan_live.py`

Mirror T2.5 against the driver: `plan()` offline from the resynced live state
(`driver.sentinel_state` read-back, incl. PRNG where available) → execute steps via
`driver.sentinel_execute.perform_step` → on live divergence (meanie now modeled, or a
resync mismatch) resync + re-`plan()`. **A missed aim is a hard crash**: if
`perform_step`'s memory-verify shows the aim landed on the wrong tile, raise and halt
(the plan is aim-exact; a miss is a model bug to investigate, not to smooth).

**Gate:** a recorded live ls0 win (`$0CDE` bit 6 set) via
`python3 scripts/run_plan_live.py --digits 0000`, with **no missed aim** in the log.

### T5.2 — Live multi-enemy win — validation

Run T5.1 on a multi-sentry / meanie landscape (relies on P-1 + T3.x). **Gate:** a
recorded live win on a multi-enemy landscape.

### T6.0 — Remove `climb_search.py` — cleanup

Once T2.5 and T5.1 are green, delete `solver/climb_search.py`, port any still-used
helpers (`_top_type`, `_foothold_eye`, `_boulder_batch`) into `solver/macros.py`, and
update every import / test. **Gate:** `pytest -n auto` green; no reference to
`climb_search` remains.

---

## Notes on defaults and where they may need revisiting

- All numeric defaults (`W_ASTAR`, `MAX_H_PER_MOVE`, `MIN_MOVE_ROUNDS`, `T_BUCKET`,
  `NEXT_COST_FLOOR`, `BEAM`, `NODE_BUDGET`, `HORIZON`, `BRANCH_HIGH`) are module
  constants, env-overridable, tuned per the phase gates.
- `MAX_H_PER_MOVE=2.0` keeps `heuristic` admissible (over-estimates height/move → never
  over-estimates cost). If A\* is too slow, raise `W_ASTAR` before touching `MAX_H_PER_MOVE`.
- The enemy-round scalars (`ROUNDS_PER_*`) match the current `climb_search` /
  `actioncost` calibration; a live divergence recalibrates them from telemetry (T5.1),
  not by adding safety margin.

### Pre-work resolved (was flagged as needing research)

- **T1.1 launch coordinate — resolved.** From the recovered human ls0 win log
  (`f74bbaa:out/play_20260704_171942.jsonl`): launch tile `(2,10)` @ eye 9.375, platform
  `(12,4)`, ladder `(8,17)→(9,7)→(10,23)→(9,3)→(2,10)→(12,4)`. Baked into the T1.1 gate.
- **T3.2 decoy pin — resolved.** Measured on ls0: a drainable object fully in an enemy's
  cone pins its facing for ~591 rounds (1 rotation in 800) vs idle rotation every ~200;
  the pin is bounded by decoy longevity, not a cooldown — ample for any build window.
  Baked into T3.2.
- **Still open (as the design pass flagged):** T0.2's static-enemy→tile LOS cache is
  only valid while no built object enters that enemy's ray (invalidate on any build inside
  an enemy sightline); T3.1's exposure-required landscape id needs a survey pass. Both are
  handled inside their own task gates, not blockers to earlier phases.
