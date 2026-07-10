# The planner (`solver/`)

How a winning action sequence is planned on the bit-exact simulator, why the
current design is what it is, the build plan for it, the outstanding work, and the
superseded design it replaced.

Reader prerequisites: [gameplay.md](gameplay.md) (the rules),
[strategy.md](strategy.md) (the principles a win follows), [simulator.md](simulator.md)
(the `sentinel/` transition function). The plan is executed by the live
[driver.md](driver.md).

---

## 1. Overview

*The Sentinel* is a **single-agent, fully deterministic** game: given the complete
state — every sentry, the Sentinel, any meanies, the board, energy, **and the
40-bit PRNG LFSR** (`sentinel.prng`, `PRND_STATE $0C7B`, tracked in memory) — the
bit-exact simulator is a total transition function with *no* residual randomness.
Hyperspace destinations, created-object facings and drain-scatter placement are all
deterministic functions of that tracked PRNG. There is nothing to plan "under
uncertainty."

The correct model is therefore a **forward search over the complete deterministic
state**, with `sentinel/` as the exact transition function. Exposure, drain, gaze
precession, meanie spawn/hunt and forced-hyperspace are not special cases to veto or
bolt on — they are consequences the search *observes through the true transition*.
The planner is a **deterministic, constructive weighted-A\* search over HTN-style
macro actions**, not a cost-weighted best-first height climb.

- **Exposure is a managed resource, not a veto** (strategy 1). A productive action
  sequence may be run *through* drain when banked energy covers the drain over the
  action window; the endgame absorb is deliberately *driven through* drain because
  it ends the drain as it lands. A fully-hidden ("never seen") route is not
  hard-wired — it is simply the **lowest-cost plan the search prefers when one
  exists**, and it falls out for ls0 for free. On dense maps with no hidden route the
  planner *chooses* survivable exposure, and can even *exploit* it (a decoy: an enemy
  locked on a drainable target stops precessing, holding its gaze away — strategy 4).

- **The hard problem is tractability** (§5–§6): a complete-state forward search over
  a 20–40-action horizon with several rotating enemies and dynamic meanies has a
  large branching factor. The design attacks this with an admissible-heuristic **A\***
  over an **HTN-style macro-action** abstraction, managed-exposure encoded as **hard
  energy feasibility over the action window**, aggressive dominance/time-bucket
  pruning, and a defined **escalation ladder** (weighted A\* → ARA\* → UCT) when
  branching explodes — with a **loud, explained failure** when no survivable route
  fits the budget.

- **A named simulator prerequisite** (§7, Phase -1): the meanie lifecycle must be a
  ROM-validated stateful mechanic before a general planner can forecast meanies. That
  work has landed (`golden_meanie.json` + the two-probe exposure byte are validated;
  see [simulator.md](simulator.md)); with it in place, **meanies are just enemies in
  the state** and the planner needs no meanie special-case.

**OSS verdict: a custom solver on `sentinel/` as the spine**, because the action
model, preconditions and costs are the game's bit-exact ray-march LOS / keyboard-aim
/ tick machinery — unencodable in PDDL without pre-grounding the whole
state-dependent transition. Two frameworks earn a narrowed role in the *heavier*
regime: **OR-Tools CP-SAT** as an optional inner solver for exposure-scheduling into
multi-sentry gaze-gap intersections, and a small **custom UCT** as the anytime
escalation tier — neither displaces the custom spine (§4).

---

## 2. Current status

The simulator (`sentinel/`) is bit-exact and the driver (`driver/`) executes and
verifies live keyboard operations. The **weighted-A\* macro planner
(`solver/astar_planner.py`) wins landscape 0 offline and against the tick-accurate
simulator** (`scripts/run_plan_simulated.py 0`: on-platform, **zero drain**). The
runner is glue: all emulator driving lives in the driver
(`driver.core.boot_and_play`, `driver.sentinel_execute.perform_step` /
`fire_hyperspace`); the runner keeps only the plan-execution loop and the CLI.

The buildability oracle (`los.landable_views` / `landable_view` /
`landable_sweep_with_centres`) is now **ROM-faithful**: it sweeps the sights cursor
at 1 px resolution (the ROM cursor-move step, `$9965`/`$9994`; every 1 px is a
distinct ray sub-angle via `prepare_vector_from_player_sights $1C10`), replacing an
earlier coarse-grid sampling that under-reported buildable tiles. `aim_target`
itself was always bit-exact.

Remaining work is toward reliable **live** wins and **other landscapes** — a
strategy-and-execution problem, tracked in §6. The legacy best-first height search
(formerly under `solver/`) has been **removed** (it is no longer a file); its
structural failure (§3) is the motivation for the current design.

**Not yet solved:** the search does **not** yet win on the faithful oracle. A full
body-pitch-band landable sweep is now ~18× slower than the old coarse grid, so the
outstanding future work is a perf adaptation of the sweep plus a search that wins
under the (now much larger) faithful buildable set.

---

## 3. Why the old legacy height search structurally could not win (history)

The removed legacy planner (formerly `solver/`, a best-first **height** search) had
a stack of soft-cost penalties (`RESERVE`, `EXPOSURE_RESERVE`,
`SEEN_DRAIN`, `REFUEL_DRAIN`, pan cost, launch-readiness, ~20 env weights). Its
architecture fought the correct formalism at every turn — you cannot tune a
soft-cost height optimizer into a complete-state feasibility search; it was replaced,
not patched. Retained here as the motivation for the current design:

1. **Exposure priced as a tradeable soft cost / vetoed by a static mask.** It leans
   on `threat.is_exposed` (the "could ANY rotation ever see this" mask, facing gate
   dropped) which flags essentially every useful high tile, so it either excludes
   winning routes or buries them under reserves; and where it does allow exposure it
   scores it as a penalty rather than testing *survivability over the window*. The
   correct treatment is neither: it is a **hard energy-feasibility constraint over the
   true drained window** (§5.3), under which a hidden route is simply cheapest.
2. **Height is the objective**, not a precondition — producing timid +0.5 creeps and
   tall exposed towers instead of cheap terrain-height gains and, when needed,
   run-through-drain productive bursts.
3. **Pan-only aim cost** (no pitch pricing) → no preference for far/shallow builds.
4. **Symmetric launch LOS** (`_launch_tiles` sweeps from the platform vantage looking
   *up*) under-counts the far down-look launch tiles that actually win (`$1D2E`).
5. **Meanies unmodeled** — only avoided via the static mask or met reactively live.
6. **Only wins ls0 when the modelled per-action drain window is hand-cut ~2.5×** — the
   tell of a wrong model being tuned, not a strategy being executed (consistent with
   the open scan/cooldown cadence calibration gap).

A recorded **human win** on ls0 embodies the strategy the greedy search does not
(strategy 6–10): zero drain (gaze forecast, not the static mask); aim-coherent
ping-pong (≈180° U-turn aims); height first, cheaply, by leapfrogging naturally-high
terrain with self-funding look-back reabsorbs; fuel deferred by visibility; and
launch from afar, looking down. The redesign (§4–§6) reproduces this as the emergent
cheapest plan.

Earlier work also fixed the climb direction: the height-dominant leaf score used to
send the climb to the *tallest reachable* foothold, which on ls0 is a corner on the
wrong side of the map — a dead end with no line to the plinth. The legacy planner was
changed to score a **clear diagonal to the plinth** (a reverse LOS sweep from the
platform vantage) above raw height. This stopped the corner-flight but did **not** by
itself produce a win — confirming the problem is the model, not a tuning knob.

---

## 4. What kind of problem this is (the formalism)

- **Single agent, non-adversarial.** `sentinel.enemies.step` advances rotation,
  targeting, draining, cooldowns, meanie spawn/hunt and forced-hyperspace as a pure
  function of the current state. Nothing *chooses* against the player; the enemies are
  fixed-rate automata. Minimax is the wrong model: there is no MAX/MIN alternation to
  run (confirmed by the re-sentinel numbers — see §8).

- **Full determinism given the complete state.** The PRNG LFSR lives in tracked
  memory (`prng.load/store`), so the three "random" effects — hyperspace destination
  (`do_hyperspace` → `_put_object_in_random_tile_below_z`), created-object facing
  (`create $1F83`), drain-scatter tree placement
  (`_consider_discharging_enemy_energy`) — are deterministic functions of state. The
  complete state is `(Game: board + all object slots incl. sentries/Sentinel/meanies +
  energy + player + PRNG bytes, world tick t)` and the transition is the real bit-exact
  sim.

- **Continuous time, discretized to enemy rounds.** Each keyboard action spans many
  rounds: `sentinel.actioncost.action_rounds` prices settle/redraw, `sentinel.aimcost`
  prices the aim pan. No mid-action pause — an action's whole window is exposed to
  whatever the enemies do during it. Time is a first-class plan dimension.

- **Energy as a managed, hard-feasible resource.** Energy 0..63; each verb a fixed
  cost. Drain is −1 per drain-cooldown *while seen*. Absorbing the drainer ends its
  drain. Feasibility is: energy > 0 throughout every action window, and ≥ the cost of
  the next required action — a real constraint the forward transition evaluates
  exactly, **neither a soft penalty nor a blanket veto** (strategy 1).

- **LOS/aim preconditions.** `create`/`absorb`/`transfer`/`win` gate on a reachable
  keyboard-aim landability (`los.landable_view`/`aim_target`) — exact ROM-faithful oracles.

- **Deterministic gaze precession → schedulable windows.** Idle enemies precess a
  fixed step every ~200 rounds over a ±10 cone. While the player and its objects are
  *unseen*, enemies stay idle, so their facing is a **plan-independent timeline** — but
  only under that condition. This timeline is a **heuristic / pruning device** (a valid
  safety lower bound while hidden), *not* a correctness crutch (§5); the *instant* the
  plan admits exposure or leaves a seen object (perturbing enemy cooldowns/rotation),
  correctness comes only from the true forward transition.

**Formalism.** A **shortest-path / cost-minimizing forward search over the complete
deterministic state graph**, transition = the bit-exact sim, with energy feasibility
over action windows as a hard resource constraint and gaze precession as a heuristic;
made tractable by macro-action (HTN) abstraction and admissible heuristics (§5).

### The exposure reframe: managed resource, emergent zero-exposure

The keystone of the earliest draft — a hard "never seen" invariant — is **demoted to
an emergent optimum**:

- **Exposure is admitted whenever it is survivable.** An action (or macro) spanning
  window `W` from a currently-seen tile is legal iff the true forward transition keeps
  energy > 0 throughout `W` and ≥ the next required cost. Cheaper still if the burst
  *ends* the drain (absorbing the drainer). This is the timed-race math of strategy 1
  as a **hard constraint over the real drained window**, not a penalty.
- **Zero-exposure is the cheap common case, not a rule.** A hidden route incurs zero
  drain, so its energy ledger is pure build/absorb and its cost (elapsed rounds) is
  minimal. The search *prefers* it automatically where it exists (ls0). On maps
  without one, it takes the cheapest *survivable* exposed route.
- **Exposure can be chosen and exploited.** *Chosen*: dense multi-sentry maps with no
  fully-hidden climb — run a productive burst through a bounded drain the buffer
  covers. *Exploited (decoy)*: an enemy locked onto a drainable target **stops
  precessing** (`enemies.step` rotates only when idle); a decoy object in one enemy's
  view *pins its gaze away* from the player's work tile for a schedulable interval,
  built as a first-class macro and verified through the true transition.

### OSS survey (the general, multi-enemy regime)

Verdict: **custom spine on `sentinel/`, with two narrowed framework roles.** The
three questions a planner must answer — is a tile visible from an eye at height `z`;
how many rounds does a keyboard action cost; what does the full enemy+meanie+PRNG
state do next — are *already implemented and ROM-validated* in `sentinel/`
(`los`/`los_jit`, `actioncost`+`aimcost`, `enemies`/`threat`/`prng`).

| Framework | Fit for the general problem |
|-----------|------------------------------|
| **Custom A\*/HTN on `sentinel/`** | **Spine.** Only thing that can host bit-exact LOS/aim/tick + state-dependent enemy & meanie transitions. |
| **OR-Tools CP-SAT** | **Optional inner solver — role grows in the heavier regime.** Once exposure-scheduling dominates (pack build/aim/decoy actions + energy-over-window into the *intersection* of several sentries' gaze gaps), this is a real CSP CP-SAT solves well; it consumes windows/costs the sim supplies, it does not host the geometry. Adopt for the build-into-window sub-problem on dense maps. |
| **Custom UCT (deterministic)** | **Escalation tier.** On dense exposure-scheduling nodes where no admissible heuristic bounds the trade-offs tightly, a deterministic UCT with guided (non-random) rollouts on the exact sim gives an anytime feasible plan without variance. ~100 lines on the same state, not a dependency. |
| **unified-planning / ENHSP / OPTIC / Fast Downward / pyperplan** (PDDL / numeric-temporal) | **No (spine).** The geometry and the *state-dependent action set* (meanies appear/expire; LOS depends on the live object stack) have no faithful declarative encoding — you would ground the reachable transition from the sim per state, i.e. rebuild the custom search. PDDL+ processes could model gaze/energy as timed numeric fluents but not the ray-march LOS. The strongest alternative only if the custom search proves unsteerable. |
| **MCTS libs / pomdp-py / OpenSpiel / RL** | **No.** Built for uncertainty/opponents we don't have; a fully observable, deterministic problem. |
| **networkx** | **Utility only.** The state graph is lazily generated with energy+tick+object labels; a hand-rolled label-correcting A\* is cleaner. Borrow patterns, not the framework. |

---

## 5. Tractability — the central design problem

A complete-state forward search over ~20–40 actions with several rotating enemies and
dynamic meanies has a large state space and branching. This is the core of the design.

### 5.1 State, transition, and the closed set

- **State** = `sentinel.Game.clone()` (bit-exact, incl. PRNG) + world tick `t`.
  Everything — player tile, eye, energy, all enemies/meanies, PRNG — is in the clone;
  enemy facing is derived, not carried separately.
- **Transition** = the real sim: apply the macro/primitive's keyboard actions,
  stepping `enemies.step` for the priced round cost so drain/rotation/meanie dynamics
  are exactly what the live game does over that window.
- **Time-bucketed closed set.** Key `(player_tile, round(eye,3), energy, t // T_BUCKET,
  enemy_phase_hash)` where `enemy_phase_hash` summarizes each enemy's facing (bearing
  bucketed to 8 units), whether it is mid-cooldown, and which meanies exist. `T_BUCKET`
  coarsens schedule equivalence, collapsing states that differ only in immaterial
  micro-timing.

### 5.2 Macro-action (HTN) abstraction — collapse the horizon

Search at the level of **parametric macros**, not keystrokes, so the effective horizon
drops from 20–40 to ~5–8 decisions. Each macro is a scripted parametric sequence whose
internal aims/tiles are solved by a small bounded primitive search and whose
feasibility (LOS, energy-over-window, safety) is checked through the true transition:

- `climb_to(T)` — reach launch/foothold tile `T` via the cheapest hop/step chain
  (terrain hops preferred; boulder steps priced `2n`).
- `build_batch(T, n)` — stack `n` boulders + capping synthoid at `T`, phased into a
  safe/survivable window (§5.4).
- `refuel(region)` — absorb below-eye fuel, generated only when energy would otherwise
  be infeasible; prefer fuel poorly-visible from height (defer broadly-visible fuel).
- `decoy(E, tile)` — place a drainable object in enemy `E`'s view to pin its gaze away
  (strategy 4); admitted only when the true transition confirms the pin over the needed
  window.
- `endgame(launch_tile)` — the terminal (§5.6): absorb the Sentinel (driven through
  drain), build the platform synthoid, transfer on.

HTN methods compose these: *"win"* → *reach a launch-ready tile* → *endgame*. This is
the primary lever that makes the search tractable.

### 5.3 Managed exposure as hard energy feasibility over the window

For any macro spanning `[t, t+W]`, compute the drain by the **true transition** (step
the enemies `W` rounds with the player at its work tile) — not a heuristic. The macro
is feasible iff energy stays > 0 throughout and ≥ the next required cost; preferred if
the burst ends the drain. Zero-exposure macros trivially pass with zero drain. For
*pruning/ordering* only, use the gaze-timeline lower/upper drain bounds to reject
obviously-fatal macros before the full forward-sim.

### 5.4 Heuristics (admissible / consistent where possible)

Cost `g` = total elapsed rounds. `h` = max of admissible components (max preserves
admissibility):

- **h_height** — `max(0, launch_eye − eye) / max_height_per_macro × min_macro_rounds`.
- **h_energy** — if reaching launch needs ≥ `k` net energy the state can never bank
  (no reachable fuel), the state is dead → prune (∞).
- **h_platform** — a lower bound on rounds to get LOS-down to the platform once high
  enough.
- **h_safety** — from the gaze timeline: if the cheapest continuation must dwell where
  the drain over its window would exceed the buffer, add the unavoidable extra
  refuel/escape rounds. Admissible while hidden; optimistic once exposed (fine for a
  heuristic).

### 5.5 Search algorithm — default and escalation ladder

- **Default: weighted A\*** over macros with the §5.4 heuristic. Bounded-suboptimal
  (weight `w≈1.5`) trades a little plan cost for a much smaller frontier; the macro
  abstraction keeps branching single-digit on hidden routes.
- **Escalate on exposure-heavy nodes: ARA\*** (anytime repairing A\*) — emit a feasible
  plan fast at high `w`, then lower `w` to improve within the time budget.
- **Top tier on combinatorial exposure scheduling: deterministic UCT** — when a node's
  choice is *which* survivable-exposure schedule / decoy combination to run and no
  admissible heuristic bounds it tightly, run UCT with **guided (greedy, non-random)
  rollouts on the exact sim** for an anytime feasible answer with no variance.
- **Optional inner CSP: OR-Tools CP-SAT** for `build_batch`/`decoy` scheduling into the
  intersection of multiple sentries' gaze gaps with energy-over-window as a resource
  (Phase 4). A greedy interval fit suffices first; adopt CP-SAT only when the
  intersection search is a measured bottleneck.

Recommended default: **weighted A\* over macros + primitive A\* inside macros**, with
ARA\* and deterministic-UCT as the escalation fallbacks behind a node-branching
threshold.

### 5.6 Pruning, symmetry, budget, failure, and the endgame terminal

- **Dominance pruning.** State `a` dominates `b` if same player tile & eye, `a.energy
  ≥ b.energy`, `a.t ≤ b.t`, and `a`'s reachable object set ⊇ `b`'s → drop `b`.
- **Symmetry.** Object-slot permutations that yield identical geometry are
  canonicalized before hashing the closed-set key.
- **60 s/script budget** (CLAUDE.md). Parallelize the offline solve across processes;
  request an explicit authorized exception rather than silently exceeding.
- **Loud, explained failure.** When no survivable route fits the budget, report the
  tightest blocker — energy deficit `Δ`, no safe/survivable window at the required
  height, or no launch tile with down-LOS — never a silent degrade or a false "win".

**Endgame terminal (down-look launch from afar).** `endgame_ready(s)` and the win
sequence:

1. Eye strictly above the platform ground (fractional counts — a z8 terrain tile at
   eye 8.875 overlooks a z8 platform).
2. **Down-look LOS to the platform**, computed the *correct asymmetric way*: run
   `los.aim_target`/`landable_view` (body-pitch band) from the player's tile and eye,
   aimed **down** at the
   platform, honoring the ROM looking-up waiver (`$1D2E`) that makes a down-shot legal
   where the reverse up-shot is blocked. Do **not** reuse the symmetric platform-vantage
   `_launch_tiles` sweep (the under-count bug). Precompute the launch candidate set by
   testing the down-shot from each sufficiently high tile.
3. Fire: `absorb` the Sentinel — **driven through drain** (the sanctioned exposure: the
   absorb ends the drain as it lands) — then `create` a synthoid on the platform tile
   and `transfer` on. Win = `actions.on_platform` and not `actions.player_dead`.

The never-seen ls0 win is just the case where every preceding macro happened to be a
zero-drain hidden one.

### 5.7 Offline plan, live reschedule ("a missed aim is a crash")

- **Solve offline** to a complete, aim-exact plan: steps `{aim (h,v); verb; tile;
  expected memory delta}` — what `driver.sentinel_execute.perform_step` consumes.
- **Execute + resync each step** in the shared loop (`scripts/run_plan_*`). On live
  divergence (an unexpected meanie now that meanies are *modeled* — so divergence means
  the sim was wrong or a real timing slip — or a resync mismatch), **re-solve from the
  true observed live state** (the complete state, incl. the read-back PRNG where
  available).
- **Missed aim = crash.** The plan is aim-exact; a miss means the model diverged and
  must be investigated, never smoothed with margin.

---

## 6. Outstanding issues

### 6.1 The plan model to build (the strategy the search must reproduce)

A **deterministic, constructive planner** (everything — gaze precession, terrain, aim
geometry — is deterministic), not a cost-weighted search:

- **Gaze oracle as a hard constraint.** Precompute the Sentinel's gaze over time; a
  build/transfer/absorb is only ever scheduled onto a (tile, time-window) the gaze is
  provably off. No soft "exposure reserve" — a move into the gaze is illegal.
- **Distance-priced aim.** Replace the pan-only cost with a real aim cost dominated by
  pitch steepness, so the planner prefers far/shallow builds and absorbs; the move
  primitive is *build ≤2 boulders at a far tile → transfer → shallow look-back
  reabsorb*.
- **Height-first phases**, self-funded by the trail, fuel deferred by the visibility
  rule; launch from a far out-of-gaze tile with LOS to the plinth.
- **Offline plan, live re-schedule.** Solve the full plan offline; re-solve on live
  divergence (meanie, etc.). A **missed aim is a crash** — the offline plan is
  aim-exact, so a miss means the model is wrong and must be investigated.

### 6.2 Open items flagged by the design pass

- **T0.2 static-enemy→tile LOS cache** is only valid while no built object enters that
  enemy's ray; invalidate on any build inside an enemy sightline.
- **T3.1 exposure-required landscape id** needs a survey pass to pick a map with no
  fully-hidden route.
- **Live wins beyond the offline/sim ls0 win** — the T5.x live gates (recorded ls0 win
  with no missed aim, then a multi-enemy landscape) remain.

Both open items are handled inside their own task gates (§6.3), not blockers to
earlier phases.

### 6.3 Implementation plan (the "how")

Mechanically executable task plan for the design above. Each task is ~1 PR. Repo
conventions apply: numpy-first / numba where the surrounding code is, black + pylint
(no unused imports/vars), `pytest -n auto`, golden-fixture validation, **60 s hard CPU
budget per script**. The solver lives in `solver/` as small modules
(`astar_planner.py` + `macros.py` over `plan_game.py`, with `cost.py` / `launch.py` /
`gaze.py` / `search_node.py`); it **replaced** the removed legacy planner.
`sentinel/` is the transition function, modified only where a task explicitly adds a
helper.

Global config defaults (module constants in `solver/astar_planner.py`, all
env-overridable): `W_ASTAR=1.5`, `MAX_H_PER_MOVE=2.0`, `MIN_MOVE_ROUNDS=290.0`,
`T_BUCKET=64`, `NEXT_COST_FLOOR=3`, `HORIZON=4000`, `NODE_BUDGET=20000`,
`T_BUDGET_S=45.0`, `BEAM=8`, `BRANCH_HIGH=24`, `SAFETY_HORIZON=256`. `MAX_H_PER_MOVE`
keeps `heuristic` admissible; if A\* is too slow, raise `W_ASTAR` before touching it.
The enemy-round scalars (`ROUNDS_PER_*`) live in `solver/cost.py` against the
`actioncost` calibration; a live divergence recalibrates them from telemetry (T5.1), not by adding
safety margin.

**Status:** Phases −1..2 have **landed** — P-1 through T2.5 are done: the weighted-A\*
macro planner (`astar_planner.py` + `macros.py`) over `plan_game.py`, the cost/gaze/
launch/search-node modules, and the `run_plan_simulated` offline-plan + replan wiring
all exist and win ls0 offline and against the tick-accurate simulator. T6.0 is also
done — the legacy height planner and the superseded geometric-visibility sweep
family in `los` and its `plan_game` wrappers are removed (the current buildability
oracle is the ROM-faithful `los.landable_*` family; geometric visibility is
`threat.player_sees_tile`). The still-open items are the exposure/decoy/escalation phases
(T3.x/T4.x), the live gates (T5.x), and the faithful-oracle perf-and-search work noted
in §2.

| ID | Title | Files | Deps | Gate |
|----|-------|-------|------|------|
| **P-1** | Stateful meanie + two-probe exposure byte | `sentinel/enemies.py`, `sentinel/relative.py`, `golden_meanie.json`, `golden_exposure.json` | — | golden 0-divergence vs ROM |
| T0.1 | Search node + closed-set key | `solver/search_node.py` | P-1 | unit: clone_apply == direct sentinel ops |
| T0.2 | Gaze timeline oracle | `solver/gaze.py` | P-1 | agrees with `ticks_until_seen` t=0, all ls0 tiles |
| T0.3 | Move cost model | `solver/cost.py` | — | matches `execute_step` settle for scripted seq |
| T1.1 | Down-look launch enumeration + terminal test | `solver/launch.py` | T0.1 | ls0 set ⊇ human launch tile, ∌ dead-end corner |
| T1.2 | Managed-exposure feasibility over window | `solver/cost.py` | T0.1,T0.3 | energy delta over window == sim exactly |
| T2.1 | Climb macros (hop + boulder-step) | `solver/macros.py` | T0.1,T0.3,T1.2 | unit: macro reproduces the intended eye/energy |
| T2.2 | Refuel macro (visibility-deferred) | `solver/macros.py` | T2.1 | refuel emitted only when energy-blocked |
| T2.3 | Endgame macro (down-look, drive-through) | `solver/macros.py` | T1.1,T2.1 | from launch state, endgame wins in sim |
| T2.4 | Weighted-A\* driver + heuristic + failure contract | `solver/astar_planner.py` | T2.1,T2.2,T2.3 | `plan(0)` won, ≤ ~12 macro steps |
| T2.5 | Wire `run_plan_simulated` to offline-plan + replan | `scripts/run_plan_simulated.py` | T2.4 | `run_plan_simulated.py 0` WON, **zero drain** |
| T3.1 | Exposed-but-survivable climb variants | `solver/macros.py` | T2.4 | no-hidden-route map WON, energy>0 all windows |
| T3.2 | Decoy macro (gaze pin) | `solver/macros.py` | T3.1 | sim shows pinned enemy stops precessing over window |
| T3.3 | Multi-sentry gap intersection + greedy batch fit | `solver/gaze.py`, `solver/macros.py` | T3.1 | batch fits intersection; drain within buffer |
| T4.1 | ARA\* anytime wrapper | `solver/astar_planner.py` | T2.4 | dense map: feasible plan within T_BUDGET |
| T4.2 | Deterministic UCT escalation tier | `solver/uct.py` | T2.4,T4.1 | high-branch node: UCT returns feasible plan |
| T4.3 | Multiprocess offline solve | `solver/astar_planner.py` | T2.4 | solve within 60 s (or authorized exception) |
| T4.4 | Loud-failure contract + unsolvable test | `solver/astar_planner.py`, `solver/tests/test_failure.py` | T2.4 | unsolvable → structured failure, nonzero exit, no false win |
| T4.5 | (optional) CP-SAT inner scheduler | `solver/schedule_cpsat.py` | T3.3 | parity with greedy; solves a greedy-miss case |
| T5.1 | Wire `run_plan_live`; missed-aim-is-crash | `scripts/run_plan_live.py` | T2.5 | recorded live ls0 win, no missed aim |
| T5.2 | Live multi-enemy win | (config/validation) | T5.1,T3.x | recorded live win, multi-enemy landscape |
| T6.0 (done) | Delete the legacy planner + superseded sweep family, update imports/tests | `solver/`, callers | T2.5 | `pytest -n auto` green; no reference to the legacy planner remains |

**Phase -1 — Meanie substrate (prerequisite).** A subagent made the meanie lifecycle
+ the two-probe exposure byte bit-exact and ROM-validated. The planner depends only on
the post-conditions: `sentinel.enemies.step(state)` **statefully** advances any meanie
(spawn from a tree, rotate/move toward the player, forced hyperspace, expiry) as a side
effect on `state`, so a node that clones a `Game`/`PlanGame` and calls `step` sees
meanies evolve exactly as the live game; meanies occupy normal object slots with
`obj_type == mm.T_MEANIE (4)`; forced hyperspace / drain-death are observable via
`actions.player_dead`/`on_platform`; the full/partial exposure classification driving
spawn is correct. Gate: `golden_meanie.json` + `golden_exposure.json`, 0-divergence vs
the py65 ROM oracle over hundreds of rounds.

**Phase 0 — Complete-state search substrate.**
- **T0.1** `solver/search_node.py`: a `Node` wrapping a `PlanGame` (reuses its
  `col`/`eye`/`steps` tracking + bit-exact `create`/`absorb`/`transfer`) plus world
  tick `t` and sights heading `(vh, vv)`. `enemy_phase_hash(state)` = each enemy's type
  + bearing>>3 + mid-cooldown flag + which meanies exist; `node_key` =
  `(player_xy, round(eye,3), energy, t//T_BUCKET, enemy_phase_hash)`. Two nodes with the
  same tile/eye/energy, coarse tick bucket, and coarse enemy phase continue identically
  enough to dedup. Gate: build from `PlanGame(0)`, clone, apply a `create`+`transfer`
  two ways (node's `PlanGame` vs raw `sentinel.actions` on a parallel `State`); assert
  equal energy/tile/eye and `node_key` stability across `.clone()`.
- **T0.2** `solver/gaze.py` — `GazeTimeline`: forward-sim a passive clone; record
  `facings[e][t]` (an `(n_enemy, horizon)` uint8 array, numpy-first) and positions;
  `seen_at`, `safe_windows`, `is_safe`, `ticks_until_seen` as an admissible heuristic /
  pruning filter (not correctness). `FOV_HALF = enemies.FOV_SCAN // 2 = 10`. Static
  enemy→tile terrain-LOS reuses `relative.can_see_object` against a phantom robot (the
  `threat._place_phantom` placement) with the facing gate dropped, intersected with the
  per-tick cone. Gate: for `Game.new(0)`, every tile's
  `GazeTimeline.ticks_until_seen(x,y,0)` equals `threat.ticks_until_seen(state, x, y,
  horizon=256)`, 0 mismatches.
- **T0.3** `solver/cost.py` — elapsed enemy-rounds per aim/action from keyboard
  geometry (`aim_rounds` / `move_rounds`, using only
  `aimcost` + `actioncost`). `ROUNDS_PER_H_STEP=16`, `ROUNDS_PER_V_STEP=8`,
  `ROUNDS_PER_UTURN=16`. `move_rounds(g, T2, use_boulder, n_boulders, view, vh, vv)`
  prices a climb macro (aim + build + transfer + look-back reabsorb), each fired verb by
  `actioncost.SETTLE`, a stacked create adding `STACK_CREATE`, and returns
  `(rounds, end_h, end_v)`. Gate: for a scripted `create→transfer→absorb`, `move_rounds`
  equals the sum of `actioncost.action_rounds` `execute_step` would apply (±1), for a hop
  and a 1-boulder step on ls0.

**Phase 1 — Managed-exposure feasibility + down-look launch.**
- **T1.1** `solver/launch.py`: `launch_tiles(state, plat, plat_ground)` = every tile
  from which a robot at its terrain height has down-look LOS to the platform
  (phantom-place a robot, aim at plat via `los.landable_view(clone, plat, slot,
  eye_z=tile_eye)`); `down_look_los(g, plat)` ceils a fractional eye (a fractional eye
  above plat_ground sees down onto it); `endgame_ready(g, plat, plat_ground) = eye >
  plat_ground and down_look_los`. **Recovered human ls0 win** (from
  `out/play_20260704_171942.jsonl`): ladder `(8,17)@5.875 → (9,7)@6.375 → (10,23)@7.375
  → (9,3)@8.375 → (2,10)@9.375 → win`; **launch tile `(2,10)` @ eye 9.375**, **platform
  `(12,4)`** (ground 9), Chebyshev-10 apart ("launch from afar"). Gate:
  `launch_tiles(state, (12,4), 9)` contains `(2,10)` and excludes the tallest dead-end
  corner; `down_look_los` True from `(2,10)`@9.375; and the *symmetric* platform-vantage
  reverse sweep does **not** see `(2,10)` (proving the down-look asymmetry recovers it).
- **T1.2** `solver/cost.py` — `survivable(g_after_actions, from_tile, window_rounds)`:
  step the real world `window_rounds` (drain/rotate/meanie), then `ok iff not
  player_dead AND energy_after > 0 AND energy_after >= NEXT_COST_FLOOR (3)`. A macro
  spanning `W` from `from_tile` is feasible iff `energy_before − drain(W) >=
  NEXT_COST_FLOOR` and not dead, drain from the real `enemies.step` loop (0 for a hidden
  window). Gate: `survivable`'s `energy_after` equals `energy_before −
  threat.drain_over_window(state, W)` exactly; an insufficient buffer returns `ok=False`.

**Phase 2 — Macro-A\* wins ls0 (never-seen as the emergent cheapest plan).**
- **T2.1** `solver/macros.py` — `expand_climb(n, gaze)`: footholds from the ROM-faithful
  keyboard-aim buildability sweep (`los.landable_sweep_with_centres`); each buildable
  tile yields a hop and (if it climbs)
  a boulder-step, priced and feasibility-gated. The climb apply on a cloned `PlanGame`:
  legality gate 1 (buildability: aim / LOS
  / energy up front, wrapping `actions.can_create` + the on-boulder landable
  check), build boulders + capping synthoid (or a lone hop synthoid), transfer, look-back
  reabsorb of the departed shell if below eye and in LOS, then legality gate 2 (T1.2
  `survivable` over the whole window). Non-regression: skip footholds not above the
  current eye. Batch size = largest `n` fitting the build cap `eye +
  ROBOT_EYE_FUDGE(2)` that energy affords, ≤ 4. Gate: a single `expand_climb` child on
  ls0 reaches the expected eye/energy.
- **T2.2** `solver/macros.py` — `expand_refuel(n, gaze, need=NEXT_COST_FLOOR+2)`: emit
  refuel children only when energy would block the cheapest climb child; absorb
  below-eye, in-LOS fuel one per tile, topmost-first; skip broadly-visible fuel unless
  it is the only affordable source; advance the world by the aim+absorb window; require
  `survivable`. Gate: with energy ≥ `need`, returns `[]`; below `need` with a reachable
  tree, returns a child whose energy increased and whose window was survivable.
- **T2.3** `solver/macros.py` — `endgame_child(n, plat, plat_ground)`: requires
  `launch.endgame_ready`; `seye = ceil(eye)`; sweep from `seye` must see `plat`;
  drive-through absorb of the Sentinel; build platform synthoid; transfer on; return a
  node iff `on_platform and not player_dead`. Gate: from a hand-constructed launch-ready
  ls0 node, `endgame_child` returns a node with `on_platform is True`.
- **T2.4** `solver/astar_planner.py` — weighted-A\* driver. `heuristic(n, plat_ground)
  = ceil(max(0, (plat_ground+1) − eye) / MAX_H_PER_MOVE) × MIN_MOVE_ROUNDS` (admissible
  lower bound). `plan(...)` builds a `GazeTimeline`, pops by `f = cost +
  W_ASTAR×heuristic`, tries `endgame_child` at each pop (win → return `PlanResult`),
  folds `node_key` into a `closed` dict (skip re-reached-at-higher-cost), expands
  `expand_climb + expand_refuel`, keeps `energy>0 and t<HORIZON`, sorts by
  `(-eye, cost)`, pushes the top `BEAM`. `PlanResult(won, steps, failure, stats)`;
  `_diagnose` picks the tightest blocker (`no_launch_los` / `energy_deficit` /
  `no_safe_window` / `budget_exhausted`). Gate: `plan(0).won is True`, ≤ ~12 macro
  steps, < 60 s, energy trace pure build/absorb (no drain deltas) — the chosen plan is
  the hidden one.
- **T2.5** `scripts/run_plan_simulated.py` — the offline-plan + replan loop (it now
  prices the aim pan via `solver.cost.aim_rounds`): `plan()` offline from the resynced
  world → execute the plan's `steps` via
  `execute_step` (advancing the world by the real `actioncost` rounds) → on any diverging
  outcome, resync (`PlanGame.from_mem(world.mem, ...)`) and re-`plan()`. Keep
  `execute_step`, `advance`, `make_world`, and the `player_dead`/`won` checks unchanged.
  **Headline gate:** `run_plan_simulated.py 0` prints **WON** with **zero drain** (every
  energy change is a build cost or absorb gain), within 60 s.

**Phase 3 — Exposure-required landscapes + decoy.**
- **T3.1** `expand_climb` variants: admit a foothold whose window is *seen* when
  `survivable` holds, rank exposed children by `energy_after` (bigger surviving buffer
  first), add a `prefer_end_drain` bonus for a burst that absorbs its drainer within the
  window. Gate: on a map with no fully-hidden route, `plan().won is True` and min energy
  over the plan ≥ 1.
- **T3.2** `expand_decoy(n, gaze)`: place a drainable object in enemy `E`'s current
  view so `E` locks on and stops precessing; verify the pin through the true transition
  (`E`'s facing unchanged over the window, `E`'s targeted object == decoy slot).
  **Measured pin (ls0):** a synthoid held in view kept the enemy's facing constant for
  **~591 rounds** (1 rotation in 800) vs an idle enemy rotating every ~200; the pin is
  bounded by decoy *longevity* (drain downgrades robot→boulder→tree→gone), not a
  cooldown — ample for any ~200–300-round build window; re-confirm the actual window
  through the true transition per plan. Gate: build a decoy in an enemy's view, step `W ≥
  300`, assert the enemy's `obj_h_angle` unchanged and targeted slot == decoy, and a work
  tile in the pinned direction is now `survivable` where it was not.
- **T3.3** `GazeTimeline.safe_windows_intersection` = intersection over enemies of each
  enemy's gaze gaps; `fit_batch(gaze, from_tile, t0, action_windows)` greedily places
  each action's `[t, t+w)` inside one intersected safe window (choose `t0` just after the
  gaze sweeps past for max slack). `expand_climb` uses `fit_batch` to size the largest
  surviving batch on multi-sentry maps. Gate: on a two-sentry landscape, every action
  window is inside the intersection and the whole macro is `survivable`.

**Phase 4 — Escalation, budget, failure.**
- **T4.1** ARA\* wrapper: on `NODE_BUDGET` without a goal, run weighted A\* at `W=3.0`,
  emit the first feasible plan, then decrement `W` by 0.5 (down to 1.0) reusing the
  inconsistent set until `T_BUDGET_S=45` elapses; `plan(anytime=True)` selects it.
- **T4.2** `solver/uct.py` — per-node when `expand_*` yields `> BRANCH_HIGH=24`
  children: UCT over macros on the exact sim with guided greedy rollouts (rollout policy
  = the T2.4 `-eye, cost` order, no randomness), `ITER=2000`, UCB1 `c=1.4`; deterministic
  across runs.
- **T4.3** Multiprocess: shard root expansion (`ProcessPoolExecutor`, `os.cpu_count()`);
  each worker runs `plan()` seeded to a disjoint subset of root climb children; the
  coordinator returns the first/cheapest win, each worker within 60 s.
- **T4.4** Loud-failure contract: `PlanResult.failure = {reason ∈ {no_launch_los,
  energy_deficit, no_safe_window, budget_exhausted, dead_end}, blocker, detail}`. On
  `won is False` runners **log loudly and exit nonzero** — never a partial "progress"
  success. Gate: a constructed unsolvable landscape → `won is False`, correct `reason`,
  runner exits nonzero, **no false win** ever reported.
- **T4.5** (optional) `solver/schedule_cpsat.py` behind `SCHED_BACKEND=cpsat`: model
  batch/decoy scheduling as CP-SAT interval vars with `NoOverlap` + per-enemy forbidden
  windows (from `gaze`), energy-over-window as a resource; invoked only when `>1` enemy
  AND greedy `fit_batch` fails AND branching > `BRANCH_HIGH`.

**Phase 5 — Live.**
- **T5.1** `scripts/run_plan_live.py` mirrors T2.5 against the driver: `plan()` offline
  from the resynced live state (`driver.sentinel_state` read-back incl. PRNG where
  available) → execute via `driver.sentinel_execute.perform_step` → on divergence resync
  + re-`plan()`. **A missed aim is a hard crash**: if the memory-verify shows the aim
  landed on the wrong tile, raise and halt. Gate: a recorded live ls0 win (`$0CDE` bit 6)
  via `run_plan_live.py --digits 0000`, no missed aim.
- **T5.2** Run T5.1 on a multi-sentry / meanie landscape (relies on P-1 + T3.x). Gate:
  a recorded live win on a multi-enemy landscape.
- **T6.0** (done) Deleted the legacy height planner and the superseded
  geometric-visibility sweep family (in `los` and its `plan_game` wrappers), ported
  the still-used helpers into `solver/macros.py`, and updated every import/test. Gate:
  `pytest -n auto` green; no reference to the legacy planner remains.

### 6.4 Verified `sentinel/` API reference (checked against current code)

Node/macros call exactly these (signatures confirmed in the cited files):

- `sentinel.game.Game.new(landscape_number)` / `.clone()` / `.step_enemies()` /
  `.player_xy()` / `.platform_xy()` / `.energy` / `.won()` — `sentinel/game.py`.
- `sentinel.enemies.step(state)` (one round: `tick_cooldowns` + `update_enemies`;
  statefully advances rotation, drain, downgrades, **meanie lifecycle**, discharge) /
  `enemies.enemy_slots(state)` / `enemies.FOV_SCAN` (=0x14) — `sentinel/enemies.py`.
- `sentinel.actions.create(state, otype, tile) -> slot|None` / `absorb(state, slot) ->
  bool` / `transfer(state, slot) -> bool` / `can_create` / `can_absorb` / `on_platform` /
  `won` / `player_dead` / `win(state, tile=None) -> bool` — `sentinel/actions.py`.
- `sentinel.threat.ticks_until_seen(state, x, y, horizon=256, object_top=ROBOT_EYE) ->
  int` / `is_exposed` / `exposed_tiles` / `gaze_distance(state, tiles) -> {tile:0..128}` /
  `meanie_safe(state, tile) -> bool` / `drain_over_window(state, ticks) -> int` /
  `player_sees_tile(state, tile, observer_slot, eye_z=None) -> bool` (the ROM geometric
  march — geometric player→tile visibility, replacing the removed `los` visibility
  helper) — `sentinel/threat.py`.
- `sentinel.los.aim_target(state, h_angle, v_angle, cur_x, cur_y, player_slot,
  eye_z=None, max_steps=20000, return_centre=False) -> (tx,ty,los[,centre])` (always
  bit-exact vs the ROM) — the current **ROM-faithful buildability oracle**
  `landable_views(state, slot=None, eye_z=None, max_steps=6000)` /
  `landable_sweep_with_centres(state, slot=None, eye_z=None, max_steps=6000,
  v_band=False) -> (views, centres)` / `landable_view(state, tile, slot=None,
  eye_z=None, max_steps=6000, v_band=False)`, which sweep the sights cursor at 1 px
  resolution (`CURSOR_CX` / `CURSOR_CY`, and `CURSOR_CX_FULL` / `CURSOR_CY_FULL` for
  the full ROM cursor range `cx∈[16,143]`, `cy∈[32,159]`) — constants `SIGHTS_CX=0x50`,
  `SIGHTS_CY=0x5F`, `AZIMUTH_STEP=8`, `PITCH_BAND` — `sentinel/los.py`. A `view` is
  `{"h_angle":int,"v_angle":int,"cursor":[cx,cy]}`.
- `sentinel.aimcost.bearing_to` / `angle_dist(a,b) -> 0..128` / `h_steps` / `v_steps` /
  `h_press_count(h0,h1) -> (n_uturn,n_step)` /
  `bearing_rounds(h0,h1,rounds_per_step,rounds_per_uturn) -> float` / `pan_steps` —
  `sentinel/aimcost.py`.
- `sentinel.actioncost.action_rounds(mem, verb, view, stacked=False) -> float` /
  `is_stacked(mem, tile) -> bool` / `SETTLE={"absorb":190,"create":290,"transfer":300}` /
  `STACK_CREATE=285` — `sentinel/actioncost.py`.
- `sentinel.state.State.clone()` / `.energy` / `.player` / `.player_xy()` /
  `.platform_xy` / `.free_slots()` / `.occupied_slots()` / `.is_empty(slot)` /
  `.slot_of_type(otype)` /
  `.obj_x/obj_y/obj_z_height/obj_z_frac/obj_h_angle/obj_type/obj_flags` (indexable views)
  — `sentinel/state.py`.
- `solver.plan_game.PlanGame(landscape)` / `.from_mem(mem,...)` / `.clone()` /
  `.create(otype, tile, view, note="") -> slot|None` / `.absorb(slot, view, note="")` /
  `.transfer(slot, note="")` / `.feasible(otype, tile) -> bool` / `.top_of(tile) ->
  float|None` / `.player_xy()` / attributes `.state .mem .col .eye .steps .free .player
  .energy .plat .plat_ground .sentinel_slot .native_won` — `solver/plan_game.py`. Module
  helpers `terrain_z` / `cheb`. (Buildability now routes through the `los.landable_*`
  oracle directly; geometric player visibility through `threat.player_sees_tile` —
  `PlanGame` no longer wraps its own visibility/sees/centre-view helpers.)
- `sentinel.memmap` for the closed-set key: `ENEMIES_UPDATE_COOLDOWN=0x0C30`,
  `OBJECTS_H_ANGLE=0x09C0`, `T_MEANIE=4`, `T_SENTINEL=5`, `ENEMY_TYPES=(1,5)`,
  `ENERGY_IN_OBJECTS`, `PRND_STATE=0x0C7B`, `NUM_SLOTS=64`.
- Runner `scripts/run_plan_simulated.py`: `execute_step(world, stp, heading, budget,
  log)`, `advance(world, rounds, budget)`, `make_world(landscape)`, resync via
  `PlanGame.from_mem(world.mem, ...)`.

---

## 7. Simulator prerequisite recap (Phase -1)

A general planner cannot forecast meanies until the meanie lifecycle is a complete,
bit-exact, golden-validated stateful mechanic — it depends on the also-approximated
two-probe exposure byte (`$0014` full vs partial), which is exactly the signal the
meanie spawn predicate keys on (spawn requires the player seen **partially**). The work
reverse-engineered and validated, golden-fixture, against the py65 ROM oracle:

1. **Exposure two-probe bit-plumbing** (`$0014` full/partial) for rotated-angle
   targets; gates draining (full) and meanie spawn (partial). `golden_exposure.json`.
2. **Spawn predicate** (`consider_creating_meanie $197D`): an enemy sees the player
   partially, a tree within 10 tiles in both axes that the enemy fully sees and that can
   itself see the player becomes a meanie owned by the enemy.
3. **Meanie movement/rotation** (`update_meanie $16F2`): rotate `MEANIE_ROTATE_STEP`
   toward the player each update until facing a player it can see; cooldowns; "target
   gone → remove".
4. **Forced hyperspace + PRNG destination** (`do_hyperspace $2147`): meanie sees the
   player → forced hyperspace; new robot lands on a PRNG-chosen flat tile ≤ the player's
   height; too little energy → death; from the platform → win.
5. **Expiry/removal** (`remove_meanie $1754`).

Gate: `golden_meanie.json` capturing enemy+meanie trajectories over hundreds of rounds
on meanie-spawning landscapes, 0-divergence vs the py65 ROM oracle. (Status: landed —
see [simulator.md](simulator.md).)

---

## 8. Superseded design — receding-horizon best-first lookahead

This is the earlier design (`SEARCH_REDESIGN.md`), **superseded** by the deterministic
constructive planner above (§4–§6). It is retained for its reasoning — several of its
findings (why not minimax, the real move-cost model, enemy modelling as annotation) are
still load-bearing and were carried forward. The removed legacy height planner
(formerly under `solver/`) implemented it; it has since been deleted (task T6.0).

### 8.1 Why the greedy climb failed

The predecessor greedy planner picked the single best-looking foothold each call (by
height, then Sentinel-gaze distance, then edge distance) and committed immediately, with
no way to know before committing whether a foothold leads anywhere further — it
discovered a dead end only after arriving, its only recourse a "reposition" fallback that
would happily move to a *lower* tile. Confirmed against the recorded human ls0 win, which
never lost height across 5 real moves (5.875 → 6.375 → 7.375 → 8.375 → 9.375 → 9.875 win)
because every landing tile's bare terrain was one full unit higher than the last. Greedy,
with no lookahead, structurally cannot guarantee that; only a search that looks ahead
before committing can.

### 8.2 Why not adversarial minimax

The re-sentinel project already ran this experiment
(`../re-sentinel/disasm/SOLVER_ALGORITHMS.md`): their minimax planner (depth-limited
negamax with a worst-case enemy band, ~930 LOC, 6 eval weights) is **beaten** on all
three benchmark landscapes by a ~30–40 line closure algorithm with no adversarial layer,
at ~1/100th the runtime. The Sentinel and sentries are **fixed-rate automata** —
`step_enemies` advances rotation and cooldowns as a pure function of tick count, and
nothing in it branches on what the player does. There is no opponent making choices; a
MAX/MIN alternation models an adversary that isn't there and pays search cost for it.
Their "closure" alternative doesn't transfer directly, though: it assumes free
arbitrary-height stacking from one base tile and terrain-only LOS that objects never
occlude — neither holds in the keyboard-real pipeline (LOS walks the live object stack;
each boulder-step needs its own keyboard-reachable centre-aim). The right shape is
**best-first / A\* over the real reachable states**, with enemy timing folded in as a
per-candidate safety annotation.

### 8.3 Receding-horizon best-first, not one giant search

"20–30 moves for a hard landscape" is the length of the final *plan*, not a search depth
to explore exhaustively — branching is dozens of candidates per node, so a naive
full-width search to depth 20 is combinatorially impossible. Instead: at each real
decision point, run a bounded lookahead of depth `D` (default 3, raised as needed), score
the leaves, and commit to the first move of the best-scoring line; re-run from the new
real state for the next decision — the pattern a chess engine uses per move. `D=3` for ls0
matches the finding that a human needed only ~5 real moves.

### 8.4 State, successors, enemy modelling

Reuse the native `Game` as the search state (it IS the keyboard-faithful model) plus a
cheap `clone()` so branches don't mutate each other. Successor generation reuses the real,
validated functions (`_candidates` for hop/boulder-step footholds, `_boulder_centre_
feasible` to gate boulder-steps on an actual keyboard-reachable centre-aim — the check
that rejects most raw candidates, so filter *before* expanding — and `_refuel`,
one-object-per-tile to avoid the stale-aim batching bug).

Enemy modelling is **annotation, not adversary**. The unused-by-greedy enemy-dynamics
module already has: deterministic per-tick rotation/cooldown advance;
`exposed_within_window(state, phase, x, y, ticks)` (an O(ticks) forward simulation — "is
this tile seen by any enemy in the next N ticks", the robustness the old minimax was
searching for, as a check not a branch); `ticks_until_seen` (safety margin, a tie-break /
soft penalty); and `meanie_spawn_threat`/`meanie_safe` (the exact ROM conditions for a
tree near the player converting into a meanie that forces a hyperspace — precisely an
observed live failure). Build an enemy phase once per real decision and advance it by the
estimated tick-cost of each simulated move as the lookahead descends. Use `meanie_safe` as
a hard pre-filter on footholds.

### 8.5 Tick-cost-per-move from the simulator's own aim geometry

Advancing the enemy state during lookahead requires how many rounds elapse per real move
— NOT a flat constant. A move is a keyboard sequence whose cost is dominated by how far
the view must **pan** to aim each sub-action, computed exactly from `sentinel.aimcost` +
the `sentinel.los` keyboard lattice. The legacy planner's move-cost function mirrored its
apply sequence (aim + build the foothold + transfer + swing the view back to look down and
reabsorb the departed shell) and prices each aim by lattice distance from the previous
heading. That return swing is often a near-180° bearing pan and is the bulk of a move's
exposure window — the cost the old flat `TICKS_PER_HOP`/`TICKS_PER_BOULDER_STEP` constants
ignored. Keystrokes convert to rounds by the ROM's scroll cadence (view scrolls one step
per frame ≈ one `enemies.step`): a ±8 bearing keystroke ≈ 16 rounds, a ±4 pitch keystroke
≈ 8 rounds, a fired action ≈ 16 rounds — the `ROUNDS_PER_H_STEP`/`ROUNDS_PER_V_STEP`/
`ROUNDS_PER_ACTION` tunables (env-overridable for live recalibration). That move-cost also
returned the ending heading, threaded into the next move so each pan is charged from where
the last left the view.

### 8.6 Search algorithm and validation

Best-first (not plain DFS/BFS) so promising lines are explored first and can cut off early:
sort candidates by a cheap heuristic (height gain, then enemy safety margin, then edge
distance), cap to `BEAM_WIDTH`, expand each on a clone, advance the enemy phase by the
move's tick-cost, recurse to depth `D`. Key properties greedy lacked: **no move is taken
unless it has a continuation within the horizon** (a node with zero non-dead-end
successors returns `-inf` and is pruned by its parent — the diagnosed bug fixed); and
`_boulder_centre_feasible`/`meanie_safe` are pre-filters before expansion, so the search
never sinks budget into a branch the keyboard can't execute. `evaluate()` at a cut leaf
scores eye height (primary), `ticks_until_seen` at the final tile (safety), edge distance,
and remaining energy. The refuel / boulder-centre-feasibility / candidate-enumeration /
foothold-eye / apply mechanics and the endgame terminal were unchanged; only the
*decision* function inside the resync-and-replan live loop changed. Validation: offline ls0
win with `D=3` that does not regress height at any step and closes near the human's ~5 real
moves; cross-check other landscapes, raising `D` only where needed; ROM-oracle validate a
real win (py65); live VICE recording as the final check — a divergence recalibrates
`TICKS_PER_ACTION` from the run's own log, not by adding static safety margin.

**Why it was superseded.** The receding-horizon design keeps a height objective with soft
costs and a static exposure mask, cannot express managed exposure as hard energy
feasibility, and models meanies only reactively. The constructive planner (§4–§6) replaces
the height objective with a complete-state feasibility search, demotes "never seen" to an
emergent optimum, prices aim by pitch, fixes the asymmetric down-look launch, and makes
meanies a validated stateful part of the state — the smallest design that wins *general*
landscapes with the ls0 win as its cheapest special case.
