# Planner redesign proposal: a general deterministic constructive solver

Status: proposal (no planner code written). Supersedes the cost-weighted
receding-horizon design in `SEARCH_REDESIGN.md`. Reader prerequisites: `README.md`
"Gameplay model" + "Strategy", `docs/outstanding-issues.md`, `docs/simulator.md`.

## 1. Executive summary

*The Sentinel* is a **single-agent, fully deterministic** game: given the complete
state — every sentry, the Sentinel, any meanies, the board, energy, **and the 40-bit
PRNG LFSR** (`sentinel.prng`, `PRND_STATE` $0C7B, already tracked in memory) — the
bit-exact simulator is a total transition function with *no* residual randomness.
Hyperspace destinations, created-object facings and drain-scatter placement are all
deterministic functions of that tracked PRNG. There is nothing to plan "under
uncertainty."

The correct general model is therefore a **forward search over the complete
deterministic state**, with `sentinel/` as the exact transition function. Exposure,
drain, gaze precession, meanie spawn/hunt and forced-hyperspace are not special
cases to veto or bolt on — they are consequences the search *observes through the
true transition*.

Within that model:

- **Exposure is a managed resource, not a veto** (README strategy 1). A productive
  action sequence may be run *through* drain when the banked energy covers the drain
  over the action window; the endgame absorb is deliberately *driven through* drain
  because it ends the drain as it lands. A fully-hidden ("never seen") route is not
  hard-wired — it is simply the **lowest-cost plan the search prefers when one
  exists**, and it falls out for ls0 for free. On dense maps with no hidden route the
  planner *chooses* survivable exposure, and can even *exploit* it (a decoy: an enemy
  locked on a drainable target stops precessing, holding its gaze away — strategy 4).

- **The hard problem is tractability**, and it is the centre of this document
  (§7–§8): a complete-state forward search over a 20–40-action horizon with several
  rotating enemies and dynamic meanies has a large branching factor. The design
  attacks this with an admissible-heuristic **A\*** over an **HTN-style macro-action**
  abstraction, managed-exposure encoded as **hard energy feasibility over the action
  window**, aggressive dominance/time-bucket pruning, and a defined **escalation
  ladder** (weighted A\* → ARA\* → UCT) when branching explodes — with a **loud,
  explained failure** when no survivable route fits the budget.

- **A named simulator prerequisite** (§6, Phase -1): the meanie lifecycle is today an
  approximation, not a ROM-validated stateful mechanic (there is no `golden_meanie`
  fixture, and `docs/simulator.md` flags it as a `meanie_threat()` capability query).
  A general planner *cannot forecast meanies* until this is a complete, bit-exact,
  golden-validated stateful lifecycle. That is prerequisite work, specified below.

**OSS verdict (revisited for the general regime, §5): still a custom solver on
`sentinel/` as the spine**, because the action model, preconditions and costs are the
game's bit-exact ray-march LOS / keyboard-aim / tick machinery — unencodable in PDDL
without pre-grounding the whole state-dependent transition. Two frameworks earn a
narrowed, genuine role in the *heavier* regime: **OR-Tools CP-SAT** as an optional
inner solver for exposure-scheduling into multi-sentry gaze-gap intersections, and a
small **custom UCT** as the anytime escalation tier — neither displaces the custom
spine.

**Phased-plan headline:** (-1) bit-exact stateful meanie lifecycle in `sentinel/`,
golden-validated → (0) complete-state search substrate (cloneable state incl. PRNG,
macro library, heuristics, pruning) → (1) managed-exposure feasibility + down-look
launch enumeration → (2) macro-A\* wins ls0 as the emergent zero-exposure cheapest
plan → (3) win an exposure-*required* multi-sentry/meanie landscape by running
through survivable drain (+ decoy) → (4) escalation tier + multiprocess within
budget, loud failure on the unsolvable → (5) live wins (ls0, then multi-enemy),
missed-aim-is-a-crash, meanie divergence handled by replan from true state.

---

## 2. What kind of problem this is (the formalism)

- **Single agent, non-adversarial.** `sentinel.enemies.step` advances rotation,
  targeting, draining, cooldowns, meanie spawn/hunt and forced-hyperspace as a pure
  function of the current state. Nothing *chooses* against the player; the enemies are
  fixed-rate automata. Minimax is the wrong model (`SEARCH_REDESIGN.md` §2, confirmed
  by the re-sentinel numbers): there is no MAX/MIN alternation to run.

- **Full determinism given the complete state.** The PRNG LFSR lives in tracked
  memory (`PRND_STATE`, `prng.load/store`), so the three "random" effects — hyperspace
  destination (`do_hyperspace` → `_put_object_in_random_tile_below_z`), created-object
  facing (`create` $1F83), drain-scatter tree placement (`_consider_discharging_enemy_
  energy`) — are deterministic functions of state. **There is no true nondeterminism.**
  The complete state is `(Game: board + all object slots incl. sentries/Sentinel/
  meanies + energy + player + PRNG bytes, world tick t)` and the transition is the
  real bit-exact sim.

- **Continuous time, discretized to enemy rounds.** Each keyboard action spans many
  rounds: `sentinel.actioncost.action_rounds` prices settle/redraw, `sentinel.aimcost`
  prices the aim pan. No mid-action pause — an action's whole window is exposed to
  whatever the enemies do during it. Time is a first-class plan dimension.

- **Energy as a managed, hard-feasible resource.** Energy 0..63; each verb a fixed
  cost. Drain is −1 per drain-cooldown *while seen*. Absorbing the drainer ends its
  drain. Feasibility is: energy > 0 throughout every action window, and ≥ the cost of
  the next required action — a real constraint the forward transition evaluates
  exactly, **neither a soft penalty nor a blanket veto** (README strategy 1).

- **LOS/aim preconditions.** `create`/`absorb`/`transfer`/`win` gate on a reachable
  near-centre sights aim (`los.centre_view`/`aim_target`) — exact, cheap oracles.

- **Deterministic gaze precession → schedulable windows.** Idle enemies precess a
  fixed step every ~200 rounds over a ±10 cone. While the player and its objects are
  *unseen*, enemies stay idle, so their facing is a **plan-independent timeline** —
  but only under that condition. This timeline is retained as a **heuristic / pruning
  device** (a valid safety lower bound while hidden), *not* a correctness crutch (§7).

- **Height-climb subproblem + long horizon** (20–40 actions), as before.

**Formalism.** A **shortest-path / cost-minimizing forward search over the complete
deterministic state graph**, transition = the bit-exact sim, with energy feasibility
over action windows as a hard resource constraint and gaze precession as a heuristic.
Rendered tractable by macro-action (HTN) abstraction and admissible heuristics (§7).

Why the alternatives still don't fit as the *spine*: minimax (no adversary); MCTS/RL/
GGP/POMDP (built for uncertainty/opponents we don't have) — though a *deterministic*
UCT re-enters as an anytime escalation tier for large branching (§5, §7). PDDL+/
numeric-temporal planners have the right shape but cannot host the geometry (§5).

---

## 3. Why the current planner structurally cannot win

`solver/climb_search.py` is a best-first **height** search with a stack of soft-cost
penalties (`RESERVE`, `EXPOSURE_RESERVE`, `SEEN_DRAIN`, `REFUEL_DRAIN`, pan cost,
launch-readiness, ~20 env weights). Its architecture fights the correct formalism at
every turn:

1. **Exposure priced as a tradeable soft cost / vetoed by a static mask.** It leans on
   `threat.is_exposed` (the "could ANY rotation ever see this" mask, facing gate
   dropped) which flags essentially every useful high tile, so it either excludes
   winning routes or buries them under reserves; and where it does allow exposure it
   scores it as a penalty rather than testing *survivability over the window*. The
   correct treatment is neither: it is a **hard energy-feasibility constraint over the
   true drained window** (§7.3), under which a hidden route is simply cheapest.
2. **Height is the objective**, not a precondition — producing timid +0.5 creeps and
   tall exposed towers instead of cheap terrain-height gains and, when needed,
   run-through-drain productive bursts.
3. **Pan-only aim cost** (no pitch pricing) → no preference for far/shallow builds.
4. **Symmetric launch LOS** (`_launch_tiles` sweeps from the platform vantage looking
   *up*) under-counts the far down-look launch tiles that actually win (`$1D2E`).
5. **Meanies unmodeled** — only avoided via the static mask or met reactively live.
6. **Only wins ls0 when the drain window is hand-cut ~2.5×** — the tell of a wrong
   model being tuned, not a strategy being executed.

You cannot tune a soft-cost height optimizer into a complete-state feasibility
search. Replace, don't patch.

---

## 4. The exposure reframe: managed resource, emergent zero-exposure

The keystone of the previous draft — a hard "never seen" invariant — is **demoted to
an emergent optimum**. The general rules:

- **Exposure is admitted whenever it is survivable.** An action (or macro) that spans
  window `W` from a currently-seen tile is legal iff the true forward transition keeps
  energy > 0 throughout `W` and ≥ the next required cost. Cheaper still if the burst
  *ends* the drain (absorbing the drainer). This is the timed-race math of README
  strategy 1 as a **hard constraint over the real drained window**, not a penalty.

- **Zero-exposure is the cheap common case, not a rule.** A hidden route incurs zero
  drain, so its energy ledger is pure build/absorb and its cost (elapsed rounds) is
  minimal. The search *prefers* it automatically where it exists (ls0). Nothing forces
  it; on maps without one, the search takes the cheapest *survivable* exposed route.

- **Exposure can be chosen and exploited.**
  - *Chosen*: dense multi-sentry maps where no fully-hidden climb exists — run a
    productive build burst through a bounded drain the buffer covers.
  - *Exploited (decoy macro)*: an enemy locked onto a drainable target **stops
    precessing** (`enemies.step`: it rotates only when idle). Placing a decoy
    object/robot in one enemy's view *pins its gaze away* from the player's work tile
    for a schedulable interval. The planner can build this as a first-class macro and
    verify its effect through the true transition.

- **The gaze timeline is a heuristic, not correctness.** While the player+objects are
  hidden, the precomputed idle-precession timeline is an exact lower bound on
  "ticks-until-seen" and a valid admissible safety heuristic and pruning filter. The
  *instant* the plan admits exposure or leaves a seen object (which perturbs enemy
  cooldowns/rotation), correctness comes only from the **true forward transition** —
  the timeline is then merely an optimistic guide that the real sim overrides.

---

## 5. OSS survey (revisited for the general, multi-enemy regime)

Verdict: **custom spine on `sentinel/`, with two narrowed framework roles.**

| Framework | What it is | Maturity | Fit for the GENERAL problem |
|-----------|-----------|----------|------------------------------|
| **Custom A\*/HTN on `sentinel/`** | Hand-rolled search using the sim as transition | n/a | **Spine.** Only thing that can host bit-exact LOS/aim/tick + state-dependent enemy & meanie transitions. |
| **OR-Tools CP-SAT** | Interval/no-overlap constraint scheduler with time windows | Very mature | **Optional inner solver — role grows in the general regime.** Once exposure-scheduling dominates (pack build/aim/decoy actions + energy-over-window into the *intersection* of several sentries' gaze gaps), this is a real CSP CP-SAT solves well. It consumes windows/costs the sim supplies; it does not host the geometry. Adopt for the build-into-window sub-problem on dense maps (Phase 3/4). |
| **Custom UCT (deterministic)** | Small in-repo MCTS/UCT | n/a | **Escalation tier (revised from the earlier "mismatch" verdict).** With macro abstraction the branching is usually small enough for A\*; but on dense exposure-scheduling nodes where no admissible heuristic bounds the trade-offs tightly, a *deterministic* UCT with guided (non-random) rollouts on the exact sim gives an anytime feasible plan without variance. Not a framework dependency — ~100 lines on the same state. |
| **unified-planning / ENHSP / OPTIC / Fast Downward / pyperplan** | PDDL / numeric-temporal / PDDL+ planners | ENHSP & UP active (UP 1.0, SoftwareX 2025); FD mature | **No (spine).** The geometry and the *state-dependent action set* (meanies appear/expire; LOS depends on the live object stack) have no faithful declarative encoding — you would ground the reachable transition from the sim per state, i.e. rebuild the custom search. PDDL+ *processes* (ENHSP) could model gaze/energy as timed numeric fluents, but not the ray-march LOS. Remains the "strongest alternative" only if the custom search proves unsteerable (§13). |
| **MCTS libs** (mcts-simple, hildensia/mcts) | Generic UCT hobby repos | Small | **No.** Prefer a purpose-built deterministic UCT on our state; generic libs add glue and unmaintained risk. |
| **pomdp-py** | POMDP/MDP toolkit | Maintained | **No.** Fully observable + deterministic; no belief state. |
| **networkx** | Graphs + A\* | Very mature | **Utility only.** Our state graph is lazily generated with energy+tick+object labels; a hand-rolled label-correcting A\* is cleaner. Borrow patterns, not the framework. |
| **OpenSpiel / Gymnasium+RL** | GGP / RL | Mature | **No.** For learning under uncertainty/opponents; mismatch. |

**Why custom still wins the spine.** The three questions a planner must answer —
is a tile visible from an eye at height `z`; how many rounds does a keyboard action
cost; what does the full enemy+meanie+PRNG state do next — are *already implemented
and ROM-validated* in `sentinel/` (`los`/`los_jit`, `actioncost`+`aimcost`,
`enemies`/`threat`/`prng`). A framework buys search control you must feed
hand-grounded facts anyway. The heavier regime *grows* the case for CP-SAT as an
inner scheduler and for a deterministic-UCT escalation tier, but not for replacing
the custom spine.

---

## 6. Simulator prerequisite (Phase -1): a bit-exact, stateful meanie lifecycle

**Honest status.** `docs/simulator.md` lists the meanie lifecycle as a *known
approximation* — exposed via `enemies.meanie_threat()` as a capability query, "not a
stateful side effect" — and there is **no `golden_meanie.json`** fixture (confirmed:
`sentinel/tests/` has golden_prng/los/actions/landscape/relative/enemies only).
`enemies.py` already contains partial machinery (`_consider_creating_meanie`,
`_update_meanie`, `do_hyperspace`), but it is **not validated against the ROM oracle**,
and it depends on the also-approximated **two-probe exposure byte** (the full/partial
`$0014` classification `docs/simulator.md` admits may be wrong for rotated-angle
targets) — which is *exactly* the signal the meanie spawn predicate keys on (spawn
requires the player seen **partially**). A general planner literally cannot forecast
meanies until this is complete and validated. This is prerequisite, not optional.

**What must be reverse-engineered / modeled and validated (golden-fixture, like every
other mechanic):**

1. **Exposure two-probe bit-plumbing** (`$0014` full vs partial). Nail the multi-probe
   accumulation so full/partial matches the ROM for rotated-angle targets. This gates
   both draining (full) and meanie spawn (partial). Validate a new
   `golden_exposure.json` sampling many (enemy, target, angle) triples against py65.
2. **Spawn predicate** (`consider_creating_meanie` $197D): enemy sees the player
   *partially*, a tree exists within 10 tiles in both axes that the enemy fully sees
   (two-screen FOV) and that can itself see the player → that tree becomes a
   meanie owned by the enemy. Already coded; must be *validated* stateful (the tree's
   slot flips type and the enemy records ownership).
3. **Meanie movement/rotation** (`update_meanie` $16F2): the meanie rotates
   `MEANIE_ROTATE_STEP` toward the player each update until it faces a player it can
   see. Confirm the step, cooldowns (`UPDATE_COOLDOWN_MEANIE_*`), and the
   "target's object gone → remove_meanie" transitions.
4. **Forced hyperspace + PRNG destination** (`do_hyperspace` $2147): meanie sees the
   player → forced hyperspace; the new robot lands on a PRNG-chosen flat tile ≤ the
   player's height (`_put_object_in_random_tile_below_z`, deterministic given
   `PRND_STATE`); too little energy → death; from the platform → win. Validate the
   *destination tile* and PRNG advance against the ROM, since a general planner may
   have to reason about (or deliberately avoid) forced-hyperspace outcomes.
5. **Expiry/removal** (`remove_meanie` $1754 / `_remove_meanie_and_reset_enemy`).

**Validation gate.** A new `golden_meanie.json` capturing enemy+meanie array
trajectories over hundreds of rounds on landscape(s) that actually spawn meanies,
0-divergence vs the py65 ROM oracle — the same bar as `golden_enemies.json`. With this
in place, **meanies are just enemies in the state**; the planner needs no meanie
special-case at all.

---

## 7. Tractability — the central design problem

A complete-state forward search over ~20–40 actions with several rotating enemies and
dynamic meanies has a large state space and branching. This section is the core of the
design.

### 7.1 State, transition, and the closed set

- **State** = `sentinel.Game.clone()` (bit-exact, incl. PRNG) + world tick `t`.
  Everything — player tile, eye, energy, all enemies/meanies, PRNG — is in the clone;
  enemy facing is derived, not carried separately.
- **Transition** = the real sim: apply the macro/primitive's keyboard actions, stepping
  `enemies.step` for the priced round cost so drain/rotation/meanie dynamics are exactly
  what the live game does over that window.
- **Time-bucketed closed set.** Key `(player_tile, round(eye,3), energy,
  t // T_BUCKET, enemy_phase_hash)` where `enemy_phase_hash` summarizes each enemy's
  facing/cooldown coarsely. `T_BUCKET` coarsens schedule equivalence. This collapses
  states that differ only in immaterial micro-timing.

### 7.2 Macro-action (HTN) abstraction — collapse the horizon

Search at the level of **parametric macros**, not keystrokes, so the effective horizon
drops from 20–40 to ~5–8 decisions. Each macro is a scripted parametric sequence whose
internal aims/tiles are solved by a small bounded primitive search and whose feasibility
(LOS, energy-over-window, safety) is checked through the true transition:

- `climb_to(T)` — reach launch/foothold tile `T` via the cheapest hop/step chain
  (terrain hops preferred; boulder steps priced `2n`).
- `build_batch(T, n)` — stack `n` boulders + capping synthoid at `T`, phased into a
  safe/ survivable window (§7.4).
- `refuel(region)` — absorb below-eye fuel, generated only when energy would otherwise
  be infeasible; prefer fuel poorly-visible from height (defer broadly-visible fuel).
- `decoy(E, tile)` — place a drainable object in enemy `E`'s view to pin its gaze away
  (strategy 4); admitted only when the true transition confirms the pin over the needed
  window.
- `endgame(launch_tile)` — the terminal (§8): absorb the Sentinel (driven through
  drain), build the platform synthoid, transfer on.

HTN methods compose these: *"win"* → *reach a launch-ready tile* → *endgame*; *"reach
launch-ready"* → a sequence of `climb_to`/`build_batch`/`refuel`/`decoy`. This is the
primary lever that makes the search tractable.

### 7.3 Managed exposure as hard energy feasibility over the window

For any macro spanning `[t, t+W]`, compute the drain by the **true transition** (step
the enemies `W` rounds with the player at its work tile) — not a heuristic. The macro
is feasible iff energy stays > 0 throughout and ≥ the next required cost; preferred if
the burst ends the drain. Zero-exposure macros trivially pass with zero drain (the
common cheap case). For *pruning/ordering* only, use the gaze-timeline lower/upper
drain bounds to reject obviously-fatal macros before the full forward-sim.

### 7.4 Heuristics (admissible / consistent where possible)

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

### 7.5 Search algorithm — default and escalation ladder

- **Default: weighted A\*** over macros with the §7.4 heuristic. Bounded-suboptimal
  (weight `w≈1.5`) trades a little plan cost for a much smaller frontier; the macro
  abstraction keeps branching single-digit on hidden routes.
- **Escalate on exposure-heavy nodes: ARA\*** (anytime repairing A\*) — emit a feasible
  plan fast at high `w`, then lower `w` to improve within the time budget.
- **Top tier on combinatorial exposure scheduling: deterministic UCT** — when a node's
  choice is *which* survivable-exposure schedule / decoy combination to run and no
  admissible heuristic bounds it tightly, run UCT with **guided (greedy, non-random)
  rollouts on the exact sim** for an anytime feasible answer with no variance. Revised
  from the earlier flat "MCTS mismatch": with the enlarged branching it earns a place —
  as an escalation tier, not the spine.
- **Optional inner CSP: OR-Tools CP-SAT** for `build_batch`/`decoy` scheduling into the
  intersection of multiple sentries' gaze gaps with energy-over-window as a resource
  (Phase 4). Single sequential agent ⇒ a greedy interval fit suffices first; adopt
  CP-SAT only when the intersection search is a measured bottleneck.

Recommended default to build first: **weighted A\* over macros + primitive A\* inside
macros**, with ARA\* and deterministic-UCT as the escalation fallbacks wired behind a
node-branching threshold.

### 7.6 Pruning, symmetry, budget, and failure

- **Dominance pruning.** State `a` dominates `b` if same player tile & eye, `a.energy ≥
  b.energy`, `a.t ≤ b.t`, and `a`'s reachable object set ⊇ `b`'s → drop `b`.
- **Symmetry.** Object-slot permutations that yield identical geometry are canonicalized
  before hashing the closed-set key.
- **60 s/script budget** (CLAUDE.md). Parallelize the offline solve across processes —
  independent root macro choices / landscape shards run concurrently (numpy-first,
  numba-compatible per repo rules; LOS is already `los_jit`). If a landscape genuinely
  needs longer, request an explicit authorized exception rather than silently exceeding.
- **Loud, explained failure.** When no survivable route fits the budget, report the
  tightest blocker — energy deficit `Δ`, no safe/survivable window at the required
  height, or no launch tile with down-LOS — never a silent degrade or a false "win".

---

## 8. Endgame terminal (down-look launch from afar)

`endgame_ready(s)` and the win sequence:

1. Eye strictly above the platform ground (fractional counts — a z8 terrain tile at eye
   8.875 overlooks a z8 platform).
2. **Down-look LOS to the platform**, computed the *correct asymmetric way*: run
   `los.aim_target`/`sees_tile` from the player's tile and eye, aimed **down** at the
   platform, honoring the ROM looking-up waiver (`$1D2E`) that makes a down-shot legal
   where the reverse up-shot is blocked. Do **not** reuse the symmetric
   platform-vantage `_launch_tiles` sweep (the under-count bug). Precompute the launch
   candidate set by testing the down-shot from each sufficiently high tile.
3. Fire: `absorb` the Sentinel — **driven through drain** (this is the sanctioned
   exposure: the absorb ends the drain as it lands) — then `create` a synthoid on the
   platform tile and `transfer` on. Win = `actions.on_platform` and not
   `actions.player_dead`.

The endgame is one HTN macro; the never-seen ls0 win is just the case where every
preceding macro happened to be a zero-drain hidden one.

---

## 9. Offline plan, live reschedule ("a missed aim is a crash")

- **Solve offline** to a complete, aim-exact plan: steps `{aim (h,v); verb; tile;
  expected memory delta}` — what `driver.sentinel_execute.perform_step` consumes.
- **Execute + resync each step** in the shared loop (`scripts/run_plan_*`). On live
  divergence (an unexpected meanie now that meanies are *modeled* — so divergence means
  the sim was wrong or a real timing slip — or a resync mismatch), **re-solve from the
  true observed live state** (the complete state, incl. the read-back PRNG where
  available). The general planner replans, it does not hand-handle meanies.
- **Missed aim = crash.** The plan is aim-exact; a miss means the model diverged and
  must be investigated, never smoothed with margin.

---

## 10. How this addresses the review + outstanding-issues

| Concern | Resolution |
|---|---|
| "Never-seen" is a brittle special case | §4 demoted to an emergent optimum; exposure is a managed resource with hard energy-feasibility over the window; hidden routes are simply cheapest |
| Multi-sentry deferred to Phase 5 | §7 macros + gaze-gap *intersection*; CP-SAT inner scheduler; multi-sentry is a first-class case from Phase 3 |
| Meanies only avoided / reactive | §6 Phase -1 makes meanies a **modeled, ROM-validated stateful** part of the state; the search forecasts spawn/hunt/forced-hyperspace through the true transition |
| One "contained randomness" | §1/§2 PRNG is tracked ⇒ **fully deterministic**; search over complete state, no timeline as a correctness crutch |
| Tractability hand-waved | §7 HTN macros, admissible heuristics, dominance/symmetry/time-bucket pruning, weighted-A\*→ARA\*→UCT ladder, multiprocess, loud failure |
| Gaze as soft cost (old planner) | §4/§7.3 hard energy feasibility over the true drained window; gaze timeline is only a heuristic |
| Pan-only aim | §7.2 aim priced by pitch+bearing (`aimcost`); far/shallow wins |
| Missed far down-look launch tiles | §8 asymmetric down-shot LOS (`$1D2E`) |
| Wins only when drain hand-cut 2.5× | §7.3 exact drained-window transition; feasibility is real, not calibrated |

---

## 11. Complexity and tractability expectations

- **Meanie/exposure validation (Phase -1):** a few hundred py65 oracle rounds per
  fixture — one-time, sub-minute.
- **Per-node work:** one LOS lattice sweep (`los_jit`, ~11×) + a bounded primitive
  search inside the chosen macro + one true-transition forward-sim for the drained
  window. Dominance/time-bucket pruning keeps the frontier small; macro abstraction
  keeps depth ~5–8.
- **Branching:** single-digit on hidden routes (the safe-window filter prunes hard);
  grows on dense exposed maps → the escalation ladder (§7.5) caps the cost.
- **ls0** should close well inside 60 s as the emergent zero-exposure plan; dense maps
  use ARA\*/UCT anytime + multiprocess, or an authorized budget exception.
- **Live re-solve** runs the same search from the resynced true state.

---

## 12. Phased implementation plan (endpoint: general landscapes)

Each phase is a small, subagent-sized unit with a concrete **sim-first, then live**
gate. The never-seen ls0 win is an *early milestone that falls out of the general
machinery*, not a separate codepath.

- **Phase -1 — Stateful meanie lifecycle in `sentinel/`** (§6). Complete + validate the
  exposure two-probe byte, spawn, movement, forced-hyperspace-with-PRNG-destination,
  expiry. **Gate:** new `golden_exposure.json` + `golden_meanie.json`, 0-divergence vs
  the py65 ROM oracle over hundreds of rounds on meanie-spawning landscapes.

- **Phase 0 — Complete-state search substrate.** Cloneable `(Game incl. PRNG, t)` state,
  the macro-action library, heuristic components, dominance/symmetry/time-bucket closed
  set, gaze-timeline safety heuristic. **Gate:** the substrate's transition reproduces
  `scripts/run_plan_simulated.py`'s tick-accurate forward model exactly (differential
  test over scripted action sequences).

- **Phase 1 — Managed-exposure feasibility + down-look launch.** Drain-over-window from
  the true transition; asymmetric down-shot launch enumeration. **Gate:** energy trace
  over any exposed window matches the sim exactly; the ls0 launch set includes the
  human-win launch tile and excludes the dead-end corner.

- **Phase 2 — Macro-A\* wins ls0 as the emergent zero-exposure plan.** Weighted A\* over
  macros + primitive A\* inside; endgame terminal. **Gate:** `run_plan_simulated 0`
  reports WON; assert the found plan is zero-drain *because it was cheapest*, not
  because exposure was forbidden (verify by confirming exposed alternatives were legal
  but costlier).

- **Phase 3 — Win an exposure-*required* multi-sentry/meanie landscape.** Run a
  productive burst through survivable drain; `decoy` macro; gaze-gap intersection.
  **Gate:** WON on a landscape with no fully-hidden route, with the energy trace proving
  energy > 0 throughout every exposed window and the meanie forecast matching the sim.

- **Phase 4 — Escalation tier + budget discipline.** ARA\*/deterministic-UCT behind a
  branching threshold; optional CP-SAT inner scheduler; multiprocess offline solve.
  **Gate:** a dense multi-enemy map solved within 60 s (or an explicit authorized
  exception), and a *constructed unsolvable* map emits a loud, explained failure (no
  silent degrade, no false win).

- **Phase 5 — Live, general.** Drive plans through the driver; missed-aim-is-a-crash;
  meanie/divergence handled by replan from the true observed state. **Gate:** a recorded
  live win on ls0 (`$0CDE` bit 6) with no missed aim, then a recorded live win on a
  multi-enemy landscape.

---

## 13. Alternatives considered

- **Strongest alternative — PDDL+/numeric-temporal via ENHSP/unified-planning.** Model
  energy as a numeric fluent and gaze as a timed process; let ENHSP search. *Rejected as
  the spine* because the ray-march LOS/aim and the *state-dependent* action set (meanies
  appear/expire, LOS follows the live object stack) have no faithful declarative
  encoding — grounding them from the sim per state rebuilds the custom search. Revisit
  only if the custom search proves unsteerable on complex maps.
- **OR-Tools CP-SAT for the whole plan.** The climb's strong sequential dependencies
  ("be up here to see there") are awkward as a monolith; CP-SAT's real value is the
  inner exposure-scheduling CSP (§5, §7.5), not the spine.
- **Pure MCTS/UCT as the spine.** Sampling for a deterministic, exactly-evaluable
  problem adds variance for no gain at the macro level; UCT is retained only as the
  anytime *escalation* tier for large-branching exposure nodes, with guided rollouts.
- **Keep best-first, flip exposure to a hard filter.** Smaller diff, but the file's
  height-objective + soft-cost + static-mask architecture cannot express managed
  exposure or modeled meanies. Replace.

**Bottom line:** a general, deterministic, complete-state forward search on the
bit-exact simulator — with meanies made a validated stateful mechanic first, exposure
managed as hard energy feasibility over the true drained window, and tractability
delivered by HTN macros + admissible heuristics + a weighted-A\*→ARA\*→UCT ladder — is
the smallest design that wins *general* landscapes, with the never-seen ls0 win falling
out as its cheapest special case.
