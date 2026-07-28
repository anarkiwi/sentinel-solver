# The stance planner: geometry as a graph, solved instead of searched

`sentinel/stancegraph.py` + `sentinel/stance_player.py`. A layered replacement for the
monolithic weighted A\* over hand-written macros ([astar_player.md](astar_player.md)),
built because that search does not fail on ls335 by running out of budget — it fails by
running out of *moves*.

```bash
python -m sentinel.stancegraph 335                 # the graph, and a route to the Sentinel
python -m sentinel.stance_player 335               # A* plus the route generator
python -m sentinel.tests.human_regress ls335.json --planner stance   # score it
```

## Why the A\* frontier empties

Measured on `Game.typed(335)`, root search, default budgets:

```
exp=1  d=0  eye=3.88 foes=7 E=10 tile=(11,17) -> 11 children
exp=4  d=11 eye=5.38 foes=7 E=10 tile=(8,21)  ->  2 children
exp=20 d=11 eye=5.38 foes=7 E=3  tile=(8,21)  ->  5 children
exp=30 d=19 eye=5.88 foes=7 E=1  tile=(9,21)  ->  0 children
RESULT 30 expansions, 5.4 s, plan=None
```

Not the 200000-node budget and not the clock: the heap **empties after 30 expansions**.
The search descends into energy exhaustion (E to 1, every generator refuses), `foes`
never leaves 7, and there is nothing to back up to. Three causes, none of them a
tunable constant:

1. **`h` is flat over the phase that matters.** The human ls335 win spends its first 48
   actions climbing 3.88 to 9.38 with **zero kills**, so `remaining * _ABSORB_EST` is a
   constant 7 throughout and `f ≈ g` — weighted A\* degenerates to Dijkstra on frames
   over a ~120-action horizon.
2. **The successor function is incomplete by construction.** Children come only from the
   macro generators; when each refuses, the node is terminal. Energy appears in no
   heuristic term, so a dead end is not costed, it is invisible.
3. **The goal is a needle, and the search runs forward.** Of 1192 stances, the Sentinel
   is landable from **13** (on 5 tiles); the sentries from 286–449. From the opening
   stance the player can land **4 tiles**. No forward ranker finds those 5 tiles.

Two further measurements shape the design. **The gaze is not avoidable**: 916 of 1192
stances lie in some enemy's sightline, so exposure has to be priced, not walled off —
which is why hard gates empty the frontier. And **the enemy clock is one scalar**: over
11000 simulated frames every enemy rotates on a shared 753-frame cadence with a fixed
per-enemy offset, so `facing_i(t) = h0_i ± 20·⌊(t + off_i)/753⌋`. The slots that miss
rotations (1, 5, 6 — 11, 6 and 9 against 15) are the `_consider_enemy_state` stall,
[plan_fidelity](plan_fidelity.md) open item 1.

## The layers

| layer | what | state |
|---|---|---|
| **L0** geometry | `(tile, k)` stances and what each can land, swept once | `StanceGraph.seen` |
| **L1** phase | exposure as a function of one rotation index | *not built yet* |
| **L2** route | resource-constrained shortest path over `(stance, energy)` | `StanceGraph.route` |
| **L3** ordering | which enemy next: Held–Karp over the sentry subsets | `stanceorder.order` |
| **L4** execution | the existing `_hop_exec` / gates / re-plan machinery | unchanged |

`StanceGraph.hops_to(tile)` is L0's other product: a backward level BFS from the strike
set giving every stance's distance to a striker (4 ms, ls335 spans 0–5 hops over 925
stances, 267 unreachable). `StancePlayer._h` uses it in place of the eye-deficit proxy.
It is *not* what fixed the cost regression below — measured neutral on ls42 — but it cuts
handover 144 from 5 expansions to 4.

L0 and L2 are implemented and wired behind `--planner stance`; L1 and L3 are the next
stages. See [Open](#open) below.

## L0 — the geometry oracle

A **stance** is `(tile, k)`: a body on `k` boulders on a bare flat tile. `KMAX` is 3
(k≥4 adds no strike tile on ls335). The sweep probes each stance and records
`landtable.landable_set` for it, giving one bool matrix:

```
seen[stance, tile]      1192 x 1024 on ls335
```

Reading a **row** is "what can this stance land" — the forward question the A\* player
already asks. Reading a **column** is "which stances can strike this tile" — the
backward question it cannot ask at all, and the one the endgame needs. That inversion is
`strike_stances(tile)`, a `np.flatnonzero` on a column.

`adj[i, j] = seen[i, tile_of(j)] and eye[j] > eye[i]` is the stance graph proper: 96472
edges on ls335, built by one fancy-index. The eye-monotone half makes it a DAG.

The sweep costs **~37 s a board** and is a pure function of the tile map (terrain plus
object placement — everything an LOS march reads), so it is cached to
`out/stancegraph/<sig>.npz` with `seen` bit-packed: 152 KB packed, ~18 KB compressed on
disk, reloaded in milliseconds.
`build(tiles=...)` restricts the sweep to a tile subset for a partial graph that is still
sound for every stance it holds.

Per stance the sweep also records `fuel` (trees reachable at an aim inside `CHEAP_AIM`,
240 f) and `hot` (how many enemies have any sight of a body there, over all rotations).

## L2 — the route

`StanceGraph.route(target, energy, ...)` is a resource-constrained shortest path over
`(stance, energy)`, minimising hops:

- **spend** per hop: `2k + 3`, plus the drains its build is billed under a sightline at
  the cone's duty cycle (`hot * CONE_DUTY * build_frames(k) / DRAIN_DELAY`).
- **income** on arrival: the cheap-aim trees the landing reaches, plus `2k' + 3` for the
  pedestal it climbed off when the new stance can land the old tile — the inchworm
  recycle of [gameplay §7](gameplay.md#7-how-a-human-wins-quick-strategy).
- **dominance**: a state is pruned when a settled state at the same stance holds ≥ energy
  in ≤ hops, which is one `best[node, have:].min()` slice.

Threat enters as a **rate**, so the state carries no clock. That is L1's job.

The parent chain is keyed by `(node, energy, hops)`. Keying it by `(node, energy)` alone
lets a later, longer push overwrite a shorter one's chain, and `_unwind` then walks a
path longer than Dijkstra actually found — measured as an 11-hop answer where the true
optimum was 6.

From the ls335 opening stance, every one of the 7 enemies is routable, in under 0.2 s
against a cached graph:

| target | tile | hops | energy on arrival |
|---|---|---|---|
| Sentinel | (28,17) | 6 | 11 |
| sentry 1 | (4,18) | 5 | 8 |
| sentry 2 | (2,1) | 4 | 10 |
| sentry 3 | (30,27) | 4 | 10 |
| sentry 4 | (7,11) | 4 | 10 |
| sentry 5 | (12,10) | 4 | 14 |
| sentry 6 | (4,24) | 4 | 14 |

The Sentinel route: `(10,17) k=3 → (13,27) k=0 → (11,13) k=3 → (22,29) k=0 → (13,8) k=2
→ (4,17) k=1`, eye 3.88 → 12.38. `(11,13)` is on the human's own line.

A route is a **feasibility certificate, not a plan**: it counts hops, prices aim cost
only through the exposure toll, and does not check the strike stance survives long enough
to aim and fire. Turning it into steps is L4's job, and every hop is re-gated there.

## Wiring: `_c_route`

`StancePlayer` is `AStarPlayer` plus one generator, and `--planner astar` is unchanged.

**The route is a FALLBACK, not an extra child.** A weighted search returns the first goal
it pops, so an added child can surface a *worse* goal sooner. Offering routes
unconditionally still won every board but cost **ls42 2 actions (32 to 34) and ls110 6
(48 to 54)**, with ls110's spare energy halved (12 to 6). Attribution was measured by
running the two halves apart:

| ls42 variant | actions | frames | energy |
|---|---|---|---|
| astar | 32 | 8679 | 4 |
| route off, graph `h` on | 32 | 8679 | 4 |
| route on, stock `h` | 34 | 9935 | 1 |
| both | 34 | 9935 | 1 |

So the generator owned all of it and the heuristic none — the eye-deficit `h` was *not*
what mis-ranked a bundled route, which is what the first attempt at a fix assumed.
`_expand` therefore offers route children only when the ranked generators produced none,
which is exactly the empty-frontier failure mode that loses ls335, and restores ls0/42/110
to their A\* lines byte for byte. `route_always=True` restores unconditional offering.

`_c_route(node, target)` asks for a route from the current stance, then walks it through
the existing `_hop_exec`, interleaving `_reclaim_one` when the next hop is unaffordable
and stopping at the first hop the live board no longer permits (returning the prefix, as
`_c_reach` does). Each routed hop is held to **exactly** `_pick_hop`'s gates — the
destination `_drain_gate` on `_TAIL_FLOOR`, the exact `_hop_price` tail against the
tile's gaze window, and the source-side `_affords` — so a route cannot smuggle a hop past
a safety test the ranked climb would refuse.

Targets are the platform once the Sentinel is gone, else the nearest living non-landable
enemies (`ROUTE_TARGETS` = 3).

### Departure, not just arrival

Those gates all ask **can the body get here**. None asked **can it leave again**, and a
stance it cannot leave is where the ls335 opening died. The route's first hop is
`(10,17) k=3`, and the gate probes exposure with a phantom on the *bare* tile: one
full-sight seer. A body on three boulders has **four**. They lock on the landed robot,
and a draining enemy stops rotating (`$178C` returns before the `$17F9` rotate), so the
cone never leaves — `_earliest_start` is `inf` for every duration down to 50 f, no
reclaim is aimable from the stack, and the climb has just spent 9 of its 10 energy
getting there.

`_try_hop` therefore compiles each hop on a **trial clone**, harvests, and only commits
if `_can_leave` holds: the landing must fund and start another hop, or strike the target
outright. A refusal leaves the pre-hop state intact for `_repair` to blacklist the stance
and re-route — the same propose/verify split, now closed on both ends of a hop. This is
the discipline `_advance_hop` already keeps for the ranked climb (`_climb_continues`),
which the route path bypassed entirely.

The affordability screen is priced at the hop the route **named** — `2k + robot` — not at
`HOP_COST`, which is the k=1 price of a hop nobody has picked yet. ls335's second hop is
a k=0 landing costing 3, held against 5 at the E=3 the router itself predicted.

Measured on the ls335 opening (`--planner stance`, 20 s/search, 20000 nodes):

| | actions | line |
|---|---|---|
| before | **0** | route stalls one hop in, at E=3 on `(10,17) k=3`, and the run loop is drained to death |
| after | **9** | `k=3` on `(10,18)`, transfer, four reclaims — E=7 at eye 5.38 |

Routing to `(7,11)` now compiles the human's own inchworm instead: `k=1` onto `(10,17)`,
three reclaims, `k=0` onto `(9,21)`, then absorbing the `(10,17)` stack back — 2 hops,
**E=12**. ls0/42/110 and handovers 111/133/144 are bit-identical.

The graph is a **snapshot** of the board the player starts on. Later creates and absorbs
change LOS; a stale hop simply fails to compile and the child truncates. Re-planning on
divergence is the existing `_plan_step_stale` / `_restale` machinery.

## Result so far

ls335 handover 144 — all 7 enemies already dead, E=11, body at (4,18), platform at
(28,17) — is the board the A\* planner loses on with **1 expansion and an empty
frontier**. With the route generator:

```
stance: WON in 5 actions, 4 expansions, 1 s against a cached graph (43 s cold)
   transfer (28,16)  absorb (28,17)  robot (28,17)  transfer (28,17)  hyperspace
```

No board regresses. Measured offline, `--planner astar` against `--planner stance`:

| board | astar | stance |
|---|---|---|
| 0 | 23 actions / 6240 f / E6 | 23 / 6240 / E6 |
| 42 | 32 / 8679 / E4 | 32 / 8679 / E4 |
| 110 | 48 / 13290 / E12 | 48 / 13290 / E12 |
| ls335 handover 144 | **lost** | **won, 5 actions** |

(The ls42/ls110 astar figures are this branch's, not the ones in
[plan_fidelity](plan_fidelity.md) — `_required_eye` moved them.)

## L3 — which enemy next

`sentinel/stanceorder.py`. The leg costs are free: `hops_to(tile)` already gives every
stance's distance to a striker, so

```
legs[i][j] = min over stances striking tiles[i] of hops_to(tiles[j])
```

is 7 backward BFS runs and a min-reduce — no Dijkstra per pair. Held–Karp then runs over
the sentry subsets (2⁶ × 6 cells) with the Sentinel forced last by the `$1B8E` lock and
the platform after it. **0.02 s** on ls335, and it reproduces the human's own kill order
on 5 of 7 positions:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| human | (2,1) | (12,10) | (7,11) | (30,27) | (4,24) | (4,18) | Sentinel |
| DP | (4,24) | (12,10) | (7,11) | (30,27) | (2,1) | (4,18) | Sentinel |

`StanceGraph.seers` (`bool[stance, slot]`) records WHICH enemies watch a stance, so
`watchers(dead=...)` and `route(dead=...)` drop an absorbed enemy's cone: the board opens
up as the hunt proceeds, which is the graph mutation the DP reasons over.

**L3 did not convert the opening board**, and the reason is architectural rather than
strategic — see [Open](#open) item 0. The ordering is right and each of its top three
goals compiles into a real 9-step child in 0.1 s; the search still returns no plan.

### `route_policy`

When route children enter the frontier: `stuck` (only a childless node), `root` (also at
depth 0), `always`. Measured over ls0 / ls42 / ls335 handovers 0, 133, 144:

| policy | ls0 | ls42 | h0 | h133 | h144 |
|---|---|---|---|---|---|
| `stuck` | 23, exp 3 | **32** | lost, exp 2 | 7 | 5 |
| `root` | 23, exp 19 | **32** | lost, exp 2 | 7 | 5 |
| `always` | 23, exp 33 | 34 | lost, exp 2 | 7 | 5 |

`stuck` is the default: equal-best and cheapest. Handover 0 loses identically under all
three, so the gating policy is NOT what refuses the opening board — an earlier hypothesis
that the fallback rule was starving the ls335 root is disproved by this table.

`_graph_hops` separates "every goal is already landable" (0 hops to go) from "the DP
bridged nothing" (no answer, fall back to the eye deficit). Both empty `_route_targets`,
and scoring the second as 0.0 made `h` optimistic exactly where the search is stuck — on
ls335 the Sentinel stands ON the platform tile, so the absorb-lock suppresses both copies
and the empty list is reached with 7 enemies alive. Fixing it changed no board's result;
it is a latent correctness bug, not a measured gain.

## Milestones

Scored by `human_regress --planner stance`, which bisects the handover axis for the
highest board the planner cannot convert.

| stage | expected ls335 `first_loss` |
|---|---|
| A\* today | 144 |
| L0+L2 | ≤ 113 |
| +L1 phase table | ≤ 89 |
| +L3 ordering DP | ≤ 44 |
| +tree-pin | 0 |

## Open

0. **A route must be COMMITTED TO, not offered as a child — the frontier never picks it.**
   Four changes were built and measured against the ls335 opening, and all four leave it
   at `plan=None`: subgoal search (`_is_goal` hook, `_subgoal_slot`, `_tick` re-planning a
   spent plan instead of waiting forever), `_harvest`, route repair (`route(blocked=...)`
   re-routing around whatever the executor's exact gates refuse), and capping stack height.

   The measurements that localise it, in order:

   - **Not energy.** Predicted vs executed, hop by hop: `route()` reaches the (4,24)
     subgoal in 4 hops predicting E=14; the executor gets E=5 after hop 0 where the model
     predicted 3, so `_harvest` *over*-delivers. Energy is not what refuses the hop.
   - **A window gate refuses it.** Hop 1 to (13,27): gaze window **450 f** against an
     exact priced tail of **1141 f**. `route()` prices a hop at its floor (448 f) while the
     real cost is aim-dominated and 2.5x that, so the planner proposes hops the executor
     must refuse. Repair blacklists them, but 916 of 1192 stances are watched.
   - **Not stack height.** Capping the route at k<=2 and k<=1 gives byte-identical searches
     (26 expansions), against the human inchworm's k in {1,2}.
   - **The route child loses the frontier race.** Those three caps being *identical* is the
     tell: under `stuck` the 26-expansion tree is driven entirely by the RANKED generators,
     so every route-side change is inert. A route child bundles ~9 steps, so its `g` is
     thousands of frames, and `f = g + 1.4h` pops many cheap shallow children first. The
     route sits unexpanded while the search burns out on shallow hops and dies at E=1.

   So the next change is architectural, not another generator: when a route exists, the
   executor should FOLLOW it — DP picks the goal, the route picks the line, and search is
   left to the fine detail — rather than dropping a macro into a best-first heap and hoping
   it wins on `f`. Ordering, route gating, the heuristic, energy and stack height are all
   ruled out by measurement above.

   **`subgoal` ships OFF** (`SUBGOAL = False`). It is the prerequisite for that
   route-following mode and is kept behind the flag, but on its own it buys nothing and
   costs: ls335's opening is unchanged with it on or off, while ls110 goes 48 actions /
   13290 f / E12 to 44 / 13707 / **E4** — 4 fewer actions for +417 frames and two thirds
   of the energy margin, on a board gated live, where [gameplay §7](gameplay.md#7-how-a-human-wins-quick-strategy)
   makes time the scarce resource. Attribution, one change at a time:

   | ls110 | actions | frames | energy |
   |---|---|---|---|
   | astar | 48 | 13290 | 12 |
   | stance, `subgoal=True` | 44 | 13707 | 4 |
   | stance, `subgoal=False` | 48 | 13290 | 12 |
   | `_harvest` disabled | 44 | 13707 | 4 |
   | `_repair` disabled | 44 | 13707 | 4 |
1. **L1 — collapse time to a phase index.** Carry `r = ⌊t/753⌋`, not 7 facings, and
   precompute per stance a bitmask of which rotation indices it is exposed at. Exposure
   becomes an O(1) lookup instead of a per-node cone march, and the drain gate becomes
   closed-form. It also shrinks the A\* dedup key, which currently over-splits (exact
   energy, full stack multiset) and under-splits (drops the cooldown bytes the gates
   read) at the same time. Blocked on the rotation stall, which the phase table must
   model or it will read optimistic.
2. **L3 — ordering as Held–Karp.** 2⁷ × 7 = 896 DP cells, with the graph mutated per
   subset (an absorbed enemy's cone leaves the exposure mask and its tile becomes a
   stance — the human occupies (4,18), (7,11) and (30,27), all absorbed sentry tiles).
   Run the DP on an exposure-free relaxation and expand true leg costs along a beam.
3. **The eye-monotone edge filter is an approximation.** The human ls335 line descends
   twice (z 9.38 → 8.88) to cross the board. `route(monotone=False)` drops it at the cost
   of losing the DAG.
4. **Only bare flat tiles are stances.** A tile holding a tree becomes a legal stance once
   the tree is absorbed, and the snapshot does not know that.
5. ~~**A missing action class.** The human line has 19 `create TREE` actions...~~
   **WITHDRAWN — the premise was a fixture artifact.** The human player confirms they
   never created a tree, and the recording agrees: on ls335 every `BOULDER` create
   debits exactly −2 energy (17/17) and every `ROBOT` −3 (18/18), while all 19 `TREE`
   "creates" debit **0**. A player create spends first ($1BBF), so a zero-cost create
   is not one. They are enemy discharges ($1A5D → `create_object #$2` →
   `put_object_in_random_tile_below_z $1238`) that `_extract.py`'s object-table diff
   attributed to the player: its LOS filter drops a candidate only when NO keyboard aim
   reaches the tile, and a discharge lands on a random tile among 1024, so plenty are
   aimable. `_extract._paid_for` now requires the energy debit. Applied to the
   recordings it drops 19 events on ls335 (156 → **137** real player actions) and 7 on
   ls110; ls0 and ls42 have no trees at all and are unaffected.

   **The committed `ls*.json` still carry the artifact**, because correcting them is
   not a row deletion. `ls335_clock.json` is a live replay *of the corrupted line* —
   and already only `reproduced: 35` of 156, first divergence at step 20 — and
   `ls335_audit.json` is scored against it, so ten `test_human_clock` /
   `test_human_audit` assertions pin numbers derived from it. Regenerating needs the
   raw `watch_play` logs, which are gitignored. Until then treat any per-index human
   figure on ls335/ls110, **including the handover indices**, as counting phantom
   events.

   What actually drains the player is the mechanism in
   [ls335_minimal.md](ls335_minimal.md): a body that transits exposed cells is drained
   all the way down, because a holding cone does not rotate off.

## Tests

`sentinel/tests/test_stancegraph.py` and `sentinel/tests/test_stance_player.py`. The
whole-board sweep is too slow for CI, so the algorithms are pinned on synthetic graphs
and the geometry on a `build(tiles=...)` subset checked against `landtable` directly.


## Where ls335 stalls now, and why it is open item 1

`--planner stance 335 --time-budget 20 --node-budget 20000` plays **9 actions and stops**:
three boulders and a robot on `(10,17)`, transfer up to eye 5.375, four reclaims back to
E=7. Then `_expand` returns **zero children** and every later search reports
`plan (1 nodes): None` — the root itself is terminal.

At that stance (`(10,17)`, eye 5.375, E=7, all 7 foes alive) every generator is empty:
`_c_reclaim`, `_c_clear`, `_c_relocate`, no absorb target, `_c_pursue` `None` for all four
pursue targets, and `_c_route` `None` for all three route targets. `fire_reason` is `None`
throughout, so nothing is refused by a fire — it is refused by a gate.

Tallying `_pick_hop`'s filters over its 20 landable tiles:

| filter | kills |
|---|---|
| no viable `k` (energy or eye) | 6 |
| `_drain_gate` | 0 |
| **`_hop_price` returns `None`** | **12** |
| window < priced tail + margin | 8 |
| `_affords` (source side) | 2 |
| **survive** | **0** |

and inside `_hop_price` the break is **`_stack_holds` on 12 of 22 (tile, k) pairs** —
nothing else. `_stack_holds` is passed the frames accumulated so far, so it is the TALL
stacks that fail: by the third boulder the forecast says a cone has arrived and `$17B7`
drains the stack to a tree before the robot caps it. ls335 is the board that needs k=3.

That makes the ls335 stall a **gaze-forecast** problem, which is
[plan_fidelity](plan_fidelity.md) open item 1 — `_cone_onset` projects a fixed rotation
cadence, but a draining enemy stops sweeping. So the forecast's arrival times are wrong in
a known direction, and `_stack_holds` inherits it.

Two things are now settled about that item and worth carrying into any attempt on it.
`enemies.step` is **byte-exact against the real 6502** on the divergent states
([human_clock.md](human_clock.md)), so the stall is not a missing branch in the simulator —
`_cone_onset` is a separate forecaster that never consults enemy state at all. And no
constant cadence reproduces the recorded facings, so the fix is not a better number.

## The smallest board that still loses

ls335 does not need 7 enemies to beat this planner. Deleting sentries localises the
failure to **three**: the Sentinel plus `(4,18)` and `(12,10)` — the only loss of the 15
two-sentry boards, with both of its own proper subsets winning
([ls335_minimal.md](ls335_minimal.md)). It dies on the same stance the A\* player dies
on, `(18,24)` at eye 8.375, having transferred onto it at E=0.

That board is small enough to checkpoint and re-enter directly, which puts a
`_pick_hop` change 210 ms from a verdict instead of a 94 s replay
([fast_iteration.md](fast_iteration.md)).

It is now **won**, in 52 actions with E=8 and all three enemies absorbed. The corpus
localised the loss to a plan-vs-execution gap rather than to the gaze forecast: a hop is
atomic, but the executor committed one keypress at a time against energy that had
drifted since the plan was made, laying a boulder for a pedestal it could no longer cap.
`_group_need` re-plans instead of entering a group it cannot finish, and is measured
byte-identical on ls0/42/110 under both planners.
