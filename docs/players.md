# The players

Two players share `BasePlayer` (`sentinel/playerbase.py`): the **phase player**
(`sentinel/phase_player.py`), the default, which wins all eight measured boards, and the
**reactive greedy player** (`sentinel/player.py`), which wins landscape 0 and is what the
live-determinism gate drives. Neither reads the PRNG. Every tile-targeted action resolves
through the ROM aim oracle (`aim.propose`/`aim.gate`, the `$1B40-$1B46` path), and each
tick advances the world by that action's real duration, so enemies rotate, target and drain
while the player aims — there are no free moves.

```bash
python -m sentinel.phase_player 335     # offline
python -m sentinel.player 0 --audit     # offline, + strict post-settle invariant accounting
python -m driver.play_player 335        # live in VICE (--player greedy for the reactive one)
```

## The phase player

Built from the game's rules rather than a cost model. Three facts:

**Time is free; only exposure costs.** An idle frame costs nothing unless a cone is on the
body, so an unseen body may wait as long as it likes. Corollary: waiting is the one move
that cannot help while you ARE being drained.

**A gap does not need forecasting.** `enemies.step` is byte-exact, so instead of predicting
when a cone arrives, clone the state, run it forward over the action's real duration and
look (`_drained_over`).

**Height buys freedom, and nothing else does.** Measured over every stance on ls335:

| eye | tiles it can land | stances it can climb to | enemies it can strike |
|---|---|---|---|
| 4.0 | 6 | 18 | 0 |
| 7.0 | 8 | 6 | 0 |
| 8.5 | 62 | 146 | 0 |
| 9.5 | 107 | 140 | 2.91 |
| 11.5 | 153 | 13 | 4.78 |

Nothing on the board can strike any enemy below eye 8.875, and the body starts at 3.875, so
the opening cannot be scored by progress toward a target: for the first five units of climb
there is no reachable target. Exposure rises with height too (`hot` 2.75 → 4.9), so the
climb is not into safety, it is into options.

### The phases

| phase | goal | ends when |
|---|---|---|
| **1a establish** | energy in hand, in fuel-rich ground | the best climb on offer is affordable |
| **1b breakout** | height, per unit of a finite purse | an enemy is landable |
| **2 convert** | absorb, Sentinel last, then the platform | hyperspace from the platform |

1a and 1b want opposite things from the same tile — 1a to stay where the trees are, 1b to
leave for height — and a single scorer blending them strands the body at eye 7.875 with 56
climbs available and one absorbable object in reach.

### The rules that decide a move

Each is a fact about the ROM, not a tuned quantity.

* **Never idle under fire** — a cone on the body makes every waiting quantum a drain.
* **A hop's danger is at the destination**, and the probe must cover the *whole* hop
  including aims (`_span`, each action's real aim-plus-settle), not a fixed duration.
* **Never land on zero** — `$1A00` kills on a drain arriving at zero energy.
* **Do not hop where you cannot eat** — a landing with no fuel under it is a dead end,
  whatever it gains in height.
* **The Sentinel is the point of no return** — `$1B91` reads `objects_flags[0]` absolutely,
  so once its slot empties EVERY later absorb is refused. The robot and the 3-energy
  hyperspace toll (`$216A`, which kills on underflow at `$2170`) must be banked before the
  strike, not gathered after it.
* **The endgame is exclusive** — with the Sentinel gone the platform is an eye gain, so an
  unguarded climb will spend the hyperspace toll on height.
* **A tie is settled by the outcome** — when the score cannot separate candidates it holds
  no information about them, so each is played to the end and one that wins is kept.

### Results

No node budget and no wall-clock cutoff: the planner is deterministic, so a run's outcome
does not depend on host load.

| board | enemies | result | actions | energy left |
|---|---|---|---|---|
| ls0 | 1 | won | 16 | 1 |
| ls42 | 2 | won | 35 | 4 |
| ls110 | 3 | won | 53 | 17 |
| ls60 | 7 | won | 55 | 15 |
| ls298 | 7 | won | 34 | 5 |
| ls321 | 7 | won | 78 | 12 |
| ls373 | 7 | won | 65 | 15 |
| ls335 | 7 | won | 73 | 20 |

ls335 live in VICE: **66 actions, final energy 25**, verified by the ROM's own
landscape-complete flag `$0CDE` bit 6.

### The tie that decided ls110

Ties are settled by rollout because no measurement available at the tie separates the
candidates. At eye 6.375 with E=10, **nine candidates scored identically** —
`(gain/cost, fuel_near) = (0.1667, 5)` for every one — so `_best_climb` returned whichever
`views.band()` yielded first, and the run died 4 ticks later at 17 actions, eye 7.375, E=0
(that stall was forced: 11 of its 12 candidates failed the arrival probe and the survivor
was the tile it died on). Forcing each of the nine and playing on, the planner otherwise
untouched:

| forced | outcome | | forced | outcome |
|---|---|---|---|---|
| (23,1) | **won** 73, E=11 | | (25,0) | lost — the one taken |
| (24,1) | **won** 53, E=17 | | (26,0) (26,8) (26,9) | lost |
| (25,4) | **won** 42, E=5 | | (24,11) (26,10) | lost |

Three of nine win outright, and the score cannot see which — nor can any other measurement
taken there: eye, energy, the landing's reachable fuel, how many enemies watch it, its
landable-set size, its hop-distance to a strike, and the 4-tick fork outcome were all
computed for the nine, and **two of the three winners rank last on every one**. Deepening
the lookahead from 4 to 8 to 16 changed nothing: the winning hop's payoff is 40 actions
away, and no valuation at any affordable depth reaches it.

So the tie is not scored, it is **played**. `_settle_tie` forks each tied candidate, builds
it, and runs the fixed ladder to termination, keeping the first that wins — no weight, no
horizon, the only thing consulted is whether the game was won. It is affordable because
ties are rare (two of fifteen ticks on ls110) and a rollout plays a whole board; a tie costs
one rollout per candidate until one wins, and the full nine at the ls110 opening, where none
does.

## The reactive greedy player

Tick-by-tick, no search tree and no lookahead branching: each tick observes the live `State`
and commits one action. `urgent` means the player's own tile window is at or below
`SAFE_FRAMES`. `_tick` picks the first applicable action:

1. **Endgame** (Sentinel already absorbed, `_endgame`) — robot on the platform tile,
   transfer in, hyperspace (`do_hyperspace $2156` sets the complete flag), waiting out a
   surviving cone on the platform for up to one rotation period first.
2. **Dissolve a meanie** (`_meanie_response`) — absorb it, but only when the aim beats its
   rotate-to-face window (`$16F2`: ±8 units per update reload); otherwise the transfer-out
   dissolve outruns it.
3. **Counterattack** (urgent only) — seen by an absorbable enemy: absorb IT rather than
   flee. No facing requirement; the budget is the seer's own drain countdown (`$0C20`). The
   Sentinel qualifies only as the last enemy standing.
4. **Hunt enemies** (non-urgent) — absorb any sentry whose tile is aim-landable, cheapest
   aim first, only while the aim leaves `SAFE_FRAMES` of the own-tile window intact. Each
   kill permanently deletes a rotating gaze.
5. **Absorb the Sentinel** (non-urgent) — dead last (the `$1B8E` slot-0 lock would strand
   every remaining enemy), no meanie alive, and the endgame affordable (robot 3 +
   hyperspace 3 − Sentinel's +4 ⇒ energy >= 2).
6. **Transfer up** — into the highest aim-landable robot that raises the eye, never into a
   gaze, safe window required unless urgent.
7. **Finish the hop in progress** (`_climb(only_tile=self.hop_tile)`).
8. **Reclaim / harvest** (non-urgent) — absorb old shells and spent pedestals below the
   eye, trees while energy is under `HOP_COST + 6`, ordered by aim-frames per energy unit,
   and only when the aim leaves room for the hop that must follow.
9. **Climb** — boulder on the best safe landable tile, robot on the pedestal, transfer.
   `_no_strand` refuses a pedestal that leaves the next hop unaffordable unless the
   abandoned shell stays keyboard-landable from the destination.
10. **Urgent fallbacks** — a cheap reclaim when cornered and under `HOP_COST`; `_escape` (a
    transfer that strictly improves the arrival window, else hyperspace as the true last
    resort); a least-bad hop while the world is still frozen; otherwise `_wait`.

`_climb_scan` ranks builds by `(gains LOS on the hunt target, robot eye height, cheaper
aim, wider window)` over the primary-plane view dict, falling back to the full pitch band,
then to graded relaxations: `seen_tier` 1 tolerates undrainable partial sight, 2 is
least-exposed no-other-choice (urgent or frozen only). Landability queries use one cheap
primary-plane sweep per tick (`_Views`), falling back to a full pitch-band sweep only for
down-looks a single-ray visibility check first confirms plausible.

### Placement invariant

`_drain_gate` (`playerbase`): a boulder is exempt (`$16E6` drains robots only); a robot or
transfer destination must keep its time-to-first-drain past the budget it will stand
exposed — the aim **plus** the post-action settle, since the object is on the board and
exposable for the whole settle. Standing in a live cone is not itself a refusal: sight only
ARMS the `$0C20` countdown (`$1825`), so `_gaze_window` already prices that body's
residual. Partial sight is not a drain; its slower meanie arm (a tree within 10 tiles,
`$19C3`) is priced into `_gaze_window` instead. Exposure is judged at ARRIVAL, on the ROM's
own `$8401` bearing and `$18B8` cone gate. `--audit` (`_account`) re-checks the invariant on
the ACTUAL placed object after each settle via `relative.can_see_object`: only a robot body
left in a live full-sight cone is recorded as a breach.

Creates are gated by `_affords`, which prices exposure as the RATE it is: an action may
spend down to 0 (`$1BBF`; `$1A00` kills only on a drain arriving at an already-empty
meter), so the test is `energy - cost - drains_over_this_span >= _reserve`. `_reserve` is 0
unless a meanie is ALIVE, when it holds the 3 a forced hyperspace costs (`$215F` kills below
it). A flat floor for merely being SEEN refused 6 of the 35 creates in the recorded ls335
human win.

### Enemy model (deterministic only)

- **Gaze window** per tile (`_gaze_window`): frames until some enemy can take energy off a
  robot there — the cone onset (`_cone_onset`: rotating ±10-unit `$0C68` cone, ±4 margin,
  from current facing, the fixed rotation step and the `$130C`/`$1317` cadence) **plus** the
  `$0C20` draining countdown that arrival only ARMS (`_drain_clock`; `$1825` loads 120
  rounds = 449.6 f, `$1A31` re-zeroes it after each drain), so a cone pass costs nothing for
  its first 449.6 f. The residual is the live cooldown byte when the enemy already holds
  that body. `inf` = never.
- **Meanie window** (`_meanie_window`): a partially-seeing enemy must rotate on, run the
  ~120-round drain countdown to the meanie branch (`$183D`/`$1852`), spawn (`$1869`) and
  rotate the meanie to face (`$171B`) — always far slower than a drain, never 0.
  `enemies.meanie_threat` omits the tree gate that `attempt_to_create_meanie $19A1`
  applies, so it over-reports; `playerbase._tree_near`/`_meanie_window` apply it.
- Hyperspace and meanie landing tiles are treated as unknowable.

### Tests (`sentinel/tests/test_player.py`)

- `test_player_wins_landscape_0` — wins alive and solvent, last verb `hyperspace`, Sentinel
  slot empty.
- `test_player_wins_landscape_0042` — **xfail** (non-strict): under the accurate view-aware
  transfer settle the greedy heuristics have no safe winning line on landscape 42 and the
  player dies escaping.
- `test_player_placement_invariant` — zero audit breaches on landscape 0 (a win) and on
  landscape 42 (a loss, but breach-free): the player refuses the unsafe transfers rather
  than taking them.

Landscape 335 is the stress board for this path — interleaved cones, short out-of-phase
windows, constant meanie arm pressure. Run it with `--audit` to exercise the relaxation
tiers.
