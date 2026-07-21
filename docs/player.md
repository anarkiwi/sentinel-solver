# The reactive player (`sentinel/player.py`)

A tick-by-tick greedy player over the `sentinel/` model. No search tree, no lookahead
branching, and **no PRNG reads**: each decision tick observes the live `State` and commits
one action. It wins landscape 0; it does not win seed 66 (typed `0042`) under the
ROM-faithful aim/settle clock — see the tests below and
[plan_fidelity.md](plan_fidelity.md). The planning alternative is
[astar_player.md](astar_player.md); both share `BasePlayer` (`sentinel/playerbase.py`).

```bash
python -m sentinel.player 0            # play landscape 0, print the action trace
python -m sentinel.player 0 --audit    # + strict post-settle invariant accounting
```

## Decision loop (`_tick`)

Each tick picks the first applicable action, then advances the world by that action's real
duration (aim pan + settle frames via `enemies.advance_frames`), so enemies rotate, target
and drain while the player aims — there are no free moves. `urgent` means the player's own
tile window is at or below `SAFE_FRAMES`.

1. **Endgame** (Sentinel already absorbed, `_endgame`) — robot on the platform tile,
   transfer in, hyperspace (`do_hyperspace $2156` sets the complete flag), waiting out a
   surviving cone on the platform for up to one rotation period first.
2. **Dissolve a meanie** (`_meanie_response`) — absorb it, but only when the aim beats its
   rotate-to-face window (`$16F2`: ±8 units per update reload); otherwise the transfer-out
   dissolve outruns it.
3. **Counterattack** (urgent only, `_counterattack`) — seen by an absorbable enemy: absorb
   IT rather than flee. No facing requirement; the budget is the seer's own drain
   countdown (`$0C20`). The Sentinel qualifies only as the last enemy standing.
4. **Hunt enemies** (non-urgent, `_hunt_enemies`) — absorb any sentry whose tile is
   aim-landable, cheapest aim first, and only while the aim leaves `SAFE_FRAMES` of the
   own-tile window intact. Each kill permanently deletes a rotating gaze.
5. **Absorb the Sentinel** (non-urgent, `_absorb_sentinel`) — dead last (the `$1B8E`
   slot-0 lock would strand every remaining enemy), no meanie alive, and the endgame
   affordable (robot 3 + hyperspace 3 − Sentinel's +4 => energy >= 2).
6. **Transfer up** (`_transfer_up`) — into the highest aim-landable robot that raises the
   eye, never into a gaze, safe window required unless urgent.
7. **Finish the hop in progress** (`_climb(only_tile=self.hop_tile)`).
8. **Reclaim / harvest** (non-urgent, `_reclaim`) — absorb old shells and spent pedestals
   below the eye, trees while energy is under `HOP_COST + 6`, ordered by aim-frames per
   energy unit, and only when the aim leaves room for the hop that must follow.
9. **Climb** (`_climb`) — boulder on the best safe landable tile, robot on the pedestal,
   transfer. `_no_strand` refuses a pedestal that leaves the next hop unaffordable unless
   the abandoned shell stays keyboard-landable from the destination.
10. **Urgent fallbacks** — a cheap reclaim when cornered and under `HOP_COST`; `_escape`
    (a transfer that strictly improves the arrival window, else hyperspace as the true
    last resort); a least-bad hop while the world is still frozen; otherwise `_wait`
    (`WAIT_FRAMES`).

`_climb_scan` ranks builds by `(gains LOS on the hunt target, robot eye height, cheaper
aim, wider window)` over the primary-plane view dict, falling back to the full pitch band,
then to graded relaxations: `seen_tier` 1 tolerates undrainable partial sight, 2 is
least-exposed no-other-choice (urgent or frozen only).

## Placement invariant

`_drain_gate` (`playerbase`): a boulder is exempt (`$16E6` drains robots only); a robot or
transfer destination must keep its time-to-first-drain past the budget it will stand
exposed — the aim **plus** the post-action settle, since the object is on the board and
exposable for the whole settle. Standing in a live cone is not itself a refusal: sight only
ARMS the `$0C20` countdown (`$1825`), so `_gaze_window` already prices that body's residual.
Partial sight is not a drain; its slower meanie arm (a tree within 10 tiles, `$19C3`) is
priced into `_gaze_window` instead. Exposure is judged at ARRIVAL, on the ROM's own `$8401`
bearing and `$18B8` cone gate.

`--audit` (`_account`) re-checks the invariant on the ACTUAL placed object after each
settle, via `relative.can_see_object`: only a robot body left in a live full-sight cone is
recorded as a breach.

Creates also respect `_reserve`: while any enemy or meanie lives, energy must never drop
below the 3 a forced hyperspace costs (`$215F` kills below it).

## Enemy model (deterministic only)

- **Gaze window** per tile (`_gaze_window`): frames until some enemy can take ENERGY off a
  robot there — the cone onset (`_cone_onset`: rotating ±10-unit `$0C68` cone, ±4 margin,
  from current facing, the fixed rotation step and the `$130C`/`$1317` cadence) **plus**
  the `$0C20` draining countdown that arrival only ARMS (`_drain_clock`; `$1825` loads 120
  rounds = 449.6 f, `$1A31` re-zeroes it after each drain), so a cone pass costs nothing
  for its first 449.6 f. The residual is the live cooldown byte when the enemy already
  holds that body. `inf` = never; `_cone_onset` remains the bare cone-arrival accessor.
- **Meanie window** (`_meanie_window`): a partially-seeing enemy must rotate on, run the
  ~120-round drain countdown to the meanie branch (`$183D`/`$1852`), spawn (`$1869`) and
  rotate the meanie to face (`$171B`) — always far slower than a drain, never 0.
- Hyperspace and meanie landing tiles are treated as unknowable; the PRNG is never read.

## Aiming and cost

Every tile-targeted action resolves through the ROM aim oracle (`aim.propose`/`aim.gate`,
the `$1B40-$1B46` path): an action fires only on a keyboard-lattice view whose ray lands
the target. `_aim_frames` prices sights toggle (`$134C` recentre) or bearing reuse, the pan
cadence (16-step scroll per ±8 bearing notch `$10EE`, 8-step per ±4 pitch notch `$1135`,
u-turn `$1B2F` as a full action tap), cursor travel, and each notch's own `plot_world`
redraw; `_settle` prices the post-action redraw. Both come from the projector cost model
([render_cost.md](render_cost.md)). The world advances by the aim before the action fires
and by the settle after, so aim cost is outcome-determining: under-charging it places a
body into a gaze the planner modelled as empty.

Landability queries use one cheap primary-plane sweep per tick (`_Views`), falling back to
one full pitch-band sweep only for down-looks a single-ray visibility check first confirms
plausible.

## Tests (`sentinel/tests/test_player.py`)

- `test_player_wins_landscape_0` — wins landscape 0 alive and solvent, last verb
  `hyperspace`, Sentinel slot empty.
- `test_player_wins_landscape_0042` — **xfail** (non-strict): under the accurate
  view-aware transfer settle the greedy heuristics have no safe winning line on seed 66
  and the player dies escaping.
- `test_player_placement_invariant` — the built-in audit records zero breaches on
  landscape 0 (still a win) and on seed 66 (loss, but breach-free): the planner correctly
  refuses the unsafe transfers rather than taking them.

Landscape 0335 is the stress board for this path — interleaved cones, short out-of-phase
windows, constant meanie arm pressure. Run it with `--audit` to exercise the graded
relaxation tiers. The mechanics that make it hard are game rules, documented in
[gameplay.md](gameplay.md).

Build it as `Game.typed(335)` — **7 enemies** (Sentinel at (28,17) height 12 plus six
sentries), player at (11,17), eye 3.875, matching the `ls335.json` human win. A landscape
number is always the number you type; `Game.new` takes the raw seed and gives a different
board. See [plan_fidelity.md](plan_fidelity.md).
