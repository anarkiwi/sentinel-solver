# The reactive player (`sentinel/player.py`)

A tick-by-tick greedy player over the `sentinel/` model. No search tree, no
lookahead branching, and **no PRNG reads**: each decision tick observes the live
`State` and commits one action. Wins landscapes 0000 (53 actions) and 0042
(Sentinel + 1 sentry, 78 actions) in the simulator, hunting and clearing every
enemy; the counts rose with the ROM-faithful settle/aim clock (§ Sim-vs-live),
under which 0335 (three sentries) and the live 0042 win no longer reproduce.

```bash
python -m sentinel.player 0        # play landscape 0, print the action trace
```

## Decision loop

Each tick picks the first applicable action, then advances the world by that
action's real duration (aim pan + settle frames via `enemies.advance_frames`),
so enemies rotate, target and drain while the player aims — there are no free
moves.

1. **Win move** — Sentinel absorbed: create a robot on the platform tile,
   transfer in, hyperspace (`do_hyperspace $2156` sets the complete flag),
   waiting out any surviving cone on the platform first.
2. **Dissolve a meanie** — absorb it, but only when the aim beats the meanie's
   rotate-to-face window (`$16F2`: ±8 units per ~10-unit reload); otherwise the
   transfer-out dissolve outruns it.
3. **Counterattack** — seen by an absorbable enemy: absorb IT (no facing
   requirement; the budget is the seer's own drain countdown `$0C20`) rather
   than flee; the Sentinel qualifies only as the last enemy standing.
4. **Hunt enemies** — absorb any sentry whose tile is aim-landable within the
   current safety window, cheapest aim first: each one permanently deletes a
   rotating gaze, worth far more than its +3.
5. **Absorb the Sentinel** — dead last (the `$1B8E` slot-0 lock would strand
   every remaining enemy), with the endgame affordable (energy ≥ 2).
6. **Transfer up** — into the highest aim-landable robot that raises the eye.
7. **Reclaim / harvest** — absorb old shells and spent pedestals from the new
   vantage, trees while below headroom, ordered by aim-frames per energy unit,
   and only when the aim leaves room for the hop that must follow.
8. **Climb** — the ping-pong hop toward LOS on the hunt target: boulder on the
   best safe landable tile, robot on the pedestal, transfer. A pedestal that
   would leave the next hop unaffordable is refused unless the abandoned shell
   stays aim-landable from the destination (no stranding).
9. **Wait / escape** — idle when the world may improve; when cornered, a
   strictly-window-improving transfer, then hyperspace as the true last resort.

Placements obey one invariant: **never create on a tile any enemy sees right
now** (any exposure, the ROM's own `$8401` bearing and `$18B8` cone gate);
transfers refuse the DANGER basis — full visibility (`$1838` drains), or
partial with a tree within 10 tiles (the meanie arm, `$19C3`) — so a harmless
partial glimpse cannot orphan a pedestal the build was allowed to make. Every
destination's danger window must cover the aim **plus the post-action settle**
(the object is on the board and exposable for the whole settle) plus a safety
margin — measured at ARRIVAL, not at plan time; graded last-resort tiers
(partial-only sight, then least-exposed) unlock only when no unseen tile exists.
The strict `--audit` flag re-checks this on the actual object after each settle
(`Player._account`), so an aim/settle exposure the plan-time gate missed is
recorded as a breach.

## Enemy model (deterministic only)

- **Gaze window** per tile: frames until some enemy's rotating ±10-unit scan
  cone (`$0C68`) covers a robot on that tile while `$1CDD` gives it full line of
  sight — computed from current facing, the fixed ±20 rotation step, and the
  `$130C`/`$1317` cooldown cadence. `0` = in a gaze now, `inf` = blocked from
  every facing.
- Transfers/builds **never target a tile in a live gaze**; hop destinations need
  a hop-sized window. When the player's own tile turns urgent, requirements relax
  in order (equal-height transfer, then least-bad transfer, then hyperspace as
  the true last resort).
- Hyperspace and meanie landings are treated as unknowable (PRNG-driven); the
  PRNG is never read.

## Aiming and cost

Every tile-targeted action resolves through the ROM aim oracle
(`aim.propose`/`aim.gate`, the `$1B40-$1B46` path): an action fires only on a
keyboard-lattice view whose ray lands the target. Aim time is priced from the
pan cadence (16-step scroll per ±8 bearing notch `$10EE`, 8-step per ±4 pitch
notch `$1135`, u-turn `$1B2F` flip, 1px/frame cursor) and settle time from the
redraw/projector cost model ([render_cost.md](render_cost.md)); both advance the
world before/after the action fires.

Landability queries use one cheap primary-plane sweep per tick, falling back to
one full pitch-band sweep only for down-looks that a single-ray visibility check
first confirms plausible.

## Why landscape 0335 is hard

Typed `0335` is internal seed `$35`: the Sentinel on its platform at (26,18)
height 11 plus **three sentries** at (0,9), (5,22) and (18,3), all at height 9
(the enemy-count draw `$3426` landed 4 under the ≥0100 cap of 8), against a
player starting at (17,28) with the usual 10 energy. Each mechanic that makes
one enemy manageable compounds against four:

- **The union of four any-rotation cones covers almost the whole board.** A
  tile is only safe if blocked from *every* facing an enemy rotates through
  (§6 of gameplay.md); with enemies spread to three corners by the
  section-placement rule (`$1528`, no two in adjacent 4×4 sections), the
  never-be-seen placement set is empty for long stretches of the climb. The
  player's graded relaxations (partial-only sight, least-exposed) engage
  constantly rather than exceptionally.
- **Safe windows are short and out of phase.** Each enemy rotates ±20 units
  every ~200 cooldown units (~750 frames); four independent phases interleave,
  so windows long enough for a full hop (~700 frames of aim + create + create +
  transfer) on a *useful* tile are rare. The player provably stalls: measured
  runs spend most ticks waiting for a window that three other cones keep
  closing.
- **Meanie pressure everywhere.** A meanie needs only a *partially* visible
  player and a fully-visible tree within 10 tiles (`$19A1`/`$19C3`). With four
  scanners and the board's trees concentrated on the low ground where the
  player must start, the arm condition is satisfied on most low tiles, forcing
  hyperspaces (3 energy each, death below 3, `$215F`).
- **The energy economy decays under gaze.** Reclaiming the shell and pedestal
  a hop leaves behind (+3/+2) is what makes climbing affordable; here the
  abandoned bodies usually sit in someone's cone and are dismantled
  robot→boulder→tree (`$1A08`) before the player can re-aim, so reclaims pay
  +2 or +1 instead. Every hop leaks energy the board cannot replace (more
  enemies also means fewer trees at generation: `48 − 3·enemies`, §3).
- **Hunting is circular — until the seer is treated as a target.** The
  absorb-lock (`$1B8E`) forces all three sentries before the Sentinel, but
  gaining line of sight on any sentry at height 9 generally means standing
  where a height-9 sentry sees back. What breaks the circle is the
  **counterattack**: being seen starts the seer's ~120-unit drain countdown
  (`$0C20`), not an instant loss — absorbs have no facing requirement and a
  u-turn costs one keystroke, so an absorbable seer whose tile is landable
  inside its own countdown is absorbed instead of fled from (fleeing spends
  energy and leaves the circle intact).

With the counterattack tier (and transfers gated on the same danger basis as
builds, so a harmless partial glimpse cannot orphan a fresh pedestal), the
player beats 0335 in the simulator (295 actions, all four enemies cleared). It
remains the stress case: most of its length is spent waiting out four
interleaved cones, so any regression in the time model or the invariant shows up
here first.

## Sim-vs-live verification (landscapes 0000 and 0042)

Outcome matrix under the quantized live clock (world frames advance only in
deliberate run-to-PC windows) and the executor-true sim charges:

| landscape | sim | live | outcome match | live/sim actions | charged vs measured frames |
|---|---|---|---|---|---|
| 0000 | WON | WON | yes | 38 / 19 | 5479 / 3616 (1.52x, over-charged) |
| 0042 | WON | WON | yes | 43 / 41 | 5659 / 5680 (1.00x) |
| 0335 | LOST (honest clock) | not attempted | — | — | — |

The rows above predate the ROM-faithful settle/aim clock; under it the sim
action counts rise (0000 53, 0042 78, still won, seen 0 times) and the **live
0042 win regressed** — it now loses deterministically by being seen during the
(13,27)→(5,30) reclaim, a sim-vs-live aim-cost gap (the sim under-charges the
long reclaim aim, so an enemy rotates into view where it predicted safety).

`measured` comes from the game's own per-frame accumulator ($1335 advances
205/frame; 205^-1 = 5 mod 256 turns the delta into an exact frame count under
256).  Paths still diverge at the tile level (identical prefix 7 actions on
0000, 1 on 0042; shared strategic skeleton throughout).

**Live aim cost is outcome-determining.** Aim cost is the world frames that pass
while lining up a move, so it sets how far every rotating enemy turns before the
move lands; under-charging it places a body into a gaze the planner modelled as
empty — an outcome flip, not a rounding detail. The 0042 regression is exactly
that: the sim under-charged the long (13,27)→(5,30) reclaim aim, the enemy
rotated further than modelled, and the body was drained where safety was
predicted. The faithful charge is `_aim_frames(view) + _settle(verb, view)`
(bearing/pitch pan + cursor travel + scene-dependent redraw via `visible_edges`;
settle model in [render_cost.md](render_cost.md)), applied *before* advancing the
enemies. Both the reactive player and the A* search
([astar_player.md](astar_player.md)) charge this.

## Test

`sentinel/tests/test_player.py` — the player wins landscapes 0000 and seed 66
(typed 0042) alive and solvent, and no create or transfer leaves an object
inside an enemy's live scan cone POST-SETTLE (the player's built-in `--audit`,
`Player._account`, judged by the ROM's own visibility on the actual object) —
the settle-aware gate refusing an aim/settle exposure the plan-time window
would miss. Run `--audit` on the 0335 stress board to exercise that path.
