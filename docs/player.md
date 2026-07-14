# The reactive player (`sentinel/player.py`)

A tick-by-tick greedy player over the `sentinel/` model. No search tree, no
lookahead branching, and **no PRNG reads**: each decision tick observes the live
`State` and commits one action. Wins landscape 0 in 21 actions with zero energy
drained.

```bash
python -m sentinel.player 0        # play landscape 0, print the action trace
```

## Decision loop

Each tick picks the first applicable action, then advances the world by that
action's real duration (aim pan + settle frames via `enemies.advance_frames`),
so enemies rotate, target and drain while the player aims — there are no free
moves.

1. **Win move** — Sentinel absorbed: create a robot on the platform tile,
   transfer in, hyperspace (`do_hyperspace $2156` sets the complete flag).
2. **Dissolve a meanie** — absorb it while absorption is still possible.
3. **Absorb the Sentinel** — dead last among absorbs (the `$1B8E` slot-0 lock),
   only with the endgame affordable (energy ≥ 2 before the +4) and no live meanie.
4. **Transfer up** — into the highest aim-landable robot that raises the eye.
5. **Reclaim / harvest** — absorb old shells (+3) and spent pedestals (+2) from
   the new vantage (never the hop in progress), trees while below headroom.
6. **Climb** — the ping-pong hop: boulder on the best safe landable tile, robot
   on the pedestal (stacking more boulders if still short), transfer.
7. **Wait** — otherwise idle a few frames and re-observe.

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
dither/redraw frame counts (`sentinel/actioncost.py`); both advance the world
before/after the action fires.

Landability queries use one cheap primary-plane sweep per tick, falling back to
one full pitch-band sweep only for down-looks that a single-ray visibility check
first confirms plausible.

## Test

`sentinel/tests/test_player.py` — the player wins landscape 0 alive and solvent,
and every transfer in the winning trace landed outside every enemy's live cone.
