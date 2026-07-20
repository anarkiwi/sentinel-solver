# The `sentinel/` simulator

A standalone, bit-exact forward model of *The Sentinel* (C64). It runs with no
emulator; the real 6502 code is used only as a test-time oracle, and its outputs
are frozen as golden fixtures so CI proves correctness without the copyrighted
ROM image.

## Modules

| Module | Role |
|--------|------|
| `sentinel/memmap.py` | RAM addresses, object types, the interleaved tile index |
| `sentinel/prng.py` | the 40-bit LFSR `prnd` and landscape seeding |
| `sentinel/state.py` | the one canonical state: a 64 KB `bytearray` laid out like the game's RAM, with typed object-array views |
| `sentinel/terrain.py` | height/slope nibble decode and the slope-facet surface |
| `sentinel/los.py` | the integer line-of-sight ray-march and sights aim vector, plus the ROM-faithful keyboard-aim buildability oracle (`landable_views` / `landable_view` / `landable_sweep_with_centres`) that sweeps the sights cursor at 1 px resolution — the ROM cursor-move step (`$9965`/`$9994`), each 1 px a distinct ray sub-angle — over the full ROM cursor range |
| `sentinel/los_jit.py` | numba fast-march of the hot LOS inner loop (bit-identical to `los.py`, ~11x faster on full sweeps); auto-used when numba is present, else `los.py` falls back to pure Python |
| `sentinel/aim.py` | the one player-action aim/LOS layer (`resolve`/`gate`/`propose`) wrapping `los.aim_target`/`landable_view` — the `$1B40-$1B46` action gate |
| `sentinel/aimcost.py` | keyboard-aim geometry: keystrokes/rounds to pan a heading (bearing on the 8-unit lattice, pitch on 4, u-turn-aware) |
| `sentinel/actioncost.py` | per-action world-advance cost in `enemies.step` units — the ROM-cited dither/redraw/tune frame counts and the `$1335/$0C50` frame→tick cadence. `action_rounds` has zero callers repo-wide; live pricing goes through `playerbase._settle` |
| `sentinel/actions.py` | absorb / create / transfer / win (mechanics; the LOS gate is the caller's, via `aim.py`) |
| `sentinel/energy.py` | the energy economy (`$2136` table `$214F`, 6-bit mask, underflow) |
| `sentinel/landscape.py` | `generate(landscape) -> State`: the from-scratch board generator |
| `sentinel/relative.py` | object-relative bearing/distance/vertical angle and the enemy field-of-view + visibility test |
| `sentinel/enemies.py` | one game round of enemy rotation / targeting / draining / cooldowns |
| `sentinel/threat.py` | enemy queries built on the above: any-rotation tile exposure, gaze distance, ticks-until-seen, meanie safety, drain-over-window |
| `sentinel/game.py` | `Game`, a facade tying the above together |

## Usage

```python
from sentinel import Game
g = Game.new(42)                     # build the board (no emulator)
print(g.player_xy(), g.energy)       # (14, 27) 10
g.create(g.state.obj_type, (x, y))   # player actions
g.step_enemies()                     # advance the world one round
if g.won(): ...
```

`State` is a mutable `bytearray`-backed image; `Game.clone()` deep-copies it so a
search can branch without side effects.

## Validation

Every mechanic is differentially validated against the real 6502 code via a py65
harness, then the ROM captures are frozen as JSON goldens replayed by CI:

| Fixture | What it pins |
|---------|--------------|
| `golden_prng.json` | the PRNG stream |
| `golden_los.json` | line-of-sight over sampled aim rays |
| `golden_actions.json` | absorb / create / transfer / energy |
| `golden_landscape.json` | full board generation (terrain + object tables + PRNG state) |
| `golden_relative.json` | relative geometry + enemy full-visibility |
| `golden_enemies.json` | enemy-array trajectories over 400 rounds |
| `golden_meanie.json` | the full meanie lifecycle + the failed-attempt path |

Bit-exact results: PRNG and LOS 12,800/12,800; landscape generation byte-for-byte;
`divide_and_arctan` 0/4040, relative angles 0/376, vertical angle 0/3000, enemy
full-visibility 0/496; the enemy round advance 0-divergence over 400 rounds, and a
48-landscape / 700-round sweep of the whole enemy update 0-divergence. The meanie
lifecycle is validated over the full object + enemy/meanie state (object table, PRNG,
tiles, player/energy, death/hyperspace flags) round for round through spawn, hunt,
forced hyperspace and a drain-death (landscape 2024, 2,486 rounds) plus the
failed-attempt path (landscape 49).

The arctan (`$3B00/$3C01`) and hypotenuse (`$3D02`) coefficient tables are reproduced
from closed-form expressions (verified byte-exact vs the ROM), so no game data is
embedded.

The two-probe `$0014` exposure byte (`$80` full / `$40` partial / `0` unseen) is
reconstructed and validated bit-exact: it is the meanie trigger (an enemy that sees the
player partially arms a meanie). The meanie lifecycle (tree → meanie → forced hyperspace
→ relocation, energy spend or death) is a stateful side effect of `enemies.step()`,
pinned by `golden_meanie.json`; `enemies.meanie_threat()` is only a query over the same
partial-visibility test.

## Coverage and gaps relative to `docs/gameplay.md`

`docs/gameplay.md` is the authoritative mechanics spec. The **bit-exact core** matches it
and is golden-pinned: the PRNG, tile geometry, LOS `$1CDD` (true triangular-facet surface
+ the look-up rule), enemy relative geometry / FOV cone / two-probe `$0014` exposure, the
enemy round (rotate / drain / downgrade chain / cooldown decrement cadence), landscape
generation, and the meanie lifecycle. The frame→round scheduling (`advance_frame`: the
`$1335`+`$0C50` divider and the `$0090` cursor) is *not* golden-pinned — it is gated
frame-for-frame by the divergence instrument and currently diverges (below).

### Unmodelled mechanics — in the spec, absent from the model

- **Player u-turn action.** `actions.py` exposes a faithful player `hyperspace`
  (`do_hyperspace $2156`, the routine a meanie forces), used by the win path: `won` is
  the landscape-complete flag (`$0CDE` bit6+7, set by a hyperspace **from the platform
  tile**), not merely standing on it, and `win` runs absorb→build→transfer→hyperspace
  with the 3-energy cost and death-if-underfunded. The PRNG-driven **landing** is
  deliberately unmodelled: a faithful solver must not read PRNG state, so a hyperspace
  destination is off-limits. The **u-turn** (free 180° facing flip, §4) is not exposed as
  a player action (it is priced in `aimcost`, below).
- **Exposure-bar aggregation** (`calculate_player_exposure $191F` / `set_bar_state
  $194D` / `$0C4F`) is not modelled; the underlying two-probe `$0014` is.
- **Meanie death-credit `$0C1C = 4`** is not written after a meanie hyperspaces the
  player (`$0C1C` is not in `memmap.py`). Affects only death-screen attribution.

### Aim cost (`playerbase._aim_frames` / `_step_aim_frames`)

Priced mechanism for mechanism against the executor's key sequence; see
[render_cost.md](render_cost.md) for the per-notch redraw term and its open gap.

- **Body pan is two separate keystroke ramps**, each notch followed by one `plot_world`:
  horizontal `$10EE` = `H_SCROLL` 16 scroll steps per ±8 bearing notch, vertical `$1135`
  = `V_SCROLL` 8 steps per ±4 pitch notch. Notch counts come from
  `aimcost.h_press_count` (u-turn-aware, returns `(n_uturn, n_step)`) and
  `aimcost.v_steps`.
- **U-turn** (`$1B2F` EOR $80): one keystroke (`TAP_FRAMES`), 0 scroll frames, no
  redraw. `h_press_count` takes it only when it strictly lowers the keystroke count
  (crossover at `d >= 9` lattice steps).
- **Cursor — DERIVED, not fitted.** `move_sights $9958` steps both axes in one call at
  1 px per gated scan, so a drive costs `max(|Δcx|, |Δcy|)` scans, plus
  `CURSOR_RAMP = popcount($6B) = 5` scans skipped by the `$0CC8` auto-repeat mask
  (reloaded `#$6B` at `$11E0`, one skip per set bit at `$11F6 ASL / BCS`) before the
  first move. Zero when the cursor is already parked.
- **Sights toggle is a state transition.** A same-bearing reuse (`last_bearing == (h,v)`)
  keeps sights on and drives from the live cursor at 0 toggle cost; otherwise
  `TOGGLE_FRAMES` is charged and `initialise_sights $134C` re-centres the cursor to
  `SIGHTS_CENTRE = (80, 95)`, which is where the drive then starts.
- **`_step_aim_frames`: a transfer charges 0 aim only on a bearing reuse** — the executor
  sends no aim keys then, `$21` firing on the object the preceding same-tile
  create/absorb parked the cursor over. On a mismatched bearing a transfer drives the
  full view (`live_player._drive_transfer_aim`) and pays the same `_aim_frames` as every
  other verb.
- Live, the reuse predicate is the **driver's** (`sights_live_on() and
  committed_bearing() == view bearing`). `LiveMixin._sync_aim_state` adopts it into
  `last_bearing`/`cursor` at every `_observe`, because the live `_fire` override bypasses
  `BasePlayer._fire`'s bookkeeping; without it every live step was charged a full aim the
  executor never drove.

### Threat-layer gaps

- **Gaze-entry double penalty is not a query.** Entering a gaze costs ≥1 off the current
  body *plus* continued draining/downgrading of the abandoned body.
  `threat.drain_over_window` tracks only the current player object; the abandoned-body
  loss is realised only by actually stepping the enemies, never priced up-front.
- **Any-rotation exposure counts only full visibility.** `threat.is_exposed` /
  `exposed_tiles` test `["full"]`, so a partially-visible tile (head seen, base blocked)
  reads as safe — yet partial visibility is exactly what arms a meanie. Under-reports the
  hazard set.
- **The two meanie queries disagree.** `threat.meanie_safe` (with a spurious tree→player
  condition) and `enemies.meanie_threat` (partial visibility only, no tree check) use
  mutually inconsistent conditions.
- Reclaim-energy and sentry-value are search policy, not model gaps: the model supports
  the underlying absorbs (energy gain; the `$1BE0` face-back is in `actions.create`).

## Known divergences — defects, not accepted error

- **Plot-span enemy suppression is not modelled.** The ROM's "object being plotted"
  update guard suppresses `update_enemies` during re-plot/scroll frames; the sim advances
  enemies through them. Since an aim is mostly plot frames, this shifts enemy phase
  across every move. Instrumented by [instrument.md](instrument.md); gated by
  `driver/test_enemy_sim_divergence.py`.
- Sound side effects carry no gameplay state and are correctly out of scope.
