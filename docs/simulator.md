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
| `sentinel/statecmp.py` | the labelled/tiered address schema shared with the live driver ([instrument.md](instrument.md)) |
| `sentinel/terrain.py` | height/slope nibble decode and the slope-facet surface |
| `sentinel/los.py` | the integer line-of-sight ray-march and sights aim vector, plus the ROM-faithful keyboard-aim buildability oracle (`landable_views` / `landable_view` / `landable_sweep_with_centres`) that sweeps the sights cursor at 1 px resolution — the ROM cursor-move step (`$9965`/`$9994`), each 1 px a distinct ray sub-angle — over the full ROM cursor clamp (cx `$10-$8F`, cy `$20-$9F`) |
| `sentinel/los_jit.py` | numba fast-march of the hot LOS inner loop (bit-identical to `los.py`); auto-used when numba is present, else `los.py` falls back to pure Python |
| `sentinel/aim.py` | the one player-action aim/LOS layer (`resolve`/`gate`/`propose`) wrapping `los.aim_target`/`landable_view` — the `$1B40-$1B46` action gate |
| `sentinel/aimcost.py` | keyboard-aim geometry: keystrokes/rounds to pan a heading (bearing on the 8-unit lattice, pitch on 4, u-turn-aware) |
| `sentinel/pancost.py` | per-notch pan redraw cost derived from `pan_viewpoint $10B7` (see [render_cost.md](render_cost.md)) |
| `sentinel/projector.py` | `plot_world $2625` terrain projector, ported bit-exactly; feeds the render-cost proxy |
| `sentinel/rendercost_py65.py` | exact `plot_world` frame cost by running the real 6502 in py65, memoized; optional and ROM-gated |
| `sentinel/actioncost.py` | per-action world-advance cost in `enemies.step` units — the ROM-cited dither/replot frame counts and the `$1335/$0C50` frame→tick cadence; live pricing goes through `playerbase._settle` |
| `sentinel/actions.py` | absorb / create / transfer / hyperspace / win (mechanics; the LOS gate is the caller's, via `aim.py`) |
| `sentinel/energy.py` | the energy economy (`$2136` table `$214F`, 6-bit mask, underflow) |
| `sentinel/landscape.py` | `generate(landscape) -> State`: the from-scratch board generator |
| `sentinel/relative.py` | object-relative bearing/distance/vertical angle and the enemy field-of-view + visibility test |
| `sentinel/enemies.py` | the enemy round and the frame clock (`advance_frame`/`advance_frames`) |
| `sentinel/threat.py` | enemy queries built on the above: any-rotation tile exposure, gaze distance, ticks-until-seen, meanie safety, drain-over-window |
| `sentinel/game.py` | `Game`, a facade tying the above together |
| `sentinel/playerbase.py` | shared player machinery: world clock, geometry, gaze windows, aim cost, action firing, run loop |
| `sentinel/player.py` | reactive priority-driven player |
| `sentinel/astar_player.py` | A* / best-first planning player ([astar_player.md](astar_player.md)) |

## Landscape numbers are what you TYPE

A landscape's canonical id is the number a player keys into the game. The ROM stores it
packed-BCD and seeds the PRNG from those bytes, so the seed is the typed digits read as
**hex**: typed `42` seeds 66, typed `335` seeds 821. Use the typed number everywhere and
let the shim convert:

```python
Game.typed(335)          # the ls335 everyone means (7 enemies, player (11,17))
landscape.seed_for(335)  # -> 821, the raw seed
Game.new(821)            # raw seed: the same board, the long way round
Game.new(335)            # a DIFFERENT board -- 335 is a seed here, not a typed code
```

`Game.typed(42)` and `Game.typed(335)` reproduce the human-win fixtures object for
object; anything keyed on a raw seed is testing a board nobody can type.

## Usage

```python
from sentinel import Game
g = Game.typed(42)                   # build the board (no emulator)
print(g.player_xy(), g.energy)       # (13, 29) 10
g.create(g.state.obj_type, (x, y))   # player actions
g.step_enemies()                     # advance the world one round
if g.won(): ...
```

`State` is a mutable `bytearray`-backed image; `Game.clone()` deep-copies it so a
search can branch without side effects.

## The clock

`enemies.advance_frame(state, plotting=False)` is one video frame: the
`$9663`/`$1317` raster cooldown tick **first**, then `UPDATES_PER_FRAME` (8)
`update_enemies` passes — one full `$0090` cursor sweep, so every slot is considered
each frame. `plotting=True` suppresses the sweep, modelling the ROM's re-plot/scroll
spans in which the foreground never reaches `update_enemies`; only the cooldown clock
advances. This clock is exact against the live ROM — see
[instrument.md](instrument.md) for the gate and [plan_fidelity.md](plan_fidelity.md)
for the measurements.

## Validation

Every mechanic is differentially validated against the real 6502 code via a py65
harness, then the ROM captures are frozen as JSON goldens replayed by CI
(`sentinel/tests/`):

| Fixture | What it pins |
|---------|--------------|
| `golden_prng.json` | the PRNG stream |
| `golden_los.json` | line-of-sight over sampled aim rays |
| `golden_actions.json` | absorb / create / transfer / energy |
| `golden_landscape.json` | full board generation (terrain + object tables + PRNG state) |
| `golden_relative.json` | relative geometry + enemy full-visibility |
| `golden_enemies.json` | enemy-array trajectories, captured every 25 rounds over 400 rounds |
| `golden_meanie.json` | the full meanie lifecycle + the failed-attempt path |
| `golden_projector.json` | `plot_world` projection |
| `golden_pan_cost.json` | per-notch pan frame cost |
| `golden_render_cost.json` | exact py65 render cost |

The meanie lifecycle (tree → meanie → forced hyperspace → relocation, energy spend or
death) is a stateful side effect of `enemies.step()`, pinned over the full object +
enemy/meanie state (object table, PRNG, tiles, player/energy, death/hyperspace flags)
round for round on landscape 2024 to round 2486, plus the failed-attempt path
(landscape 49).

The arctan (`$3B00/$3C01`) and hypotenuse (`$3D02`) coefficient tables are reproduced
from closed-form expressions (verified byte-exact vs the ROM), so no game data is
embedded. The two-probe `$0014` exposure byte (`$80` full / `$40` partial / `0`
unseen) — the meanie trigger — is reconstructed bit-exact.

## Coverage and gaps relative to `docs/gameplay.md`

`docs/gameplay.md` is the authoritative mechanics spec. The **bit-exact core** matches it
and is golden-pinned: the PRNG, tile geometry, LOS `$1CDD`, enemy relative geometry / FOV
cone / two-probe `$0014` exposure, the enemy round (rotate / drain / downgrade chain /
cooldown decrement cadence), landscape generation, and the meanie lifecycle. The frame
clock is not golden-pinned; it is gated frame-for-frame by the divergence instrument.

### Unmodelled mechanics — in the spec, absent from the model

- **PRNG-driven landing coordinates.** `actions.hyperspace` (`do_hyperspace $2156`, the
  routine a meanie forces) is faithful, and `win` is gated on the landscape-complete flag
  (`$0CDE` bit6+7, set by a hyperspace **from the platform tile**), running
  absorb→build→transfer→hyperspace with the 3-energy cost and death-if-underfunded. The
  **landing coordinates** of a hyperspace (and of a meanie relocation) are deliberately
  unread: a faithful solver must not read PRNG state. The PRNG draw *rate* is likewise
  unmodellable.
- **Player u-turn action.** The free 180° facing flip (`$1B2F` EOR `$80`) is priced in
  `aimcost`/`playerbase` but is not exposed as a player action in `actions.py`.
- **Exposure-bar aggregation** (`calculate_player_exposure $191F` / `set_bar_state
  $194D` / `$0C4F`) is not modelled; the underlying two-probe `$0014` is.
- **Meanie death-credit `$0C1C = 4`** is not written after a meanie hyperspaces the
  player (`$0C1C` is not in `memmap.py`). Affects only death-screen attribution.
- Sound side effects carry no gameplay state and are out of scope.

### Aim cost (`playerbase._aim_frames` / `_step_aim_frames`)

Priced mechanism for mechanism against the executor's key sequence; the per-notch
redraw term is in [render_cost.md](render_cost.md).

- **Body pan is two keystroke ramps** via `pancost.pan_frames`, each notch followed by one
  `plot_world`: horizontal `$10EE` = `H_SCROLL` 16 scroll steps per ±8 bearing notch,
  vertical `$1135` = `V_SCROLL` 8 steps per ±4 pitch notch. Notch counts come from
  `aimcost.h_press_count` (u-turn-aware, returns `(n_uturn, n_step)`) and `aimcost.v_steps`.
- **U-turn**: one action tap (`UTURN_FRAMES`), 0 scroll frames, no redraw.
  `h_press_count` takes it only when it strictly lowers the keystroke count (crossover at
  `d >= 9` lattice steps). Keying one also *unfreezes the world mid-aim* — `$12D5 CMP #$22
  / BCS $12DE` lets codes >= `$22` skip the sights-on check and fall into `$12E1 LSR
  $0CE5`; `_aim_unfreeze_split` returns the frames of aim elapsing before that.
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
  create/absorb parked the cursor over; otherwise it pays a full `_aim_frames`. Live,
  the reuse predicate is the **driver's** (`sights_live_on() and committed_bearing() ==
  view bearing`), adopted into `last_bearing`/`cursor` by `LiveMixin._sync_aim_state` at
  every `_observe` (the live `_fire` override bypasses `BasePlayer._fire`'s bookkeeping).

### Threat-layer gaps

- **Gaze-entry double penalty is not a query.** Entering a gaze costs ≥1 off the current
  body *plus* continued draining/downgrading of the abandoned body. The players price
  only the body they stand in (`playerbase._player_window`); the abandoned-body loss is
  realised only by actually stepping the enemies, never priced up-front.
- **`enemies.meanie_threat` omits the tree gate.** The full
  `attempt_to_create_meanie $19A1` condition is partial player visibility **plus** a
  fully-visible tree within 10 tiles in both axes (no tree→player LOS test, matching the
  ROM); `meanie_threat` tests only partial visibility, so it over-reports.
  `playerbase._tree_near`/`_meanie_window` apply the tree gate.
- Reclaim-energy and sentry-value are search policy, not model gaps: the model supports
  the underlying absorbs (energy gain; the `$1BE0` face-back is in `actions.create`).
