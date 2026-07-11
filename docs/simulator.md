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
| `sentinel/actioncost.py` | per-action world-advance cost in `enemies.step` units — the ROM-cited dither/redraw/tune frame counts and the `$1335/$0C50` frame→tick cadence |
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

Bit-exact results established during the port: PRNG and LOS 12,800/12,800; landscape
generation byte-for-byte; `divide_and_arctan` 0/4040, relative angles 0/376,
vertical angle 0/3000, enemy full-visibility 0/496; the enemy round advance
0-divergence over 400 rounds on validated landscapes. The meanie lifecycle is
validated over the full object + enemy/meanie state (object table, PRNG, tiles,
player/energy, death/hyperspace flags) round for round through spawn, hunt, forced
hyperspace and a drain-death (landscape 2024, 2,486 rounds) plus the failed-attempt
path (landscape 49); a 48-landscape / 700-round sweep of the whole enemy update is
0-divergence.

The arctan (`$3B00/$3C01`) and hypotenuse (`$3D02`) coefficient tables are
reproduced from closed-form expressions (verified byte-exact vs the ROM), so no
game data is embedded.

The two-probe `$0014` exposure byte (`$80` full / `$40` partial / `0` unseen) is
reconstructed and validated bit-exact against the ROM: it is the trigger the meanie
lifecycle turns on (an enemy that sees the player partially arms a meanie), so the
0-divergence meanie runs above exercise it across rotated enemy angles.

The meanie lifecycle (tree → meanie → forced hyperspace → player relocation, energy
spend or death) is modelled as a stateful side effect of `enemies.step()` and pinned
by `golden_meanie.json`; `enemies.meanie_threat()` is only a query over the same
partial-visibility test.

## Coverage and gaps relative to `docs/gameplay.md`

`docs/gameplay.md` is the authoritative mechanics spec. The **bit-exact core**
matches it and is golden-pinned: the PRNG, tile geometry, LOS `$1CDD` (true
triangular-facet surface + the look-up rule), enemy relative geometry / FOV cone /
two-probe `$0014` exposure, the enemy round (rotate / drain / downgrade chain /
cooldown cadence and the `$1335`+`$0C50` frame→tick divider), landscape generation,
and the meanie lifecycle. The items below are where the model **diverges from or
omits** the spec, grouped by kind. (Audited against the ROM and the code.)

### Correctness divergences — the model contradicts the ROM (should be fixed)

*(The absorb-lock, win-condition, `meanie_safe` and landscape-BCD items previously
listed here have been fixed; see the notes folded into the sections below.)*

### Unmodelled mechanics — present in the spec, absent from the model

- **Player u-turn action.** `actions.py` now exposes a faithful player `hyperspace`
  (`do_hyperspace $2156`, the same routine a meanie forces), used by the win path —
  `won` is now the landscape-complete flag (`$0CDE` bit6+7, set by a hyperspace
  **from the platform tile**), not merely standing on it, and `win` runs
  absorb→build→transfer→hyperspace with the 3-energy cost and death-if-underfunded.
  Its PRNG-driven **landing** remains deliberately unmodelled for planning — a
  faithful solver must not read PRNG state, so a hyperspace destination is off-limits
  and the solver must not rely on it. The **u-turn** (a free 180° facing flip, §4)
  is still not exposed.
- **Exposure-bar aggregation** (`calculate_player_exposure $191F` /
  `set_bar_state $194D` / `$0C4F`) is not modelled; the underlying two-probe `$0014`
  is. Not needed by a solver, but a literal spec gap.
- **Meanie death-credit `$0C1C = 4`** is not written after a meanie hyperspaces the
  player (`$0C1C` isn't even in `memmap.py`). Affects only death-screen attribution.

### Solver-cost / strategy-fidelity gaps — newer §7 / "Writing a solver" material

The strategy sections of gameplay.md added a time/aim cost model and threat concepts
the cost/threat layer only partly captures:

- **Aim cost sums the axes; diagonal steering isn't modelled.** `aimcost.pan_steps`
  returns `|Δh|/8 + |Δv|/4`, but §4 (`move_sights $9958` moves both cursor axes per
  frame) says off-axis cost is ≈ `max`, not the sum. And the **sights-toggle
  re-centre** (`$134C` → cursor (80, 95)) isn't modelled as a state transition, so an
  off/on toggle reads as free. Both skew the aim-time estimate. (`aimcost` *does*
  model the u-turn as a +16-step single keystroke, and `actioncost` models the
  frame→tick cadence — those are present.)
- **Gaze-entry double penalty is not a query.** §7 / core queries: entering a gaze
  costs ≥1 off the current body *plus* continued draining/downgrading of the
  abandoned body. `threat.drain_over_window` tracks only the current player object;
  the abandoned-body loss is realised only if you actually `step` the enemies, never
  priced up-front.
- **Any-rotation exposure counts only full visibility.** `threat.is_exposed` /
  `exposed_tiles` test `["full"]`, so a partially-visible tile (head seen, base
  blocked) reads as safe — yet partial visibility is exactly what arms a meanie (§6).
  Under-reports the hazard set.
- **The two meanie queries disagree.** `threat.meanie_safe` (with the spurious
  tree→player condition) and `enemies.meanie_threat` (partial-visibility only, no
  tree check) use different, mutually inconsistent conditions.
- **Reclaim-energy and sentry-value are search policy, mechanically supported.** The
  reclaim loop (absorb your own boulders/shells, §7) and treating a sentry-absorb as
  hazard removal are solver heuristics; the model supports the underlying absorbs
  (energy gain; the `$1BE0` face-back is in `actions.create`), so these are for the
  search to exploit, not model gaps.

## Known approximations

- The rendering/sound side effects of the enemy update (re-plotting, sound, the
  "object being plotted" update guard) are not modelled; they do not change the
  gameplay state a strategy search reasons about.
