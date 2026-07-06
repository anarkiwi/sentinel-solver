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
| `sentinel/los.py` | the integer line-of-sight ray-march and sights aim vector |
| `sentinel/actions.py` | absorb / create / transfer / win and the energy economy |
| `sentinel/landscape.py` | `generate(landscape) -> State`: the from-scratch board generator |
| `sentinel/relative.py` | object-relative bearing/distance/vertical angle and the enemy field-of-view + visibility test |
| `sentinel/enemies.py` | one game round of enemy rotation / targeting / draining / cooldowns |
| `sentinel/threat.py` | planner-facing enemy queries built on the above: any-rotation tile exposure, gaze distance, ticks-until-seen, meanie safety, drain-over-window |
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

Bit-exact results established during the port: PRNG and LOS 12,800/12,800; landscape
generation byte-for-byte; `divide_and_arctan` 0/4040, relative angles 0/376,
vertical angle 0/3000, enemy full-visibility 0/496; the enemy round advance
0-divergence over 400 rounds on validated landscapes.

The arctan (`$3B00/$3C01`) and hypotenuse (`$3D02`) coefficient tables are
reproduced from closed-form expressions (verified byte-exact vs the ROM), so no
game data is embedded.

## Known approximations

- The two-probe `$0014` exposure byte stores a full/partial classification whose
  exact multi-probe bit-plumbing is not fully reconstructed; a rare rotated-angle
  target may be classed partial where the ROM classes it fully visible.
- The meanie lifecycle (tree → meanie → forced hyperspace) is exposed as the
  `enemies.meanie_threat()` capability query rather than a stateful side effect.
- The rendering/sound side effects of the enemy update (re-plotting, the
  energy-discharge that scatters trees when an enemy is absorbed) are not modelled;
  they do not change the gameplay state a strategy search reasons about.
