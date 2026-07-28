# Retrograde regression: where the A\* player loses the human's line

The recorded human wins are a ground-truth *line* through a landscape. Handing the
planner the human's board at event `i` and asking it to finish alone turns that line
into a ruler: the highest handover the planner cannot convert is the exact board its
search cannot recover — and the human's own move at that `i` is the move it missed.

Handovers are **bisected**, not walked. Each round probes `workers` evenly spaced
handovers inside the live interval concurrently, so the interval divides by
`workers+1` per round: 156 events settle in about three rounds rather than a hundred
sequential attempts. `--linear` keeps the exhaustive backward scan for a full
win/lose profile; `--indices` runs an explicit list.

```bash
python -m sentinel.tests.human_regress ls335.json --out out/ls335_regress.json --diagram
python -m sentinel.tests.human_regress ls335.json --planner phase    # score the phase player
python -m sentinel.isoview 335                       # the board at entry, no annotations
```

`--planner` selects which planner is scored: `astar` (the default,
[astar_player.md](astar_player.md)) or `phase` (the phase player,
[phase_player.md](phase_player.md)).

## The handover board

`human_regress.state_at(fixture, i)` is the human's PRE-action state at event `i`:

| from | what |
|---|---|
| `landscape.generate(seed)` | byte-exact terrain (never stored in the fixture) |
| fixture event `objects` | every object's tile / type / height / stack flags |
| fixture event `player`, `energy` | player slot, stance, aim angles, energy |
| fixture event `enemy_clock` | true mid-game facings + rotation/drain/update cooldowns |
| fixture event `cooldown_*` | the `$1335` bresenham accumulator and `$0C50` gate |

`$0CE5` (enemies-frozen-until-first-action) is cleared: mid-game, the enemies run.
Not recovered: `$0090`, which enemy slot `update_enemies` processes this round — the
round pointer starts wherever `generate` left it, so a handover's enemy phase is right
to within one round of updates.

## "Cannot win" is a budget statement

Each attempt runs `AStarPlayer` under a node budget, a per-search wall clock and a hard
per-attempt wall clock, all recorded in the artifact's `budgets`. Attempts are
independent, so the scan runs one **spawned** process per index (a fork aborts: the
numba LOS march leaves an OpenMP runtime in the parent) and the parent kills any that
outlives the cap. A signal-based cap does not work — a `SIGALRM` raised inside the
numba march corrupts the dispatcher.

Wall clock is contention-sensitive, so workers share the machine deliberately:
`march_batch` is `parallel=True` and an uncapped worker takes a thread per core, which
put 10 workers at load 117 on 24 threads and inflated attempts ~8x. Each worker is
pinned to `cores // batch` numba threads.

A capped attempt is still never reported as a loss directly — the scan re-runs the top
capped index **alone** at `escalate` x the cap (default 3x) and only then calls it.
Outcomes are `won` / `lost` / `capped`; `capped` in the artifact means undecided, not
defeat.

## The diagram

`sentinel/isoview.py` renders any `State` as a standalone isometric SVG:

- the 32x32 height field as a lit mesh — flat tiles flat, sloped tiles from their four
  ROM corner heights (`check_sloping_tile`'s `$73..$76` square, resolved through object
  stacks by `los._slope_corner_z`);
- objects as typed glyphs at their own `z_height`, drawn back-to-front along the
  isometric diagonal so near hides far;
- each live enemy's scan cone as a ground wedge on its true recorded facing
  (`FOV_HALF`), and the platform as a gold diamond;
- an action line as numbered arrows: **solid** = what the human did next from this
  board, **dashed** = what the planner did instead, badges stacked when several
  actions land on one tile;
- a panel with the board scalars, per-enemy bearing-to-player and cone offset, and the
  numbered action list with energy.

`human_regress --diagram` writes that SVG for the first losing handover into
`renders/`, which is what a correction is written against.
