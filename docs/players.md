# The search player

`driver/search_player.py` plays the real game. It plans on the ROM itself, running
cycle-exact on vagrant-moses, so a candidate's outcome is the game's own: there is no model
of time, of the enemies or of a move's effect to get wrong.

```bash
python -m driver.play_player 335 --emu          # open loop on the emulator
python -m driver.play_player 335                # live in VICE, recording an AVI
python -m driver.play_player 335 --preset narrow --width 8 --commit 2
```

`--preset` picks the knobs and `--width/--depth/--branch/--commit` override them one by one:

| preset | width | depth | branch | commit |
|---|---|---|---|---|
| `wide` (default) | 16 | 30 | 3 | 10 |
| `narrow` | 4 | 12 | 2 | 1 |

## How it plans

A receding-horizon beam over machine states (`SearchPlayer.run`). From the current machine it
searches `depth` plies, or until a child wins, then plays the first `commit` moves of the best
line and searches again from where they leave it.

| part | what it is |
|---|---|
| moves | `moves.views` over `moves.Board`: every candidate of every parent, each with the frame-cheapest keyboard view of its tile ([view selection](architecture.md#view-selection-viewspy)); `moves.action_views` re-aims each later action of a hop at the stack its creates leave |
| expansion | each move one `kbd_prog.compile_move` program; one `prog.forks` (`run_forks`, `keep=True`) per parent, the parents concurrently with the cores shared by program count; `kbd_prog.move_facts` read off each child |
| selection | `moves.keep_best(width)` under `moves.lexicographic`, ties to fewer frames; identity is the machine state (`machine_key`: RAM digest, cycle, registers, monitor state), so transpositions merge; a refused move (`status` not `HALT_OK`) is no child |
| execution | the committed moves are played on the session with the same programs (`prog.execute`) and each move's facts and machine key checked against the child that predicted it (`audit`, `lookahead_audit` in the run's JSON) |

Each parent's expansion is memoized on its machine key and carried into the next search, so
a replan re-expands only what the new root's beam adds.

## The move set

`moves.moves` generates, per board, numpy over the object and tile arrays of many boards:

* **absorb** — an object topping a keyboard-landable tile that is a tree, a sentry, a spent
  robot or boulder at or below the eye, or the Sentinel once it is the last enemy and the
  endgame is affordable; enemies first, then by energy value. Nothing once the Sentinel is
  gone: `$1B91` reads `objects_flags[0]` absolutely and refuses every later absorb.
* **hop** — `k` boulders (at most three), a robot and the transfer onto stackable ground
  (bare flat tile, boulder or platform) that raises the eye, affordable and leaving either a
  robot's energy or fuel in reach; ranked by height gained per energy, then fuel.
* **mount** — a transfer into a robot already standing higher.
* **wait** — `WAIT_QUANTUM` (60) frames.
* **relocate** — the hyperspace, wherever its 3-energy toll is in hand. Its landing is
  PRNG-driven; the child is wherever the machine puts the body.

Absorbs, hops and mounts are each capped at `branch` per parent. With the Sentinel gone the
platform tile takes a hop (no boulders, ranked first) or a mount whatever its height, and
relocating from the platform is the win (`compile_move("finish")`). A move's waits stop past
the flag stores of `kill_player` (`$1A05`), the toll underflow (`$217D`) and the win
(`$219B`), so a death or the win ends the move there (`Prog.guard`).

## The objective

`moves.lexicographic`: a child lives while the player does (a win counts as alive); among the
living the order is won, fewer enemies, higher eye, more energy, the first key most
significant. `keep_best` breaks ties to fewer frames, then the lower position, so the order is
total.

## Live in VICE

Without `--emu`, VICE boots and records as for any live run and the search plans on an
emulated session of the same landscape (`play_player.emu_session`). Each committed move's
program is also `prog.interpret`ed on VICE and its `move_facts` compared with the session's
(`live_audit`, the entry as move 0). The first disagreement is logged, recorded as
`divergence` and ends the run; a win counts only with VICE's own `$0CDE` bit 6
(`landscape_done`). The run's JSON is `out/play_player_NNNN_search.json`, the AVI
`renders/player_lsN_search.avi` ([media](media.md)).

## Tools

* `python -m driver.decision_audit N [--game wide] [--preset narrow] [--decisions 12]` plays
  the `--game` search's game once on the emulator, clones the machine before each move, and at
  each decision forces every candidate the search generates there on a fork and continues it
  with a fresh `--preset` search to the end. It reports won, alive, energy, eye and actions
  per candidate, the chosen one starred (`--out` writes the rows as JSON).
* `python -m driver.visibility_audit N... [--preset wide]` plays each board's search game,
  caching the RAM after every move, and on every tile whose object stack is entry's compares
  the ROM's enemy and player sight checks on that RAM against the entry RAM with the
  observers moved to where they stand.
* `python -m driver.prune_audit N [--preset wide]` records every fork of a board's search
  game with what the entry-time sight table (`sentinel.sight_table`) says of a hop's landing
  and whether a beam kept it, and scores pruning rules on forks and frames saved against
  kept forks lost.
* `python -m driver.search_player N PLY [--preset P] [--each]` builds search ply `PLY`'s fork
  batches from the entry to landscape `N` (`ply_batches`) and times each `prog.forks`, a
  reproducible workload for profiling.
* `python -m driver.emu_suite` plays the suite boards with the search player on the emulator
  in parallel processes (`--preset`, `--jobs`, `--tabulate` re-reads a run).

## Results

`python -m driver.emu_suite --out docs/results/search.json --readme` plays the ten suite
boards (0 42 60 110 298 321 335 373 1442 9716) and regenerates the README results table from
`docs/results/search.json` and the live runs in `docs/results/live.json`. The README table is
the record; it is not repeated here.

## Retired players

The phase, greedy and beam players planned on a hand-written world clock (an enemy and frame
model of the ROM) and were retired with it. The live ls335 and ls1442 phase-player wins in
`docs/media/` came from them; their `docs/results/live.json` rows carry `player: phase`.
