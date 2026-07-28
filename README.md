***Claude - including Fable - was unable to solve this simple game from 1986 without my help. Is what it is.***


# The Sentinel — ROM-faithful model + live driver

A ROM-faithful model of **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64, plus a live driver that plays the real game in
[VICE](https://vice-emu.sourceforge.io/) (asid-vice) by keyboard and records an AVI.
Transition primitives are validated byte-for-byte against the real 6502 code (golden
fixtures, so CI proves them without the ROM); the enemy clock is gated frame-for-frame
against the running game by the divergence instrument.

The phase player wins **live, on the real game**, verified by the ROM's own
landscape-complete flag (`$0CDE` bit 6) — landscape 335 in 66 actions, final energy 25:

![the phase player winning landscape 335 live in VICE](docs/media/ls335_phase_win.png)

```bash
pip install -r requirements.txt
pytest -n auto

python -m sentinel.phase_player 335     # offline, prints the action trace
python -m driver.play_player 335        # live in VICE, records an AVI
python -m driver.instrument 42          # race the model against the ROM, frame for frame
```

| landscape | enemies | offline | live |
|---|---|---|---|
| 0 | 1 | 16 actions | — |
| 42 | 2 | 35 | — |
| 60 | 7 | 55 | — |
| 110 | 3 | 53 | — |
| 298 | 7 | 34 | — |
| 321 | 7 | 78 | — |
| 335 | 7 | 73 | **66 actions** |
| 373 | 7 | 65 | — |

A landscape is identified by one number: the one a player types on the keypad. Every tool
here — `driver.play_player`, `sentinel.phase_player`, `sentinel.player`, `sentinel.isoview`
— takes exactly that number, and `Game.typed(335)` builds the same board offline.

## Layout

| Area | Path | Role |
|------|------|------|
| Model | `sentinel/` | standalone forward model — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator). Transition primitives are byte-for-byte against the 6502; see [architecture.md](docs/architecture.md) for what is exact and what is not |
| Phase player | `sentinel/phase_player.py` | the default player: freedom first, then convert. Wins all eight measured boards |
| Reactive player | `sentinel/player.py` | tick-by-tick greedy player over the same `BasePlayer` |
| Landscape analyzer | `sentinel/landscan.py` | enemy count and terrain shape per landscape, to match one board to another |
| Driver | `driver/` | boot, enter a landscape, run memory-verified live keyboard operations (aim → fire → verify), record. Imports only `sentinel/` |
| Instrument | `driver/instrument.py`, `sentinel/statecmp.py` | frame-locked sim-vs-emulator divergence: seed the sim from the live image, step both one frame at a time, report the first disagreement |

## Fixtures (not distributed)

The game is copyrighted and is **not** included. There is **one** supplied fixture: place
your own copy of the C64 tape image at `sentinel-gold.tap`.

`out/sentinel_stage2.bin` — the 64 KB memory image of the loaded game, used by the
`oracle`-marked tests that regenerate the goldens — is **not** a second supplied file: it is
**generated** from the tape. `driver/dump_stage2.py` boots the tape in asid-vice, dumps RAM,
and verifies the image by running the ROM's own generator against the model:

```bash
python -m driver.dump_stage2                    # --out PATH, --force to re-dump
```

Both files are gitignored; tests that need them auto-skip when absent. Generating the image
and running the live driver need Docker and the `anarkiwi/asid-vice:latest` image (build
from https://github.com/anarkiwi/asid-vice).

## Docs

- [gameplay.md](docs/gameplay.md) — the game's rules and mechanics, ROM-derived spec.
- [architecture.md](docs/architecture.md) — how the game works, a table of the ROM routines,
  how the model mirrors them, then the subsystems: landability filter, render-cost model,
  driver, instrument, measurement tooling.
- [players.md](docs/players.md) — the phase player and the reactive greedy player: the rules
  that decide a move, the phases, current results.
- [open_items.md](docs/open_items.md) — everything unsolved, and what was disproved.
