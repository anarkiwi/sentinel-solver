***Claude - including Fable - was unable to solve this simple game from 1986 without my help. Is what it is.***


# The Sentinel — bit-exact model + live driver

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
| Model | `sentinel/` | standalone bit-exact forward model — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator) |
| Phase player | `sentinel/phase_player.py` | the default player: freedom first, then convert. Wins all eight measured boards |
| Reactive player | `sentinel/player.py` | tick-by-tick greedy player over the same `BasePlayer` |
| Landscape analyzer | `sentinel/landscan.py` | enemy count and terrain shape per landscape, to match one board to another |
| Driver | `driver/` | boot, enter a landscape, run memory-verified live keyboard operations (aim → fire → verify), record. Imports only `sentinel/` |
| Instrument | `driver/instrument.py`, `sentinel/statecmp.py` | frame-locked sim-vs-emulator divergence: seed the sim from the live image, step both one frame at a time, report the first disagreement |

## Fixtures (not distributed)

The game is copyrighted and is **not** included. Place your own copies at
`sentinel-gold.tap` (C64 tape image, used by the live driver) and
`out/sentinel_stage2.bin` (64 KB memory image of the loaded game, used only by the
`oracle`-marked tests that regenerate the goldens). Both are gitignored; tests that need
them auto-skip when absent. The live driver additionally needs Docker and the
`anarkiwi/asid-vice:latest` image (build from https://github.com/anarkiwi/asid-vice).

## Docs

- [gameplay.md](docs/gameplay.md) — the game's rules and mechanics, ROM-derived spec.
- [architecture.md](docs/architecture.md) — the model's modules, the landability filter and
  render-cost model, the driver, the instrument, and the measurement tooling.
- [players.md](docs/players.md) — the phase player and the reactive greedy player: the rules
  that decide a move, the phases, current results.
- [open_items.md](docs/open_items.md) — everything unsolved, and what was disproved.
