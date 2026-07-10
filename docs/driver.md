# The live driver (`driver/`)

The driver executes a plan against the **real** game running in
[VICE](https://vice-emu.sourceforge.io/) (asid-vice) inside Docker, with no
display: it boots the tape, enters a landscape, drives the sights by keyboard,
fires create/absorb/transfer/hyperspace, and verifies each result from the
game's own memory. It imports only `sentinel/` (for the bit-exact LOS/aim oracle
and the memory map).

## Package surface

| Module | Role |
|--------|------|
| `driver/core.py` | the canonical `SentinelDriver` + the container/boot/connect/navigate/record lifecycle (`boot_and_play`, `GameSession`, `validate_avi`), object-array reads, native aim snap (`snap_view`), live LOS probe (`probe_tile`). |
| `driver/kbd_aim.py` | keyboard aim geometry: the verified pan/cursor cycles, `snap_keyboard_view` (native grid search for a view landing on a tile), and `KbdDriver` (checkpoint-driven, U-turn-aware key driver). |
| `driver/sentinel_execute.py` | `Executor` (raw reads + decoded live state), `perform_step` (the aim → fire → verify plan-step primitive), `fire_hyperspace`, and `verify` (memory-delta arbiter). |
| `driver/boot.py` | robust tape boot with load-signature polling and the reusable VICE snapshot save/load (`boot.vsf`, code-entry `.vsf`). |
| `driver/sentinel_state.py` | read live 64 KB memory into a structured `GameState` (`ViceSource`/`Py65Source`), `verify_entry` against the standalone generator, `mem_image` for wrapping in a sim `State`. |
| `driver/watch_play.py` | passive logger of a **human** playing (connects to a stock binary monitor, polls state + records video); the source of the recorded human-win fixtures. Does not send keys. |

## Boot / enter / record lifecycle

`SentinelDriver.boot()` (via `driver.boot.boot_loaded`) launches an
`asid-vice:latest` container, connects a `BinMon`, and polls RAM for the loaded
game's signature (`A5 0B 85` at `$35A4`) instead of a fixed sleep, because the
multi-stage tape load's timing under warp varies and occasionally JAMs the 6502
(closing the socket) — so the whole container launch is retried. On first load
it caches a reusable boot snapshot (`renders/boot.vsf`) via the VICE monitor's
`MON_CMD_DUMP`/`MON_CMD_UNDUMP` opcodes so later runs can resume the title
screen instead of re-loading the ~50 s tape.

`enter_landscape(N)` / `navigate` drive the real title menu by keyboard — the
proven live path. The ROM's "SECRET ENTRY CODE?" gate (`$14DC`) computes its
jump-to-play from the code-validation result, so it cannot simply be bypassed;
instead three code-check sites (`$14DF`, `$2565`, `$2570`) are patched to accept
any code, and the menu is navigated as a player would (type the landscape
digits, a dummy code, dismiss the isometric preview, enter play). A landscape's
number is the last two typed digits read as one BCD byte, whose value **is** the
internal seed (e.g. `"0042"` → byte `$42` → seed 66); `landscape_from_digits`
parses the last two characters as hex, reproducing the ROM's BCD-to-binary step.
A code-entry-screen snapshot (`vice_code_entry.vsf`) is restored when present to
skip the tape boot; it is landscape-agnostic (digits are always typed after
restore).

`boot_and_play(...)` is the emulator-side glue the live runner should **not**
own: it handles container lifecycle + boot retries, the binmon connect, the
title-menu navigation with the cached snapshot, the in-play check, and AVI
start/stop, then hands a `GameSession` (the live `BinMon`, the entered
landscape, the entry-match result, the record start time, the AVI host path) to
a `play_fn(session)` that runs the actual plan loop. The AVI is **finalized in a
`finally`** even when the plan loop raises (an aim-exact crash, a mid-run
divergence) — the recording of the attempt up to the failure is the deliverable,
so it is stopped and flushed before the exception propagates, otherwise the
container teardown kills VICE mid-write and the AVI has no frame index.
`validate_avi` sanity-checks the result (RIFF/AVI header + at least one frame in
the `movi` list). Recording is skipped with `NO_RECORD=1` (which also leaves warp
on).

## The keyboard-aim → fire → memory-verify primitive

Every plan step is an aim onto a tile, a single action keystroke, and a memory
check that arbitrates the outcome.

**Choosing a view.** The tile an action fires on is not a pixel target but a
keyboard *view* — a bearing `h_angle` (an 8-unit lattice, reachable = `h0 + 8k`),
a pitch `v_angle` (a 4-unit lattice, clamped to the band `$CD..$35`), and the
sights cursor `(cx, cy)`. A single settled key press advances the cursor by 9 px
(the driver's `CX_GRID`/`CY_GRID` notch cycle, wrapping at the band edge with an
`h_angle`/`v_angle` carry), but the underlying ROM cursor-move routines
(`$9965`/`$9994`) step 1 px, and a scan-consumed press nudges the cursor a single
pixel — so the driver can in principle reach any cursor pixel (it searches a
coarse step-3 window for speed). The ROM's
`prepare_vector_from_player_sights` (`$1C10`) combines them: `h_eff = h + cx>>3`,
`v_eff = v + (cy-5)>>4`. `snap_keyboard_view` / `snap_view` search that grid with
`sentinel.los.aim_target` (bit-exact vs the ROM aim) for a `(h, v, cursor)` whose
native ray lands on the target tile with LOS — centred-cursor lattice first, then
a small bounded cursor window as a fallback (bounded so an unreachable tile costs
seconds, never minutes), preferring a low pan and a small tile-centre fraction.
The native aim snap is done with the **CPU halted** so enemies do not advance
while the driver thinks; no live-CPU LOS probe is used to choose a view (it would
wedge the ROM's incremental plotter).

**Driving the keys.** `KbdDriver` reaches the chosen view with all
gating/feedback from memory reads and checkpoint PCs, never wall-clock timing:

- Coarse rotation happens **sights off**, where the bearing is settled and
  stable. `coarse_h` picks the shorter of stepping ±8 or one U-turn (`EOR $80`,
  `handle_uturn $1B2F`) plus a short correction (`aimcost.h_press_count` chooses
  the minimal-keystroke plan); `coarse_v` pitches ±4. `_pan_angle` HOLDs the
  direction key and resumes once with a VICE condition-gated checkpoint at
  `PC_PAN_DONE $365D` (`@cpu:($addr) == $want`) that stops the CPU exactly when
  the angle register reads the wanted value — the scroll runs through every
  intermediate notch at full speed, no per-notch read-back. A stepwise fallback
  (`_pan_angle_stepwise`) halts one attempt at a time at `$365D` (reached once
  per attempt on commit, undo, and clamp, both axes) and stops on `stall_bail`
  consecutive zero-movement attempts.
- Fine cursor selection happens **sights on**; `fine_cursor` drives each axis one
  checkpoint-confirmed pixel at a time (`move_sights` STA PCs
  `$997C`/`$9990`/`$99B8`/`$99D2`), re-reading and re-choosing direction so a
  stray move self-corrects and never runs to the wrap band. A sights-on toggle
  re-centres the cursor (`$134C`), so the snapped cursor is driven explicitly.
- `tap_action` fires an action key **exactly once** via one idle full scan then a
  press-while-halted at the gated scan call site (`$9678`→`$967B`), so an action
  is at-most-once by the single-scan press and at-least-once by the verify-retry
  loop. It never re-fires on a false-negative latch (a second create/absorb would
  stack an extra object). `sights_set` toggles SPACE (`$0C5F` bit 7) one gated
  scan per toggle.

**Verifying.** `driver.sentinel_execute.perform_step` drives the view, confirms
the read-back angles reached the request (a mismatch = a pan clamp/no-converge,
so the sights point at the wrong tile → **do not fire**, return `aim_miss`),
fires the action key once, then `verify` arbitrates from the live memory delta:
the exact on-tile object-count change **and** the exact energy delta, flagging any
other global object-count change (wrong-tile landing, a meanie spawn, held-key
extra creates) as a divergence. Outcomes are `ok`, `best_effort_miss` (a
non-Sentinel fuel absorb missed — non-fatal), `drained` (energy already below a
create's cost before firing — no keys sent), `aim_miss` (the aim never reached
the view — nothing fired), `diverge` (the primary on-tile effect happened but a
secondary invariant moved under the aim — a live world-divergence, resync +
replan), or `fail`. A win is verified by the ROM's own landscape-complete flag
(`$0CDE` bit 6) after `fire_hyperspace` from the platform tile.

The `sentinel.los` probe (`probe_tile`) stays **advisory** — it reads the live
CPU asynchronously and can itself be churned — so the arbiter of a fired action
is always the ROM object-count/energy delta, never the probe.

### Reading the live sights bearing correctly

How the driver verifies that a keyboard aim landed on the requested view, and the
read-timing bug that made a correct aim look like a miss.

**Where the aim is stored.** The tile an action fires on is derived by
`prepare_vector_from_player_sights` (`$1C10`) from three live values:

- `objects_h_angle` — `$09C0 + player_slot` (bearing, moves on an 8-unit lattice)
- `objects_v_angle` — `$0140 + player_slot` (pitch, 4-unit lattice)
- the sights cursor — `$0CC6` / `$0CC7`

(See the re-sentinel disassembly `disasm/INPUT.md` §3–4.)

**Why a raw read of `objects_h_angle` is unreliable.** The foreground loop
(`$363D`) calls `JSR $10B7 pan_viewpoint` **every frame** at `$365A`.
`pan_viewpoint` does a settle dance — add `+$14`, `JSR plot_world` (`$2625`),
then a `−$0C` fix-up for a net `+8` — so **mid-frame the byte transiently holds
the un-fixed value** (e.g. a committed `$60` reads back as `$73`, which is off
the 8-unit lattice). The value is only the true, on-screen bearing at **`$365D`**,
the instruction right after the `JSR` returns — which is why `$365D` is the
reliable per-attempt checkpoint the pan primitive (`kbd_aim._pan_angle`) already
syncs to.

The churn is only observable while the **sights are ON** (the per-frame plot
dance is live); with sights **OFF** the bearing is settled and stable. It is also
aggravated by active enemies, because more redraw work widens the transient
window — but it is not caused by drain or by the player being moved.

The aim-vector scratch `$003D`/`$003E` (and `$0040`) is shared with the
enemy-relative-angle math, so those bytes churn under active enemies too; they
are not a stable source for the player's aim either.

**The bug and the fix.** An earlier live runner drove the coarse angles (which
land via the `$365D`-synced pan, sights-off and correct), then turned **sights
on**, then read `objects_h_angle` **asynchronously** and compared it to the
requested bearing. That read caught the sights-on pan-dance transient (`$73`) and
declared a **false aim miss**, refusing to fire a perfectly good aim. Its
timing-dependence is why the symptom appeared to come and go.

Fix: read the h/v angles **while the sights are still OFF** (immediately after
`coarse_h`/`coarse_v`, before `sights_on`), where `objects_h_angle` is settled.
The cursor is read sights-on (it is stable there). The `sentinel.los` ray probe
(`probe_tile`) stays advisory — it reads asynchronously and can itself be churned,
so the arbiter of a fired action remains the ROM object-count/energy delta
(`verify()`). This is the sights-off read that `perform_step` performs today.

## Container / bridge-IP plumbing and known gotchas

- **Bridge IP, not published ports.** Host `-p` port publishing is not reachable
  in this environment (`127.0.0.1:6502` is unreachable); the driver connects to
  the container's **docker bridge IP** (`docker inspect` of
  `NetworkSettings.Networks`). `BINMON_HOST` / `BINMON_PORT` env vars override.
  A missing bridge IP falls back to `127.0.0.1`.
- **`/scratch` bind-mount only.** The tape image and the `/renders` volume are
  bind-mounted into the container; the AVI, `boot.vsf`, and `vice_code_entry.vsf`
  land on the mounted `/renders` volume (gitignored) so they persist on the host.
  Snapshot save/load paths are **paths inside the emulator process**, so they must
  point at `/renders/...`.
- **Stale containers.** `free_stale_containers` / `kill_stale` remove any leftover
  `asid-vice:latest` container still holding port 6502 (a SIGKILLed driver can
  orphan a `--rm` container) before a new launch.
- **Monitor resilience.** A warp/AVI stall can drop the monitor socket mid-op;
  `reconnect` re-opens it and `robust` retries an op across a drop. Under live AVI
  recording (warp off) the ZMBV encoder can back-pressure the socket for seconds,
  so the pan/scan hang guards use generous env-overridable timeouts
  (`KBD_PAN_TIMEOUT`, `KBD_STA_TIMEOUT`) — they only cost time on a truly dead
  socket, never on the happy path.
- **Dwell spawns meanies.** At true gameplay speed (warp off during recording),
  any dead dwell in which the CPU runs — a clamped pan whose checkpoint PC never
  recurs, a post-action consumption wait — is live time in which the Sentinel can
  spawn a ring meanie. The checkpoint waits are therefore capped short and the
  caller resyncs + halts before the next action.
- **Full-image reads.** The simulator is defined over the full 64 KB image (enemy
  stepping reads ROM tables such as `ROTATION_SPEED_TABLE $9D37`), read in two
  32 KB halves because `mem_get`'s response length is a u16 (a single 64 KB request
  comes back empty). A resync only needs the first 4 KB, where all mutable
  play state lives.

## Plan steps

Each step `perform_step` consumes is `{verb, otype, target tile, view}`. A step
whose `view` is `None` is a **deferred aim** — an on-boulder synthoid that must
re-aim after the boulder just landed, or an absorb whose coarse sweep did not
resolve a view — re-proposed here against **current** live memory via the shared
aim proposer (`sentinel.aim.propose` at the player's true eye), so the sim and the
live driver never diverge on how a tile is aimed.

A **missed aim is treated as a crash** (`aim_miss`). A step is aim-exact, so a
miss means the model diverged from the real game and must be investigated, never
smoothed over with margin — the driver resyncs from true memory on a
world-divergence, but a genuine wrong-tile landing is the hard failure the live
contract raises on.

`KbdDriver.idle_until_rotation` idles the *inert* game — no keystrokes, so only
the enemies precess — frame-stepping via `run_until_pc($9630)` while reading
`OBJECTS_H_ANGLE[enemy]` until a threatening enemy's facing changes (its
`ROTATION_COOLDOWN` reloaded to 200) or a bounded cap trips, so a move can wait out
an enemy cone from readable cooldown bytes and the enemy's fixed rotation step only
(`ROTATION_SPEED_TABLE`, `ROTATION_COOLDOWN`, the 1-in-3 `$0C50` gate), no frame
model.
