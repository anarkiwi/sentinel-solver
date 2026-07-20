# The live driver (`driver/`)

Executes a plan against the **real** game in [VICE](https://vice-emu.sourceforge.io/)
(asid-vice) inside Docker, headless: boots the tape, enters a landscape, drives the sights
by keyboard, fires create/absorb/transfer/hyperspace, and verifies each result from the
game's own memory. Imports only `sentinel/` (bit-exact LOS/aim oracle + memory map).

    python -m driver.play_player 42 --player astar   # A* wins landscape 42 live, records an AVI

Accuracy and ranked open problems: [plan_fidelity.md](plan_fidelity.md). Game rules:
[gameplay.md](gameplay.md).

## Package surface

| Module | Role |
|--------|------|
| `core.py` | container/boot/connect/navigate/record lifecycle (`boot_and_play`, `GameSession`, `validate_avi`), `SentinelDriver`, `live_image`, live LOS probe `probe_tile`. |
| `kbd_aim.py` | keyboard aim geometry: pan/cursor cycles, `KbdDriver` (checkpoint-driven, U-turn-aware key driver). |
| `sentinel_execute.py` | `Executor`, `perform_step` (aim → fire → verify), `fire_hyperspace`, `verify` (memory-delta arbiter). |
| `live_player.py` | `LiveMixin` (observation + execution over live memory, no decision logic) composed with the sim players into `LiveGreedy`/`LiveAStar`; `MeasuringKbdDriver` times each aim primitive. |
| `play_player.py` | runner: `boot_and_play` + a player + AVI validation → `out/play_player_<digits>.json`. |
| `clock.py` | machine-side clock: `frames` (exact wrap-free `$9630` checkpoint hit count) and `run_frames`. No host clock. |
| `boot.py` | tape boot with load-signature polling, bridge-IP lookup, container reaping, VICE snapshot save/load. |
| `sentinel_state.py` | live memory → structured `GameState` (`ViceSource`/`Py65Source`), `verify_entry`, `mem_image`. |
| `instrument.py` | frame-locked sim-vs-ROM divergence race (`python -m driver.instrument 42`). |
| `frozen_run.py` | RTS-stubs `update_enemies` ($16B5) live: isolates frame-cost fidelity from search/energy. |
| `plan_audit.py` | per-step plan-vs-live enemy-phase / dwell-window audit. |
| `replay_human.py` | replays a recorded human line, capturing per-step enemy phase into `<fixture>_truth.json`. |
| `watch_play.py` | passive logger of a **human** playing; sends no keys. |

## No wall-clock waits

Every wait keys on a PC or a memory predicate, never `time.sleep`: a host delay is
warp-dependent (warp on under `NO_RECORD=1`, off while recording), so it would make
measured frame counts differ between modes. `test_no_sleep.py` is an AST guard failing on
any unmarked `time.sleep` in the live-play modules; waits outside the emulated machine
(docker lifecycle, AVI muxer, tape-loader polling) carry an inline `# sleep-ok: <reason>`
pinned by that test. In play, `bm.auto_resume = False` — a read that resumes the CPU
advances the machine by host-timed frames, so the world moves only in deliberate
`run_frames`/checkpoint windows.

## Boot / enter / record lifecycle

`boot.boot_loaded` launches an `anarkiwi/asid-vice:latest` container, connects a `BinMon`,
and polls RAM for the load signature (`A5 0B 85` at `$35A4`); the tape load's timing under
warp varies and can JAM the 6502, so the whole launch is retried. A reusable boot snapshot
(`renders/boot.vsf`) is cached via the monitor's `MON_CMD_DUMP`/`MON_CMD_UNDUMP` opcodes
(`0x41`/`0x42`).

`navigate` drives the real title menu by keyboard. The "SECRET ENTRY CODE?" gate (`$14DC`)
computes its jump-to-play from the code-validation result, so it cannot be bypassed:
`$14DF`, `$2565`, `$2570` are patched to accept any code, then the menu is navigated as a
player would (landscape digits, dummy code, dismiss the isometric preview, enter play).
`landscape_from_digits` parses the typed 4-digit code as **hex** — the ROM stores it as a
packed-BCD word and seeds the PRNG from both bytes (`seed_prnd_from_landscape_number
$33ED` → `$0C7B`/`$0C7C`), and decimal digits equal hex nibbles, so `"0042"` → seed `0x42`
(66). `_enter_play` polls real predicates in emulated frames: generation has installed the
player when `$0C0A` is nonzero; play has started when the busy-plotting gate `$0CE4` bit 7
releases. The landscape-agnostic code-entry snapshot (`vice_code_entry.vsf`) is restored
when present to skip the tape boot (digits are typed after restore).

`boot_and_play` owns container lifecycle + boot retries, the binmon connect, navigation,
the in-play check and AVI start/stop, then hands a `GameSession` to `play_fn(session)`.
The AVI is finalized in a `finally` even when the plan loop raises — otherwise container
teardown kills VICE mid-write and the file has no frame index. `validate_avi` checks the
RIFF/AVI header + at least one frame in the `movi` list. `NO_RECORD=1` skips recording and
leaves warp on.

## The keyboard-aim → fire → memory-verify primitive

**Choosing a view.** A view is a bearing `h_angle` (8-unit lattice, `h0 + 8k`), a pitch
`v_angle` (4-unit lattice, band `$CD..$35`), and the sights cursor `(cx, cy)`;
`prepare_vector_from_player_sights` (`$1C10`) combines them as `h_eff = h + cx>>3`,
`v_eff = v + (cy-5)>>4`. A settled press moves the cursor 9 px but the ROM cursor
routines (`$9965`/`$9994`) step 1 px, so any pixel is reachable; the search uses a step-3
window. The shared proposer `sentinel.aim.propose` searches that grid with
`sentinel.los.aim_target` (bit-exact vs the ROM aim) for a `(h, v, cursor)` whose native
ray lands on the tile with LOS, preferring a low pan and a small tile-centre fraction,
with the **CPU halted** so enemies do not advance while the driver thinks.

**Driving the keys.** `KbdDriver` gates on memory reads and checkpoint PCs only:

- Coarse rotation runs **sights off**. `coarse_h` takes the shorter of ±8 steps or one
  U-turn (`EOR $80`, `handle_uturn $1B2F`) plus a correction (`aimcost.h_press_count`);
  `coarse_v` pitches ±4 over the linearised `$CD..$35` band. Both use `_pan_angle`: HOLD
  the direction key, frame-step at `PC_PAN_DONE $365D` (the instruction after the
  foreground loop's `JSR pan_viewpoint` at `$365A`, reached on every outcome — commit,
  undo, clamp — on both axes), read the settled angle each frame. It ends only on state:
  `ok`, `hyperspace` (player slot `$0B` changed mid-aim), or `unreachable` (no movement
  for longer than one notch scroll — an aim-proposer bug).
- Fine cursor selection runs **sights on**: `fine_cursor` drives each axis one
  checkpoint-confirmed pixel at a time (`move_sights` STAs
  `$997C`/`$9990`/`$99B8`/`$99D2`), pressing while halted and releasing after the store,
  since a held key auto-repeats in an accelerating burst (`$11F6 ASL $0CC8`). The
  sights-on toggle re-centres the cursor (`$134C`), so it is driven explicitly.
- `tap_action` fires a key **exactly once**: one idle full scan (re-arming `$0C51` at
  `$11EA`), then a press-while-halted across the gated scan call site `$9678`→`$967B`,
  want-flags read at `$0CE8..$0CEB`, released before the next scan — at-most-once by the
  single-scan press, at-least-once by the verify-retry loop. It never re-fires on a
  false-negative latch (a second create/absorb would stack an extra object). `sights_set`
  toggles SPACE (`$0C5F` bit 7), whose `$1236` edge latch needs that idle re-arm.
- `_run_to_scan` treats a timeout while `$0CE4` bit 7 is set as a redraw still running and
  re-arms; conceding there leaks the redraw's frames into the next primitive. A timeout
  with the gate open is a real stall.

**Reading the bearing.** Read h/v **sights-off** (right after `coarse_h`/`coarse_v`),
where `objects_h_angle` (`$09C0+slot`) and `objects_v_angle` (`$0140+slot`) are settled.
Sights-on, the foreground loop (`$363D`) calls `pan_viewpoint` (`$10B7`) every frame, and
its settle dance (`+$14`, `JSR plot_world $2625`, `−$0C` for a net `+8`) leaves the byte
transiently off-lattice mid-frame (a committed `$60` can read `$73`); active enemies widen
that window. The cursor (`$0CC6`/`$0CC7`) is stable sights-on and read there. The
aim-vector scratch `$003D`/`$003E`/`$0040` is shared with the enemy-relative-angle math
and is never a stable source for the player's aim.

**Verifying.** `perform_step` drives the view, confirms the read-back angles and cursor
reached the request (a mismatch means a clamp/no-converge, so the sights point elsewhere →
**do not fire**, `aim_miss`), fires once, then `verify` arbitrates the memory delta: the
exact on-tile object-count change **and** the exact energy delta, flagging any other
global object-count change (wrong-tile landing, meanie spawn, extra creates) as a
divergence. Outcomes: `ok`, `best_effort_miss` (non-Sentinel absorb missed, non-fatal),
`drained` (energy below the create's cost before or after aiming — no keys sent),
`aim_miss`, `aim_hyperspace`, `diverge` (primary effect landed, a secondary invariant
moved — resync + replan), `fail`. `classify_outcome` checks the primary effect **before**
the best-effort shortcut, so an absorb that landed but coincided with a Sentinel discharge
is a divergence to resync, not a miss to retry. A win is the ROM's landscape-complete flag
(`$0CDE` bit 6) after `fire_hyperspace` from the platform tile. `probe_tile` is
**advisory** — it reads the live CPU asynchronously and can itself be churned.

**Aim reuse.** A create/absorb leaves sights ON and the bearing untouched (SPACE at
`$11B3` is the only sights toggle), so when the committed bearing already equals the next
view's, only the cursor is driven, skipping the OFF→ON toggle's `initialise_sights`
(`$134C`) re-centre. The native-LOS probe gates that fast path; a slot change (transfer)
or a non-converged pan clears the committed bearing.

## Container / bridge-IP plumbing and known gotchas

- **Bridge IP, not published ports.** Host `-p` publishing is not reachable here
  (`127.0.0.1:6502` is unreachable); every boot path connects to the container's docker
  bridge IP, resolved by `boot.bridge_ip` (`docker inspect` of `NetworkSettings.Networks`)
  and used by `boot_loaded` and `boot_and_play` alike. `BINMON_HOST`/`BINMON_PORT`
  override; a missing bridge IP falls back to `127.0.0.1`. A headless boot is:

  ```python
  from driver import core
  drv = core.SentinelDriver.boot(record_mount="renders")  # bridge-IP connect + warp
  drv.enter_landscape(0x0335)                              # types "0335"
  # ... drive drv.bm ...
  drv.close()
  ```

  The container launches `warp=True`; `WarpMode` is not settable on this asid-vice build
  (opcode `0x52` → err `0x8f`), so a failed warp set is non-fatal.
- **Concurrent runs are safe.** The host publish is `-p 0:6502` (docker picks a free
  port), and teardown (`boot.kill_stale`) is scoped by
  `boot.stale_filter()` to containers named `asid-vice-<own pid>-*`; a blanket `ancestor=`
  sweep would remove another run's healthy container (`VICE_REAP_ORPHANS=1` opts in).
- **Bind-mounts.** Tape image and `/renders` are bind-mounted in; the AVI, `boot.vsf` and
  `vice_code_entry.vsf` land on `/renders` (gitignored). Snapshot paths are paths **inside
  the emulator process**, so they must be `/renders/...`.
- **Monitor service is frame-quantized while the CPU runs.** Measured per command: 0.04 ms
  halted, ~1.4 ms running under warp, ~23.5 ms at real-time pace (one PAL frame +
  overhead) — **independent of read size**; VICE services the binary monitor once per
  emulated frame, and recording correlates with slowness only because `video_record`
  forces warp off. So: read while halted, and treat a multi-second monitor timeout as a
  wait on a PC that can never recur, not as back-pressure.
- **Dwell spawns meanies.** With warp off, any dead dwell in which the CPU runs is live
  time in which the Sentinel can spawn a ring meanie; checkpoint waits are capped short
  and the caller resyncs + halts before the next action.
- **Full-image reads.** The simulator is defined over the full 64 KB image (enemy stepping
  reads ROM tables such as `ROTATION_SPEED_TABLE $9D37`), read in two 32 KB halves because
  `mem_get`'s response length is a u16 (a single 64 KB request comes back empty).

## Plan steps

A step is `{verb, otype, target tile, view}`, plus `min_energy` on a create (a post-aim
gate: a mid-aim drain must not push it below the reserve). A create/absorb step with
`view: None` is a **deferred aim** — an on-boulder synthoid re-aiming after the boulder
landed, or an absorb whose coarse sweep resolved no view — re-proposed against **current**
live memory via the shared proposer (`sentinel.aim.propose` at the player's true eye), so
sim and driver never diverge on how a tile is aimed. Transfer aims are driven by
`LiveMixin._drive_transfer_aim`, not `perform_step`. A missed aim is a crash: a step is
aim-exact, so a miss means the model diverged from the real game and must be investigated,
never smoothed over with margin.

`LiveMixin` keeps think time out of the live world: `_observe` snapshots memory and leaves
the CPU halted, `_advance` is a no-op, `_wait` spends real time frame-exactly via
`clock.run_frames`. `_plan_step_stale` re-validates the next planned step against the live
enemy phase **on the window the plan gated it with** — the body window for an absorb, the
target tile's gaze window for a build or transfer — and releases a margin-only block once
the step has already waited and the raw budget clears.
