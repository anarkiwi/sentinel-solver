# Driver aim: reading the live sights bearing correctly

How the live driver verifies that a keyboard aim landed on the requested view,
and the read-timing bug that made a correct aim look like a miss.

## Where the aim is stored

The tile an action fires on is derived by `prepare_vector_from_player_sights`
(`$1C10`) from three live values:

- `objects_h_angle` — `$09C0 + player_slot` (bearing, moves on an 8-unit lattice)
- `objects_v_angle` — `$0140 + player_slot` (pitch, 4-unit lattice)
- the sights cursor — `$0CC6` / `$0CC7`

(See the re-sentinel disassembly `disasm/INPUT.md` §3–4.)

## Why a raw read of `objects_h_angle` is unreliable

The foreground loop (`$363D`) calls `JSR $10B7 pan_viewpoint` **every frame** at
`$365A`. `pan_viewpoint` does a settle dance — add `+$14`, `JSR plot_world`
(`$2625`), then a `−$0C` fix-up for a net `+8` — so **mid-frame the byte
transiently holds the un-fixed value** (e.g. a committed `$60` reads back as
`$73`, which is off the 8-unit lattice). The value is only the true, on-screen
bearing at **`$365D`**, the instruction right after the `JSR` returns — which is
why `$365D` is the reliable per-attempt checkpoint the pan primitive
(`kbd_aim._pan_angle`) already syncs to.

The churn is only observable while the **sights are ON** (the per-frame plot
dance is live); with sights **OFF** the bearing is settled and stable. It is also
aggravated by active enemies, because more redraw work widens the transient
window — but it is not caused by drain or by the player being moved.

The aim-vector scratch `$003D`/`$003E` (and `$0040`) is shared with the
enemy-relative-angle math, so those bytes churn under active enemies too; they
are not a stable source for the player's aim either.

## The bug and the fix

`scripts/run_plan_live.py::perform_step` drove the coarse angles
(`coarse_h`/`coarse_v`, which land via the `$365D`-synced pan, sights-off and
correct), then turned **sights on**, then read `objects_h_angle` **asynchronously**
and compared it to the requested bearing. That read caught the sights-on pan-dance
transient (`$73`) and declared a **false aim miss**, refusing to fire a perfectly
good aim. Its timing-dependence is why the symptom appeared to come and go.

Fix: read the h/v angles **while the sights are still OFF** (immediately after
`coarse_h`/`coarse_v`, before `sights_on`), where `objects_h_angle` is settled.
The cursor is read sights-on (it is stable there). The `sentinel.los` ray probe
(`probe_tile`) stays advisory — it reads asynchronously and can itself be churned,
so the arbiter of a fired action remains the ROM object-count/energy delta
(`verify()`).
