# Open items

Everything known to be wrong or unproven, each stated as: what is wrong, what shows it, what
would resolve it. Anything not here is either correct or out of scope
([architecture.md](architecture.md#not-modelled-deliberate-scope)); a mechanism once
understood moves there and stops being an open item.

## 1. One object vertex angle is ten units out

**Wrong.** `rendercost` reproduces the ROM's own loop counts on every golden view but one:
on 2024,16,0 it runs `$2FA1` twice more than the ROM. The cause is not the fill: one object
vertex's 16-bit horizontal angle is 10 units low, which moves its `screen_x` byte across a
multiple of 32.

**Shown by.** Tracing every `$2EE4` of that view, the edges that disagree differ on their
input `$0039` alone. `$2D93` builds that byte as `(($A800,X + $0011) : ($0BA0,X + $0029)) >> 5`,
and the vertex's pair is `$FF80` on the machine against `$FF76` in the model; every other
vertex of those polygons is byte-identical. `$2D93` itself is exact against the real routine
on random inputs, and the ROM's `$0B40` accesses are self-consistent, so neither is the
mechanism.

**Mechanism.** `$0078` is stale in bit 0, not in the bits `$0E2C` reads. The two RORs put the
divide's round-9 and round-10 quotient bits in bits 6 and 7, so what `$0E2C BIT` reads is
never stale — but `$0DF1 ROR $78` rotates the *incoming* bit 0 out into the carry `$0DF3 ROL A`
consumes, and that can tip round 10's `$0DF6 CMP $76`, hence `$0E30 BVS`, hence the
interpolation, hence the angle. Within a `plot_world` pass `$0078` has no writer but `$0D4A`
itself, so its incoming bit 0 is the previous call's own output shifted twice.
`relative._divide_and_arctan` seeds it 0.

**Also not derived.**

- **`$3030`'s two overflow guards** are modelled as a loop rather than the ROM's re-entry into
  `$3022`; equivalent in every case checked, but not the same control flow.
- **`$27CE`, the observer's own tile.** `$2793` forces its four corners off the edges of the
  screen and plots it through `plot_checkerboard_tile`; the model charges the branch and the
  `$279B` examine (at `$001D + 1`, which `$2797 INC $0026` makes it) but not that polygon,
  because two of its corners inherit a screen_y low byte from whichever row last used the other
  `$0005` bank. No golden view reaches it, so it is unmeasured rather than known-small.
- **The object term without the game image.** `objectcost` needs the model geometry, so a
  checkout without `out/sentinel_stage2.bin` (or without numba) falls back to
  `projector._inview_object_base`'s floor and under-charges every object.

**Resolves.** Threading `$0078` through `_divide_and_arctan` from call to call — one byte of
state, started from the image, since `$0F9E`/`$1D9D`/`$1DC1` write it outside `plot_world` —
so the call whose round-10 bit the residue tips takes the machine's interpolation branch.

## 2. The driver's wall-clock timeouts are the residual load sensitivity

**Wrong.** `kbd_aim` still holds host clocks in an otherwise machine-clocked driver, and a
timeout there is retried rather than raised.

**Shown by.** On VICE a monitor round-trip at real-time pace is far slower than halted, so
host contention can push a wait past its timeout and the aim re-drives; the emulator times
the same waits in PAL cycles and is immune. What remains is the `_RU_COMMIT` socket backstop
and the swallowed timeouts in `_run_to_scan` and `tap_action`. A live run that diverges this
way stops at the first `live_audit` disagreement rather than playing on.

**Resolves.** `$365D` recurs every frame, so a timeout there means the game left the play
loop — raise, do not retry.

## 3. Landability filter: unproven corners

**Wrong.** Two soundness claims are handled but not proved.

**Shown by.** Alias landings (the 8-bit z compare wrapping, `landtable.WRAP_D`) are kept
unconditionally as candidates by `landtable._stop_one`, so the answer is sound either way,
but they are not proved unreachable: band rays do exceed the alias distance at a non-origin
cell. Both the partition and the march are keyed to `landtable.MAX_STEPS`, an explicit
argument, so a caller marching further must re-query with the same cap.

**Resolves.** A reachability proof for the alias distance on a real board.

## 4. Landscape 9795 is lost

**Wrong.** Of the eight hardest boards in `out/hardest_128.txt` (9795 9716 9950 9760 9831
9717 9329 9159) the wide search wins seven; on 9795 it climbs but spends its energy before
the Sentinel is in reach and stops with no surviving move.

**Shown by.** `python -m driver.emu_suite 9795 9716 9950 9760 9831 9717 9329 9159 --player
search --out-dir out/logs/emu_hard8`.

**Resolves.** Read 9795 move by move with `driver.decision_audit 9795`, and the rest of the
128 with `driver.emu_suite`.

## 5. Half the children are still forked

**Wrong.** Bound-first expansion forks about half of all children (`stats["children"]`
against `stats["bounded"]` in `driver.emu_suite` results). A child must be forked while its
bound reaches the beam's last kept key, and where they tie only frames decide, for which
there is no bound; how much each accounts for is unmeasured.

**Shown by.** The stats above; nothing yet splits the forked children by why.

**Resolves.** Count forked children whose bound ties the cut; if ties dominate, a proven
lower bound on a move's frames compared where bounds tie.

## Disproved — do not resurrect

Each is a hypothesis and the reason it fails.

- **Pruning forks by an entry-time sight table.** The table is right (on unchanged tiles
  0.19% of enemy-sight answers differ from entry's, `driver.visibility_audit`), but it does
  not separate good forks from bad: 80% of hop forks land where an enemy sees the body fully,
  the beam keeps 26% of those against 11% of unseen landings, and any rule pruning on sight
  prunes forks the beam kept (`driver.prune_audit`). A landing's danger is when a scan
  arrives, which the table cannot say.
- **A K-pruned landability table.** The first landing hit sits deep inside a candidate row, so
  a stored-first-`K` row decides almost no landing query at any affordable `K`, and a `K`
  large enough to decide the row would be terabytes. The closed-form walk needs no storage.
- **View selection only needs the `$F5` up/level pitch plane.** A pedestal is aimed at by its
  TILE, routinely *below* the eye even when the robot on it is not, so `moves.views` sweeps
  the band lattice too.
- **A far tile is an expensive aim.** Aim cost is angular, not spatial: it is bearing and
  pitch notches plus cursor drive, none of which grows with the tile's distance.
