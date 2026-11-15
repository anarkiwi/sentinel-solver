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

## 2. Landability filter: unproven corners

**Wrong.** Two soundness claims are handled but not proved.

**Shown by.** Alias landings (the 8-bit z compare wrapping, `landtable.WRAP_D`) are kept
unconditionally as candidates by `landtable._stop_one`, so the answer is sound either way,
but they are not proved unreachable: band rays do exceed the alias distance at a non-origin
cell. Both the partition and the march are keyed to `landtable.MAX_STEPS`, an explicit
argument, so a caller marching further must re-query with the same cap.

**Resolves.** A reachability proof for the alias distance on a real board.

## 3. The forked children are bounded by loose keys and frame ties

**Wrong.** Bound-first expansion still forks children it does not keep. Each batch now forks
only children that could still displace the `width`-th survivor (`search_player.may_displace`:
the bound beats its key, or ties it with no more frames than it, a move costing at least one
frame and a wait exactly `WAIT_QUANTUM`), so nothing the cut already excludes is forked.

**Shown by.** Against each step's final cut, the forks of eight moves of 335 and 9716 split:

| forked because | share | kept |
|---|---|---|
| bound above the cut | 43% | 39% |
| bound ties the cut, frames no more | 42% | 51% |
| the cut rose after the batch, or the beam was not yet full | 12% | 0 |
| no bound | 1% | all |

A tied child that loses has a slack (the cut's frames less its parent's) of a median 71% of its
move's frames, so a frame bound prunes it only within that of the true cost; the bounds above
the cut are loose in energy (a drain) and in `seen` (bounded 0).

**Resolves.** A move frame bound that close (the cycle-exact pan and render cost of its aim),
or bounds on a move's drain and end exposure; the rest is forking before the cut is known.

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
