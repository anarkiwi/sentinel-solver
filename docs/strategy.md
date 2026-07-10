# Strategy

How a landscape is won. These are consequences of the rules in
[gameplay.md](gameplay.md), not new mechanics — the principles a correct planner
must follow. Enumerated so we can extend the list.

1. **Exposure is a timed resource race, not a hard veto.** Because time is
   continuous, once an enemy has line of sight the player is drained ~1 energy per
   drain-cooldown for as long as that sightline holds, and every action costs time
   during which the drain keeps running. Survival is a race between banked energy
   and the drain over the time the planned actions take: a productive sequence —
   create a boulder, create a synthoid, transfer up — *can* be completed while
   being drained, provided the player has enough energy banked to pay the drain
   across its duration. What does **not** work is trying to *recover* energy by
   absorbing low-value objects while drained: a tree gives +1, but its aim+absorb
   spans several drain periods, so chasing trees to catch up nets negative — the
   absorb rate cannot exceed a continuous drain when each absorb spends multiple
   drain intervals. You only come out ahead by absorbing *faster than you are
   drained*, in practice by absorbing the **drainer itself**: absorbing the
   Sentinel (or a sentry) removes the source and ends the drain as it lands, so an
   endgame absorb can be driven straight through a drain even from low energy. The
   other escape is to break line of sight (transfer to an unseen robot, or
   hyperspace). So budget exposure as a timed energy cost over the action window —
   spendable against a sufficient buffer, or against a move that ends the drain —
   but never treat a seen tile as free, and never fund a plan on tree-refueling
   done while being drained.

2. **Only line of sight matters — there is no "danger ring."** A tile's safety is
   determined solely by whether the Sentinel (at any facing it can rotate to) has
   line of sight to it, never by how many tiles away the plinth is. A tile can be
   far from the plinth yet in full view (deadly), or immediately adjacent to it yet
   occluded / behind the Sentinel's back (safe). Standing on a boulder right next
   to the Sentinel while it is not looking at you is perfectly fine — indeed a
   boulder adjacent to the plinth, built and occupied while the Sentinel faces
   away, is a valid *penultimate winning position*: from there you absorb the
   Sentinel and take its platform. Gate builds and footholds on an actual
   line-of-sight/gaze query to the specific tile (`sentinel/threat.py`), not on
   Chebyshev distance to the platform.

3. **Prefer high, terrain-covered edge and corner tiles.** Centrality is a
   non-signal: a single rotating enemy points at any given tile for the same ~8%
   of its rotation (its FOV cone ÷ 256) wherever that tile sits, so being "in the
   middle" does not make you more seen — only occlusion does. What genuinely
   favours an edge or corner tile is geometry:
   - **Cover is easier to hold.** Safety is "the enemy's one bearing to me is
     blocked at every facing it can reach." At a corner the map boundary is at your
     back and all terrain, occluders and enemies lie in a ≤90° quadrant in front of
     you, so a single occluding rise can make the tile permanently safe; in the
     open centre the threat bearing can come from any of 360°.
   - **Your climb stack stays hidden.** The boulders/synthoids of a climb are
     exposed objects too (seen → downgraded → meanie spawn). A corner lets the
     stack grow in the Sentinel's occlusion shadow instead of out in the open —
     this, not a distance "ring," is the real reason not to build in the Sentinel's
     face.
   - **Longest reach.** From a corner the board extends up to its full ~45-tile
     diagonal in front of you, giving the longest unobstructed sightlines — best
     for surveying, reaching a distant platform, and firing the long-range endgame
     absorb from afar. From the centre the maximum reach in any direction is only
     ~16 tiles.
   - **Fewer enemies bear on you.** On multi-sentry maps a corner tile often sits
     outside some sentries' arc entirely.

   The catch: edge tiles are often **low**, and low tiles have short,
   easily-blocked sightlines — useless for the win unless you can climb high there.
   And the farther the launch tile is from the platform, the thinner and more
   occludable the sightline to it, so a corner launch only pays off when that long
   diagonal to the plinth is genuinely clear. Net: the ideal launch tile is a
   **high, terrain-covered corner with a clear diagonal to the plinth**, scored by
   LOS and occlusion — the opposite of what a centrality or proximity heuristic
   would pick.

4. **Place and time builds against the enemy's predictable gaze precession.**
   Enemy rotation is deterministic: an idle enemy adds a fixed per-enemy step to
   its facing every ~200 ticks (`rotate_enemy` $1805, `ROTATION_SPEED_TABLE`) and
   scans a ±10-unit cone. A fixed step added mod-256 does not oscillate — it
   *precesses*, walking the compass in a knowable, periodic sequence, so for any
   tile you can compute exactly when the gaze will next fall within the cone of its
   bearing and how long the gap is until then. This converts "exposed at some
   facing" into "safe for a schedulable window": permanently-occluded tiles are
   ideal but rare, and anticipating the precession lets you safely use the many
   high footholds that are only *briefly* visible, during the long intervals the
   gaze is elsewhere. It is the lever that keeps the candidate set from collapsing
   near good high ground. Combined with strategy 1, a dwell-and-build is safe iff
   the whole action sequence — priced in ticks (`sentinel/actioncost.py`) — fits
   inside the gaze gap and is started just after the gaze sweeps past; since time
   is continuous with no mid-action pause, a build that overruns the gap is caught.
   So optimize placement to tiles whose safe window ≥ the planned build duration,
   and phase the build into that window. With multiple sentries the safe interval
   is the *intersection* of every enemy's gaze gaps, so anticipate each precession
   to find a tile-and-time where all gazes point away at once. Two schedule levers:
   an enemy locked onto a drainable target stops precessing (it rotates only while
   idle), so a decoy elsewhere can hold a gaze away — and, conversely, a careless
   exposed build can hijack a gaze onto you. Because all of this is deterministic,
   the planner must forecast the gaze schedule (`enemies.step` /
   `threat.ticks_until_seen`) rather than veto every ever-visible tile with a
   static mask.

5. **Gain height early, and batch boulders to gain it in fewer transfers.** Height
   is the win resource: your eye height sets how far you can see and absorb, you
   cannot look up (a tile above your eye is unseeable), and the endgame is a
   long-range line-of-sight shot fired from launch height — so every unit of eye
   height unlocks more of the map (fuel, covered high ground, the platform itself).
   Given continuous time, *when* and *how* you climb matters:
   - **Transfers dominate the time and the risk.** A transfer is the most
     expensive action (hyperspace-tune wait + full redraw, ~300 ticks) and is also
     the moment your position changes — a fresh landing that must be safe. So
     maximize *height gained per transfer*: the transfer count sets both the
     cumulative drain time and the number of exposed landings you must schedule.
   - **Use more than one boulder at once.** A boulder raises a tile half a unit and
     builds are capped ~2 units above your eye, so you can stack ~4 boulders on a
     target tile in a single dwell, cap it with a synthoid, and transfer once to
     gain the full ~2 units — instead of build-one/transfer/build-one, which pays a
     full transfer for every half unit. Batching amortizes the fixed per-transfer
     cost and concentrates the whole climb into one safe tile and one gaze gap
     (strategy 4) rather than four landings and four windows. The batch size is
     bounded by the build slack and by how much build time fits inside the gaze
     gap — stack as many boulders as the safe window affords, phased into it.
   - **Do it early.** The opening is often the cheapest exposure window (enemies
     not yet precessed/locked onto you, energy buffer still full to fund the
     climb), and height compounds: the sooner you see far, the sooner you can spot
     fuel, covered high ground and the endgame launch tile and plan the rest of the
     route — whereas staying low keeps you blind, reactive and forced into whatever
     few (often exposed) tiles a short sightline reaches, bleeding drain across many
     small moves. Early, batched height converts to a shorter, safer, better-
     informed remainder.

## Human-win refinements (observed on ls0)

A recorded human win on landscape 0 (never seen, zero drain) refines the
principles above with concrete, exploitable structure. These are strategy — how
the game is won — distinct from the solver's status in reproducing them, which
lives in [planner.md](planner.md#6-outstanding-issues).

6. **A clean win takes zero drain.** Every energy change in the human run is
   build/absorb accounting; the Sentinel never had line of sight to the player,
   even while it stood **cheb-3 from the plinth**. Safety is the deterministic
   **gaze forecast** (`threat.ticks_until_seen`, strategy 4), not a static
   "could ever be seen" mask (`threat.is_exposed`) — that mask flags nearly every
   useful high tile. Real exposure = *the gaze catches you during your action
   window*.

7. **Aim-coherent ping-pong.** Consecutive aim bearings flip by ≈180° each move
   (measured Δ ≈ 128 units), so every aim is a single U-turn keystroke, never an
   arbitrary sweep. The XY looks scattered; the *bearing sequence* is maximally
   tight. Cheap actions and gaze-avoidance are the **same lever** — a cheap
   (short-aim) action finishes inside a gaze gap; an expensive one (long pan, tall
   build) overruns the gap and gets caught. **After a transfer the view already
   faces back toward the origin tile** (the ROM leaves the sights pointed at where
   you came from), so the look-back reabsorb of the departed shell costs *zero*
   extra aiming. This is why alternating sides of the map is cheap: the next build
   target sits ≈180° from that look-back heading, so both the reabsorb and the next
   build are reached by a single U-turn from the settled post-transfer view. A
   candidate ordering that prefers the far, ≈180°-opposite tile on rising terrain
   is therefore both the cheapest-to-aim *and* the highest-climbing move.

8. **Height first, cheaply — leapfrog rising terrain.** Height comes from
   **transferring to naturally-high terrain** (landing-tile terrain 5→6→7→8, +1
   each) with *minimal* builds, not tall boulder towers on low ground. Build and
   absorb at a **distance** (shallow aim, cheap), never adjacent (steep down-look,
   expensive): ≤2 boulders, far transfer, shallow look-back reabsorb. The trail
   self-funds — the shell you leave behind is reabsorbed from the next tile.

9. **Fuel is deferred by visibility.** Don't grab low-value fuel early. Fuel that
   is visible from many places is deferred (get it later from height); fuel that is
   hard to see from elsewhere, or needed to escape a terrain pocket on a steep
   landscape, is taken opportunistically now.

10. **Launch from afar, looking down.** The win is fired from a far high tile
    looking *down* onto the platform (strategy 3's longest reach). The endgame/launch
    LOS is asymmetric: the player looking *down* is allowed where the platform-vantage
    looking *up* is blocked (`$1D2E`), so far high launch tiles are valid even when a
    naive symmetric query under-counts them.
