# The phase player: freedom first, then convert

`sentinel/freeplayer.py`. A second planner, written from the game's own rules rather
than from a cost model, because the weighted search
([stance_planner.md](stance_planner.md)) does not convert ls335 and its failures were
never in the search — they were in what the search was asked to optimise.

It wins **ls335 from entry** in 73 actions — the board this repo is named for losing —
and seven of the eight boards measured. See [Where it stands](#where-it-stands).

## The three facts it is built on

**Time is free; only exposure costs.** An idle frame costs nothing unless a cone is on
the body, so a body nobody can see may wait as long as it likes. The corollary is the
rule the first version got wrong: waiting is the one move that cannot help while you
ARE being drained.

**A gap does not need forecasting.** `enemies.step` is byte-exact, so instead of
predicting when a cone arrives (`_cone_onset`, and the errors that cost this repo a
documented open item), clone the state, run it forward over the action's real duration,
and look. `_drained_over` is four lines and is never wrong.

**Height buys freedom, and nothing else does.** Measured over the whole stance graph on
ls335:

| eye | tiles it can land | stances it can climb to | enemies it can strike |
|---|---|---|---|
| 4.0 | 6 | 18 | 0 |
| 7.0 | 8 | 6 | 0 |
| 8.5 | 62 | 146 | 0 |
| 9.5 | 107 | 140 | 2.91 |
| 11.5 | 153 | 13 | 4.78 |

**Nothing on the board can strike any enemy below eye 8.875**, and the body starts at
3.875. So the opening cannot be scored by progress toward a target: for the first five
units of climb there is no reachable target to progress toward. Exposure rises with
height too (`hot` 2.75 -> 4.9), so the climb is not into safety, it is into options.

## The phases

| phase | goal | ends when |
|---|---|---|
| **1a establish** | energy in hand, in fuel-rich ground | the best climb on offer is affordable |
| **1b breakout** | height, per unit of a finite purse | an enemy is landable |
| **2 convert** | absorb, Sentinel last, then the platform | hyperspace from the platform |

Splitting 1 from 2 took ls335 from unwinnable to won. Splitting 1a from 1b then halved
the opening, 107 actions to 55: the two want opposite things from the same tile — 1a wants
to stay where the trees are, 1b wants to leave for height — and a single scorer
blending them strands the body at eye 7.875 with 56 climbs available and one absorbable
object in reach.

## The rules that decide a move

Each is a fact about the ROM, not a tuned quantity.

* **Never idle under fire** — a cone on the body makes every waiting quantum a drain.
* **A hop's danger is at the destination**, and the probe must cover the *whole* hop
  including aims. Checking 420 frames instead of ~1500 is why it kept landing in live
  cones at E=1.
* **Never land on zero** — $1A00 kills on a drain arriving at zero energy.
* **Do not hop where you cannot eat** — a landing with no fuel under it is a dead end,
  whatever it gains in height.
* **The Sentinel is the point of no return** — $1B91 reads `objects_flags[0]`
  absolutely, so once its slot empties EVERY later absorb is refused. The robot and the
  3-energy hyperspace toll ($216A, which kills on underflow at $2170) must be banked
  before the strike, not gathered after it.
* **The endgame is exclusive** — with the Sentinel gone the platform is an eye gain, so
  an unguarded climb will spend the hyperspace toll on height.

## Where it stands

There is no node budget and no wall-clock cutoff: the planner is deterministic, so a
run's outcome does not depend on host load and the times below are observation only.

| board | enemies | result | actions | eye | wall |
|---|---|---|---|---|---|
| ls0 | 1 | won | 16 | 6.875 | 12 s |
| ls42 | 2 | won | 32 | 8.875 | 38 s |
| ls110 | 3 | **lost** | 17 | 7.375 | 20 s |
| ls60 | 7 | won | 56 | 7.875 | 71 s |
| ls298 | 7 | won | 34 | 8.875 | 43 s |
| ls321 | 7 | won | 85 | 7.875 | 107 s |
| ls373 | 7 | won | 65 | 7.875 | 150 s |
| ls335 | 7 | won | 73 | 8.875 | 80 s |

**7 of 8.** ls335 and ls373 converted when the four invented `BUILD_FRAMES` durations
were replaced by each action's real aim-plus-settle (`_span`). That is worth stating
plainly: the planner was not mis-reasoning about those boards, it was asking for gaps of
the wrong length, and the strategy was sound before the arithmetic was.

## What is not solved

**ls110 walks into a local trap and no global rule fixes it.** At 17 actions it stands
at eye 7.375 with 3 energy under a cone, and is drained to death over the next 765
frames. The deadlock is mechanical, and was measured rather than inferred:

* the only climb on offer costs exactly 3, and paying it lands the body on zero, which
  $1A00 kills;
* every remaining absorbable object is out of view, so `_refuel` cannot raise the purse;
* no hop is available, so the under-fire fallback finds nothing.

The mistake is therefore several actions upstream, and the fork arbiter cannot see it:
forks play `rollout=True` — a fixed policy order — so a fork's future is not the future
the top level will actually have. Deepening lookahead from 4 to 8 to 16 changes nothing
on ls110 (measured), which rules out depth as the cause and leaves policy mismatch.

Four rules were tried against this trap and none is committed — each either traded
boards or could not see it:

| rule tried | result |
|---|---|
| absorb only if its value exceeds the drain over its own span | ls110 won, ls321 + ls373 lost — 6/8 |
| under fire, let escape pre-empt harvesting | no change anywhere: at the stall there is no escape to pre-empt |
| make `_supply`'s affordability test agree with `_best_climb`'s | ls42 + ls110 lost — 6/8 |
| rank a rollout that ends stuck below one that does not | no change: four ticks out the body is not yet stuck by any local measure |

The first of those also **disproves an earlier claim in this document**, that `_refuel`
harvests "at a loss" under a cone. It cannot: idling through the same span costs the
same drain and yields nothing, so an absorb under fire is never worse than waiting.
What ls110 loses to is not the absorb's price, it is the *time* the absorb occupies.

**So the next step is to make forks play the policy they are ranking.** Give
`_arbitrate` a rollout that runs the real `_tick` — arbiter included — rather than the
fixed ladder. That is what makes the score mean what it claims. It is blocked on cost,
and the cost is one function: **97% of runtime is `los._landable_batch`**, the
pitch-band sweep, at ~0.4 s a call. Making that cheap is the enabling work, and it is
engineering rather than strategy.
