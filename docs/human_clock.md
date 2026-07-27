# The recorded clock: measuring what an action cost

A `watch_play/3` fixture event carries the whole pre-action enemy clock — the `$1335`
Bresenham accumulator, the `$0C50` gate, and every enemy's `$0C28` rotation cooldown.
That is enough to recover, exactly, how many frames the real game advanced between two
recorded actions, with no cost model in the loop.

```bash
python -m pytest sentinel/tests/test_human_clock.py -q
```

## How the span is recovered

`$130C` adds `$CD` to the accumulator each frame and runs `$1317` only on the carry
(205/256 of frames); `$1317` decrements the cooldowns only on every third carry
(`$0C50`). So:

| quantity | what it pins |
|---|---|
| `$1335` accumulator | the frame count **mod 256** (`$CD` is invertible mod 256) |
| `$0C50` gate | lifts that to **mod 768** (205 carries per 256 frames, `205 % 3 == 1`) |
| `$0C28` sawtooth | picks the multiple — 200 rounds, reloaded at `$1813`, sticking at 1 |

An enemy whose rotation cooldown reloaded or sits on the stick value cannot vote on the
round count; a span is **exact** only when every remaining voter agrees. 91 of the 155
ls335 spans are exact. `human_clock.span_frames` is closed-form, and
`test_closed_form_matches_the_stepped_clock` checks it against the stepped loop over the
whole `(accumulator, gate)` space.

## What it grades

**The cooldown clock is right.** Seeded with a recorded clock and advanced by the
measured span, `enemies.advance_frames` lands on the next event's recorded `$1335`/`$0C50`
reading on **91 of 91** exact spans.

**The action cost model is not 6% long.** `span[i] = settle(verb[i-1]) + think[i] +
aim[i]`, and the human's think time sits on the measured side, so a correct bill should
land *under* the span. Over the exact spans between genuine player actions the bill is
**0.993x** measured — near parity, which means it overcharges by whatever the think time
was. The earlier "~6% long" figure came from the quantised `h_angle` proxy (an angle
only moves when a rotation fires, so it reads 0 or 749 and nothing between); the clock
supersedes it. On **17** spans the bill exceeds the *entire* elapsed time, which is proof
of overcharge regardless of think time.

**The enemy behaviour is not.** Over the same spans the facings follow on only **67 of
91**: an enemy the ROM holds in an earlier branch of `consider_enemy_state` (`$16E6`)
rotates in ours. Rotation is the last thing that routine reaches — after discharge, a
held target, a drainable robot, and a drainable boulder/tree have all declined — so a
facing that runs ahead means we exit a branch the ROM stays in.

Energy says the same thing, but only once the action is placed correctly. A bracket
fires *when the action lands*, so within span `i` the action is the LAST thing that
happens, not the first — seed the clock, advance the span, then apply it. Done that way
**83 of 91** exact-span actions reproduce the human's next energy. The misses are
all off by exactly one energy and go in *both* directions (both directions), so they
are drain-timing scatter inside the span, not a systematic over-credit.

A drain does not decrement a counter — `$1A08` DOWNGRADES its target (robot → boulder →
tree → gone). So an absorb whose object was drained mid-span yields one less, and whether
we agree turns entirely on placing the drain on the right side of the action. That makes
the residual an *ordering* question: where inside the span the drain falls.

Swapping our billed cost for the exactly measured span moves the per-action outcome by
one action (83/91 against the charged span's equivalent). The charge model is therefore **not** the lever for the
ls335 replay floor; the enemy branch is.

## Why `$0C30` is not a score (a retracted lead)

An earlier revision scored the enemy advance on each enemy's `update_cooldown` (`$0C30`),
because the recorded value sits on its stick value 1 in 78% of ls335 samples while ours
never does. Re-recording ls42 live killed that lead:

| capture | method | `$0C30 == 1` |
|---|---|---|
| ls335 | `watch_play`, polls a free-running machine | 77% |
| ls42 | `replay_human`, halts at a driver checkpoint | 8% |

Same register, same game, opposite readings — `$0C30` depends on **where in the loop the
capture stops**, so scoring on it measures the recorder, not the model. Our sim reads it
at a frame boundary and matches neither (290/506 async, 3/18 halted).
`test_update_cooldown_is_sampling_dependent_and_not_a_score` pins both sides so the trap
stays closed.

**Facings are the sound score.** A facing only moves when a rotation actually fires, so it
is a real state change no sampling position can manufacture.

The image is still worth having read: `$16ED LDA #$04 / STA $0C30,X` sits immediately
after the `$16E9` gate on every path, `$16D9 DEC $90 / BPL / LDA #$07` confirms the
8-slot cursor, and `$131C LDX #$17` the 24-byte cooldown sweep — all as modelled.

`advance_frames` takes a `plotting` flag that suppresses `update_enemies` while keeping
the cooldown clock — the ROM's behaviour whenever the foreground is inside `plot_world`,
the dither loop, or a scroll, none of which call `$16B5`. On facings alone, over the 91
exact spans: `plotting=False` 67/91, `plotting=True` 55/91. Neither extreme is right,
because a span is *part* plotting (the settle's dither and replot, the aim's scroll
notches, each already counted term by term in `actioncost`) and part idle main loop.

## The phase split

`BasePlayer._aim_phases` now returns the aim as ordered `(frames, plotting)` segments and
`_settle` is advanced as plotting, so an action evolves the world the way the ROM does:

| segment | ROM | runs `$16B5`? |
|---|---|---|
| sights toggle | `$134C` re-centre + `plot_sights` | no |
| u-turn taps | action tap `$23`, no scroll, no replot | yes |
| pan notches | `$10EE`/`$1135` scroll + `plot_world` per notch | no |
| cursor drive + firing tap | gated `move_sights`/`tap_action` scans | yes |
| settle | `$1FA4`/`$86A5` dither + replot, or `$357D` redraw | no |

Scored on facings over the 117 exact spans the split gives **90/117** against
idle-only's 89 and plotting-only's 64. On the larger sample that margin is thin — one
span — so the split rests mainly on the replay floors and action counts below, not on
facings. `UPDATES_PER_FRAME` is not a remaining lever: 2, 3, 4 and 8 all score within a
few of each other, because an enemy's own `$16E9` gate rate-limits it harder than the
`$90` cursor does.

Search and executor share one sequence (`_aim_head_tail` + `advance_phases`), so a plan
is priced against the world evolution `_fire` will actually produce. Letting them drift
costs real quality — with only the executor split, ls0 went from 23 actions to 25; with
both sharing the sequence it is 23 again.

Replay floors moved **ls42 8 → 14** and **ls335 15 → 19**, and offline ls110 went
**48 → 37 actions** (ls42 32 → 34).

## All three boards, live

`replay_human` now records `$1335`/`$0C50` alongside the per-enemy cooldowns, so any
re-recorded line is measurable the way ls335 is. Re-recording all three
(`ls*_clock.json`, derived state with no `mem` — the same non-copyrighted class as
`ls42_truth.json`) settles what one board could not:

| board | enemies | capture | exact spans | clock round-trip | facings |
|---|---|---|---|---|---|
| ls0 | 1 | live | 16 | 16/16 | **16/16** |
| ls42 | 2 | live | 10 | 10/10 | **10/10** |
| ls335 | 7 | live | 18 | 18/18 | 12/18 |
| ls335 | 7 | async | 117 | 117/117 | 89/117 |

The cooldown clock round-trips perfectly everywhere. The **facing** gap is ls335's alone,
and it survives re-recording by the checkpoint method, so it is a real defect of the model
on a seven-enemy board — not an artifact of the async recorder that produced its fixture.
That is the open question, now well posed: what does a seven-enemy board do that a one-
or two-enemy board does not.

Recording ls0 also paid for itself indirectly. `rounds_between` demanded two agreeing
enemies, which a single-enemy board can never supply, so ls0 silently measured nothing.
One voter suffices — `span_frames` must satisfy (bres, gate) AND the decrement count
jointly, so a wrong delta yields no candidate rather than a wrong one. Dropping the guard
took ls335 from 91 exact spans to **117** and unlocked ls0 entirely, with the round-trip
still 100%.

## Fixture hygiene

The dither loop (`$1FA4`, ~50 f) and the transfer tune wait (`$35D5`, 96 f) are hard
floors on how close two real player actions can be. **8** exact ls335 spans fall below
the floor for the action preceding them, so those bracket pairs are one action recorded
twice — a recorder artifact class beyond the two `_is_player_action` already drops
(enemy discharge trees, drain ticks minted as self-transfers). ls335 also carries 33
events of those two known classes; `human_replay` skips them.
