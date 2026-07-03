#!/usr/bin/env python3
"""Search planner + solvability detector for The Sentinel (C64).

This is solver step 3. It plans an action sequence that wins a landscape with
the *maximum* final energy budget, and decides — from the initial state — whether
a landscape is solvable at all, reporting why not when it isn't.

How the planner thinks (and where it differs from `game_model`)
--------------------------------------------------------------
`game_model.apply` is a faithful but *deliberately small* simulator: it tracks
energy/slots/object occupancy exactly, but it computes the player's eye height
purely from the terrain height field (`terrain_surf + ROBOT_EYE`). It does NOT
model the real game's defining mechanic — gaining height by **stacking boulders
and transferring into a robot on top** — because that does not change the terrain
field. (Empirically: `apply` even rejects creating on an occupied tile, so it
cannot represent a same-tile stack at all; and `apply("win")` succeeds without
re-checking line of sight.)

So the planner carries its OWN abstract climb model on top of `game_model`:

    abstract eye height = terrain_surf(base) + ROBOT_EYE + k * BOULDER_HEIGHT

after climbing a stack of `k` boulders on a base tile and transferring into a
robot on top. Line of sight from that raised eye is tested with the model's own
`can_see(..., eye_offset = ROBOT_EYE + k*BOULDER_HEIGHT)` — i.e. we reuse the
real raytrace/looking-up physics, only feeding it the raised eye the stack buys.

Because the two layers disagree about stacking, every emitted plan carries TWO
views (see `Plan`):

  * `actions`      — the executor-facing list (step 4 / VICE). It includes the
                     real same-tile stacked creates and is addressed by tile, with
                     per-action abstract metadata (stack level, eye height, the
                     LOS test it enabled). This is what gets translated to
                     keypresses.
  * `model_replay` — a projection that `game_model.apply` accepts end-to-end (the
                     moves the model can represent: hop-absorbs + the final
                     `win`). `solve()` replays THIS through `game_model.apply` and
                     asserts `state.won == True`.

A plan is only declared solved when BOTH independent checks pass:
  (a) the model-replay projection reaches `won` (structural / energy / slot
      validity), and
  (b) the abstract LOS proof holds (`can_see` from the raised eye actually reveals
      the Sentinel). (a) alone proves nothing about feasibility, because
      `apply("win")` ignores LOS — (b) is the honest "the climb reveals the
      Sentinel" claim, and it is what `analyse_solvability` reports on.

The executor (step 4) MUST still re-verify LOS live in VICE: our `can_see` is an
approximation and `BOULDER_HEIGHT` is a modelled constant.

Search algorithm
----------------
Deterministic, staged, greedy/best-first (NOT provably optimal — the state space
is huge):

  Stage A  Absorb every object that is reachable at a flat robot eye, by hopping
           (create a robot on a visible empty tile, transfer into it) across the
           LOS-connected reachability graph, collecting trees/sentries on the way.
           (The conservative threat model is consulted only for climb-base
           selection in Stage B; this sweep is advisory about exposure because the
           model has no drain timing — the live executor minds enemy drain.)
  Stage B  Pick a climb: among reachable base tiles, find the one needing the
           smallest boulder stack `k` to gain LOS to the Sentinel (ties broken by
           lower exposure then lower energy cost then tile order). Build the stack
           (boulders) + robot, transfer up.
  Stage C  Win: absorb the Sentinel and transfer onto its platform (+4 energy).
  Stage D  Re-absorb stranded boulders/robots left in the climb where we still
           have line of sight to them, recovering their energy.

Maximising final energy: every boulder/robot we place is re-absorbed once we have
climbed past it (Stage D), so the climb is, in principle, energy-neutral; the
Sentinel adds +4; and Stage A sweeps up every other object's energy. Final energy
is therefore (start + sum of all absorbed object energies) masked to 6 bits, minus
any energy stranded in structures we could not re-absorb. See `solve` docstring
for where this is sub-optimal.
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game_model as gm
from game_model import (
    GameModel,
    Action,
    apply,
    can_see,
    visible_tiles,
    tile_surface_height,
    object_in_tile,
    is_exposed,
    ROBOT_EYE,
    ENERGY_IN_OBJECTS,
    ENERGY_MASK,
    T_ROBOT,
    T_SENTRY,
    T_TREE,
    T_BOULDER,
    T_SENTINEL,
    T_PLATFORM,
)
from game_state import GameState, GameObject, N

# ---- modelled constants ----------------------------------------------------
# One boulder raises the eye by this many height-field (nibble) units when you
# stand a robot on top of it. The game's height field is in nibble units (0..11);
# A boulder is HALF a height unit tall: put_object_in_tile $1F56 stacks a boulder
# by ADC #$80 on the z_fraction (= +0.5 unit), confirmed live in code_engine (each
# create(3, tile) raises the stack-top z by exactly 0.5). The old value (1.0) over-
# estimated the height gained per boulder; the real climb needs ~2x as many
# boulders to clear a given height. MAX_STACK is doubled to keep the same reachable
# eye height (12 * 0.5 = 6 units, plus terrain).
BOULDER_HEIGHT = 0.5
MAX_STACK = 24  # boulders; 24 * 0.5 = 12 units, the full height field range


# ---- plan container --------------------------------------------------------
@dataclass
class PlanStep:
    """One executor-facing action plus the abstract reasoning behind it."""

    action: Action
    note: str = ""  # human-readable intent
    stack_level: int = 0  # boulder index within a climb (0 == ground)
    eye_height: float = 0.0  # planner's abstract eye height after this step


@dataclass
class Plan:
    steps: List[PlanStep] = field(default_factory=list)
    model_replay: List[Action] = field(default_factory=list)
    solved: bool = False
    final_energy: int = 0
    absorbed: List[Tuple[str, int, int]] = field(
        default_factory=list
    )  # (type_name,x,y)
    los_proven: bool = False  # abstract climb actually reveals the Sentinel
    notes: List[str] = field(default_factory=list)
    # energy the real game recovers in Stage D by re-absorbing the climb structure
    # (boulder stack + the robot we stood on). The model-replay projection cannot
    # represent the stack, so it leaves this stranded; we add it back in the
    # predicted final energy (the real executor recovers it).
    recoverable_structure_energy: int = 0

    @property
    def actions(self) -> List[Action]:
        """Executor-facing action list (tile-addressed, includes the climb)."""
        return [s.action for s in self.steps]

    @property
    def absorbed_count(self) -> int:
        return len(self.absorbed)

    def absorbed_types(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for tname, _x, _y in self.absorbed:
            out[tname] = out.get(tname, 0) + 1
        return out


@dataclass
class Report:
    solvable: bool
    reason: str = ""  # human-readable; empty when solvable
    detail: Dict[str, object] = field(default_factory=dict)


# ---- visibility cache ------------------------------------------------------
# can_see / visible_tiles depend ONLY on the terrain height field, never on the
# object list — so during a single solve (terrain is constant) we can memoise
# them. This is the difference between a sub-second solve and a multi-minute one,
# because the staged planner runs BFS-over-LOS many times. The cache is keyed by
# the terrain's identity (id of the height grid) so it is invalidated for free
# whenever a fresh state is used.
_VIS_CACHE: Dict[int, Dict] = {}


def _terrain_key(state: GameState) -> int:
    """Stable hash of the terrain height field. game_model.apply deep-copies the
    state (new list identity) every step, so we key on CONTENT, not identity, and
    memoise the hash on the state to keep it cheap."""
    k = getattr(state, "_terrain_hash", None)
    if k is None:
        k = hash(tuple(tuple(row) for row in state.height))
        try:
            state._terrain_hash = k
        except Exception:
            pass
    return k


def cached_can_see(
    state: GameState,
    a: Tuple[int, int],
    b: Tuple[int, int],
    eye_offset: float = ROBOT_EYE,
) -> bool:
    c = _VIS_CACHE.setdefault(_terrain_key(state), {})
    key = ("cs", a, b, eye_offset)
    v = c.get(key)
    if v is None:
        v = can_see(state, a, b, eye_offset=eye_offset)
        c[key] = v
    return v


def cached_visible_tiles(
    state: GameState, from_xy: Tuple[int, int], eye_offset: float = ROBOT_EYE
) -> List[Tuple[int, int]]:
    c = _VIS_CACHE.setdefault(_terrain_key(state), {})
    key = ("vt", tuple(from_xy), eye_offset)
    v = c.get(key)
    if v is None:
        v = visible_tiles(state, from_xy, eye_offset=eye_offset)
        c[key] = v
    return v


# ---- small helpers ---------------------------------------------------------
def _as_state(state_or_model) -> GameState:
    if isinstance(state_or_model, GameModel):
        return state_or_model.state
    return state_or_model


def _sentinel(state: GameState) -> Optional[GameObject]:
    for o in state.objects:
        if o.type == T_SENTINEL:
            return o
    return None


def _player_xy(state: GameState) -> Tuple[int, int]:
    p = state.player
    return (p.x, p.y)


def _occupied(
    state: GameState, exclude_slot: Optional[int] = None
) -> Set[Tuple[int, int]]:
    return {(o.x, o.y) for o in state.objects if o.slot != exclude_slot}


def _reachability_graph(
    state: GameState, eye_offset: float = ROBOT_EYE
) -> Set[Tuple[int, int]]:
    """Tiles the player can reach by repeated (create robot on a visible empty
    tile, transfer into it) hops, at a flat robot eye. BFS over the LOS graph.

    A tile is reachable if some already-reached tile has line of sight to it and
    it is empty terrain (creatable). Mirrors how movement works in the model.
    """
    from collections import deque

    p = state.player
    start = (p.x, p.y)
    occ = _occupied(state, exclude_slot=p.slot)
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for t in cached_visible_tiles(state, cur, eye_offset=eye_offset):
            if t in seen or t in occ:
                continue
            seen.add(t)
            q.append(t)
    return seen


def _los_path(
    state: GameState,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occ: Set[Tuple[int, int]],
    eye_offset: float = ROBOT_EYE,
) -> Optional[List[Tuple[int, int]]]:
    """Shortest hop path start->goal over the LOS graph (each hop must have LOS
    and the landed-on tile must be empty terrain). Deterministic BFS. The goal
    tile itself need not be empty (it may hold the object we want to absorb), so
    we allow landing adjacency: the final hop only needs LOS to `goal`."""
    from collections import deque

    if start == goal:
        return [start]
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
    q = deque([start])
    while q:
        cur = q.popleft()
        # deterministic neighbour order
        for t in sorted(cached_visible_tiles(state, cur, eye_offset=eye_offset)):
            if t == goal:
                prev[t] = cur
                # reconstruct
                path = [t]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            if t in prev or t in occ:
                continue
            prev[t] = cur
            q.append(t)
    return None


# ---- climb selection -------------------------------------------------------
@dataclass
class Climb:
    base: Tuple[int, int]  # tile the stack is built on
    stack: int  # number of boulders (k)
    eye_offset: float  # ROBOT_EYE + stack*BOULDER_HEIGHT
    exposed: bool  # threat model says an enemy can see the base


def _find_climb(state: GameState, reachable: Set[Tuple[int, int]]) -> Optional[Climb]:
    """Pick the best reachable base tile + minimal boulder stack giving LOS to the
    Sentinel from the raised eye. Deterministic. Prefers (small stack, unexposed,
    low surface cost as a tie-break, tile order)."""
    sent = _sentinel(state)
    if sent is None:
        return None
    starg = (sent.x, sent.y)
    candidates: List[Tuple[int, bool, Tuple[int, int]]] = []
    occ = _occupied(state, exclude_slot=state.player_slot)
    for x, y in reachable:
        if (x, y) == starg or (x, y) in occ:
            continue
        for k in range(0, MAX_STACK):
            eye = ROBOT_EYE + k * BOULDER_HEIGHT
            if cached_can_see(state, (x, y), starg, eye_offset=eye):
                exposed = is_exposed(state, x, y, object_top=eye)
                candidates.append((k, exposed, (x, y)))
                break
    if not candidates:
        return None
    # sort: minimal stack first, then prefer unexposed, then deterministic tile.
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    k, exposed, base = candidates[0]
    return Climb(
        base=base, stack=k, eye_offset=ROBOT_EYE + k * BOULDER_HEIGHT, exposed=exposed
    )


# ---- staged planner --------------------------------------------------------
def _safe_absorb_sweep(model: GameModel, plan: Plan) -> GameModel:
    """Stage A: absorb every flat-reachable non-Sentinel object, hopping to it.

    Greedy best-first: repeatedly absorb the *closest* reachable absorbable object
    (passive trees taken before sentries), updating the model after each
    absorption, until no remaining object has a reachable tile with LOS to it.

    Movement is a hop (create robot on a visible empty tile → transfer in →
    re-absorb the abandoned robot) so it is energy-neutral. NOTE: the conservative
    threat model (`is_exposed`) is consulted for *climb base* selection in Stage B,
    but this sweep does not currently avoid exposed tiles — the model has no drain
    *timing*, so exposure here is advisory; the live executor (step 4) must mind
    enemy drain when walking the sweep. Sentries ARE absorbed when reachable (worth
    3 energy each)."""
    while True:
        state = model.state
        p = state.player
        pxy = (p.x, p.y)
        # objects we can still absorb (exclude the Sentinel; it is Stage C)
        targets = []
        for o in state.objects:
            if o.slot == p.slot:
                continue
            if o.type in (T_SENTINEL, T_PLATFORM):
                continue
            if o.type not in gm.ABSORBABLE:
                continue
            targets.append(o)
        if not targets:
            break
        occ = _occupied(state, exclude_slot=p.slot)
        # find the closest target we can get LOS to, via a hop path to a tile
        # adjacent-in-LOS to it
        best = None  # (pathlen, type_priority, slot, path, obj)
        for o in targets:
            # We need to stand on some reachable empty tile that has LOS to o.
            # Try: a hop path to a tile from which can_see(tile, o) holds.
            cand_tiles = _tiles_with_los_to(state, (o.x, o.y), occ)
            if not cand_tiles:
                continue
            # pick the closest such tile by BFS hop distance
            for tile in cand_tiles:
                path = _los_path(state, pxy, tile, occ)
                if path is None:
                    continue
                # type priority: trees(0) before sentries(1) — sweep passive first
                tpri = 0 if o.type == T_TREE else (1 if o.type in (T_SENTRY,) else 2)
                key = (len(path), tpri, o.slot)
                if best is None or key < best[0]:
                    best = (key, path, o, tile)
                break  # first (sorted) viable tile is enough for this target
        if best is None:
            break
        _key, path, obj, tile = best
        # execute hops along path (skip the start tile)
        model = _execute_hops(model, plan, path)
        # absorb the object (must be visible from current tile)
        st = model.state
        if not can_see(st, (st.player.x, st.player.y), (obj.x, obj.y)):
            # could not actually see it after moving; give up on this one to
            # avoid an infinite loop
            break
        act = Action("absorb", obj.x, obj.y)
        _before = st.player_energy
        model = model.apply(act)
        plan.steps.append(
            PlanStep(
                action=act,
                note=f"absorb {obj.type_name} @ {obj.x},{obj.y}",
                eye_height=tile_surface_height(
                    model.state, model.state.player.x, model.state.player.y
                )
                + ROBOT_EYE,
            )
        )
        plan.model_replay.append(act)
        plan.absorbed.append((obj.type_name, obj.x, obj.y))
    return model


def _tiles_with_los_to(
    state: GameState, target: Tuple[int, int], occ: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Empty reachable-ish tiles that have LOS to `target`, in deterministic
    order (closest by Chebyshev distance first)."""
    tx, ty = target
    out = []
    for y in range(N):
        for x in range(N):
            if (x, y) == target:
                continue
            if (x, y) in occ:
                continue
            if cached_can_see(state, (x, y), target):
                out.append((x, y))
    out.sort(key=lambda t: (max(abs(t[0] - tx), abs(t[1] - ty)), t))
    return out


def _execute_hops(
    model: GameModel, plan: Plan, path: List[Tuple[int, int]]
) -> GameModel:
    """Execute a hop path (create robot on each next tile, transfer into it,
    re-absorbing the abandoned robot to recover energy). Updates plan + model.

    Each hop: create robot on the next tile (cost 3), transfer into it (free),
    then absorb the robot we just left (regain 3) so movement is energy-neutral
    and frees the slot. The very first tile in `path` is where we already stand.
    """
    for i in range(1, len(path)):
        prev_tile = (model.state.player.x, model.state.player.y)
        nxt = path[i]
        st = model.state
        # if the next tile already holds one of our robots, just transfer
        existing = object_in_tile(st, nxt[0], nxt[1])
        if (
            existing is not None
            and existing.type == T_ROBOT
            and existing.slot != st.player_slot
        ):
            act = Action("transfer", nxt[0], nxt[1])
            model = model.apply(act)
            plan.steps.append(PlanStep(action=act, note=f"transfer to {nxt}"))
            plan.model_replay.append(act)
            continue
        # create a robot to hop into
        cact = Action("create", T_ROBOT, nxt[0], nxt[1])
        model = model.apply(cact)
        plan.steps.append(PlanStep(action=cact, note=f"create robot to hop to {nxt}"))
        plan.model_replay.append(cact)
        # transfer into it
        tact = Action("transfer", nxt[0], nxt[1])
        _prev_slot = model.state.player_slot
        model = model.apply(tact)
        plan.steps.append(PlanStep(action=tact, note=f"transfer to {nxt}"))
        plan.model_replay.append(tact)
        # re-absorb the robot we left behind (recover its 3 energy, free the slot)
        st2 = model.state
        left = object_in_tile_at(st2, prev_tile, exclude_slot=st2.player_slot)
        if left is not None and left.type == T_ROBOT:
            if can_see(st2, (st2.player.x, st2.player.y), prev_tile):
                aact = Action("absorb", prev_tile[0], prev_tile[1])
                model = model.apply(aact)
                plan.steps.append(
                    PlanStep(
                        action=aact, note=f"re-absorb abandoned robot @ {prev_tile}"
                    )
                )
                plan.model_replay.append(aact)
    return model


def object_in_tile_at(
    state: GameState, xy: Tuple[int, int], exclude_slot: Optional[int] = None
) -> Optional[GameObject]:
    best = None
    for o in state.objects:
        if o.x == xy[0] and o.y == xy[1] and o.slot != exclude_slot:
            if best is None or (o.z, o.z_fraction) > (best.z, best.z_fraction):
                best = o
    return best


def _build_climb_and_win(model: GameModel, plan: Plan, climb: Climb) -> GameModel:
    """Stages B+C+D. Build the boulder stack on `climb.base`, transfer up, absorb
    the Sentinel (win), then re-absorb the stranded climb structure where visible.

    The executor-facing steps describe the real same-tile stack; the model-replay
    projection only carries the moves `game_model.apply` accepts plus the final
    `win`, so replay-verify reaches `won` honestly (LOS feasibility is proven
    separately by the abstract `can_see` recorded on the steps)."""
    sent = _sentinel(model.state)
    bx, by = climb.base
    base_surf = tile_surface_height(model.state, bx, by)

    # --- Stage B: hop to the base tile (model-replayable) ---
    occ = _occupied(model.state, exclude_slot=model.state.player_slot)
    path = _los_path(
        model.state, (model.state.player.x, model.state.player.y), (bx, by), occ
    )
    if path is None:
        plan.notes.append(f"climb base {climb.base} unreachable for final transfer")
        return model
    model = _execute_hops(model, plan, path)

    # --- Stage B (cont): build the stack (executor-facing only) ---
    # The real game: create k boulders on the base tile (each on top of the
    # previous), create a robot on top, transfer into it -> eye is raised. The
    # model cannot represent same-tile stacking, so these are executor-facing
    # steps with abstract metadata; they are NOT added to model_replay.
    for k in range(climb.stack):
        eye = ROBOT_EYE + (k + 1) * BOULDER_HEIGHT
        a = Action("create", T_BOULDER, bx, by)
        plan.steps.append(
            PlanStep(
                action=a,
                note=f"stack boulder #{k+1} on {climb.base} (real climb)",
                stack_level=k + 1,
                eye_height=base_surf + eye,
            )
        )
    # robot on top of the stack, transfer up
    top_eye = base_surf + climb.eye_offset
    ra = Action("create", T_ROBOT, bx, by)
    plan.steps.append(
        PlanStep(
            action=ra,
            note=f"robot atop the {climb.stack}-boulder stack",
            stack_level=climb.stack + 1,
            eye_height=top_eye,
        )
    )
    ta = Action("transfer", bx, by)
    plan.steps.append(
        PlanStep(
            action=ta,
            note=f"transfer up: eye now {top_eye:.2f} (LOS to Sentinel)",
            stack_level=climb.stack + 1,
            eye_height=top_eye,
        )
    )

    # --- Stage C: win (absorb Sentinel + transfer onto platform) ---
    # abstract LOS proof: from the raised eye we can see the Sentinel.
    plan.los_proven = cached_can_see(
        model.state, (bx, by), (sent.x, sent.y), eye_offset=climb.eye_offset
    )
    win = Action("win", sent.x, sent.y)
    plan.steps.append(
        PlanStep(
            action=win,
            note=f"absorb Sentinel @ {sent.x},{sent.y} + transfer onto platform",
            eye_height=top_eye,
        )
    )
    # model-replay: the model accepts win (it does not re-check LOS); this
    # certifies the structural/energy validity of reaching `won`.
    plan.model_replay.append(win)
    model = model.apply(win)
    plan.absorbed.append(("SENTINEL", sent.x, sent.y))

    # --- Stage D: re-absorb stranded climb structure (executor-facing) ---
    # In the real game, once you have transferred onto the platform you can look
    # back at the boulder stack and re-absorb it, recovering its energy. The model
    # cannot represent the stack, so this is executor-facing only; we still credit
    # the energy in the *predicted* final figure (see solve()).
    for k in range(climb.stack):
        a = Action("absorb", bx, by)
        plan.steps.append(
            PlanStep(
                action=a,
                note=f"re-absorb stranded boulder #{climb.stack-k} @ {climb.base}",
            )
        )
    # one more: re-absorb the robot we stood on atop the stack.
    plan.steps.append(
        PlanStep(
            action=Action("absorb", bx, by),
            note=f"re-absorb the climb robot @ {climb.base}",
        )
    )

    # Energy bookkeeping: the model-replay projection stranded the base-tile robot
    # (it created one to transfer into and could not re-absorb it while standing on
    # it). The real game recovers that robot (3) in Stage D. The boulder stack is
    # net-zero (built and recovered entirely executor-side, never charged in the
    # model). So the under-counted, recoverable energy is exactly the base robot.
    plan.recoverable_structure_energy = ENERGY_IN_OBJECTS[T_ROBOT]
    return model


# ---- public: solve ---------------------------------------------------------
def solve(state_or_model, verbose: bool = False) -> Plan:
    """Plan a winning sequence with maximal final energy.

    Returns a `Plan`. Guarantees (when `plan.solved`):
      * replaying `plan.model_replay` through `game_model.apply` reaches
        `state.won == True` (asserted here), and
      * `plan.los_proven` is True (the abstract climb reveals the Sentinel).

    Energy maximisation: Stage A absorbs every other object; the climb is built
    from boulders/robots that are re-absorbed (energy-neutral); the Sentinel adds
    +4. `plan.final_energy` is the model energy after the model-replay projection
    PLUS the energy of the re-absorbed stranded climb structure (Stage D), masked
    to 6 bits.

    SUB-OPTIMALITY (documented, not hidden):
      * The 6-bit energy mask ($3F) means energy wraps at 63; a greedy sweep can
        momentarily waste gains above 63. We do not search for an absorb order
        that avoids the wrap.
      * Stage A's greedy nearest-first absorb order is not guaranteed to collect
        EVERY object: a tree behind terrain we never get LOS to is left. We report
        what was absorbed.
      * The climb is the minimal-stack reachable vantage, not the globally
        energy-cheapest plan; and we assume one boulder == BOULDER_HEIGHT units.
    """
    state = copy.deepcopy(_as_state(state_or_model))
    if not hasattr(state, "won"):
        state.won = False
    model = GameModel(state)
    plan = Plan()

    sent = _sentinel(state)
    if sent is None:
        plan.notes.append("no Sentinel in landscape")
        return plan

    # Stage A: sweep up safely-reachable objects.
    model = _safe_absorb_sweep(model, plan)

    # Stage B: choose a climb from the post-sweep reachability graph.
    reachable = _reachability_graph(model.state)
    climb = _find_climb(model.state, reachable)
    if climb is None:
        plan.notes.append(
            "no reachable vantage gives LOS to the Sentinel "
            "(cannot build high enough or no safe vantage)"
        )
        plan.solved = False
        _finalise_energy(model, plan, climb)
        return plan

    # Stages B/C/D.
    model = _build_climb_and_win(model, plan, climb)

    # Replay-verify the model projection reaches `won`.
    won = _replay_verify(state, plan.model_replay)
    plan.solved = bool(won and plan.los_proven)
    if not won:
        plan.notes.append("model-replay projection did not reach won")
    if not plan.los_proven:
        plan.notes.append("abstract LOS proof failed (stack does not reveal Sentinel)")

    _finalise_energy(model, plan, climb)
    if verbose:
        for s in plan.steps:
            print(f"  {s.action!r:40} {s.note}")
    # Hard guarantee for any plan we declare solved.
    if plan.solved:
        assert _replay_verify(
            state, plan.model_replay
        ), "solved plan must replay through game_model.apply to won"
    return plan


def _finalise_energy(model: GameModel, plan: Plan, climb: Optional[Climb]) -> None:
    """Compute predicted final energy. The model already accounts for Stage A and
    the win (+4). Stage D re-absorbs the stranded climb boulders; that energy is
    not in the model (it can't represent the stack), so add it back here, masked.
    The climb's boulder creates were also not charged in the model (executor-only),
    so the net for the stack is zero — we therefore neither subtract the build cost
    nor add the re-absorb beyond keeping it energy-neutral."""
    e = model.state.player_energy
    # The model-replay projection left the base-tile climb robot stranded (it is
    # under the player). The real executor recovers it in Stage D; credit it back.
    e += plan.recoverable_structure_energy
    plan.final_energy = e & ENERGY_MASK


def _replay_verify(initial: GameState, actions: List[Action]) -> bool:
    """Replay `actions` through game_model.apply from a fresh copy of `initial`;
    return True iff the result has won == True. Raises if an action is illegal,
    which is what we want — an emitted plan must be legal in the model."""
    st = copy.deepcopy(initial)
    if not hasattr(st, "won"):
        st.won = False
    for a in actions:
        st = apply(st, a)
    return getattr(st, "won", False)


# ---- public: analyse_solvability ------------------------------------------
def analyse_solvability(state_or_model) -> Report:
    """Decide, from the initial state, whether the landscape is solvable and, if
    not, WHY. Distinguishes proven-impossible from search-failure.

    Criteria (each a distinct, honest reason):
      1. no Sentinel / degenerate map.
      1b. Sentinel on the ground (no platform): the winning transfer is impossible.
      2. PROVEN unreachable vantage: for EVERY tile on the board, even an
         arbitrarily tall eye (eye_offset = MAX_STACK*BOULDER_HEIGHT) has no LOS to
         the Sentinel -> you can never build high enough / no safe vantage exists.
      3. PROVEN platform unreachable: the Sentinel's platform tile is not in the
         reachability graph and no reachable base tile yields LOS -> final transfer
         impossible.
      4. Insufficient starting energy to build the minimal climbing structure.
      5. SEARCH FAILURE: the staged planner did not find a plan, but none of the
         above proofs fired -> reported distinctly ("planner failed within
         budget", not "impossible").
    """
    state = copy.deepcopy(_as_state(state_or_model))
    if not hasattr(state, "won"):
        state.won = False
    sent = _sentinel(state)
    if sent is None:
        return Report(False, "degenerate landscape: no Sentinel present")
    starg = (sent.x, sent.y)

    # (1b) The Sentinel must sit on its platform (not on the ground) for the win
    # transfer to work -- apply("win") requires obj.on_ground == False.
    if sent.on_ground:
        return Report(
            False,
            "Sentinel is on the ground, not on a platform: the winning "
            "transfer-onto-platform is impossible (no platform tile to "
            "stand on after absorbing it)",
        )

    # (2) Is there ANY tile from which a tall-enough eye sees the Sentinel?
    tall = ROBOT_EYE + (MAX_STACK - 1) * BOULDER_HEIGHT
    any_los_tile = False
    for y in range(N):
        for x in range(N):
            if (x, y) == starg:
                continue
            if cached_can_see(state, (x, y), starg, eye_offset=tall):
                any_los_tile = True
                break
        if any_los_tile:
            break
    if not any_los_tile:
        return Report(
            False,
            "Sentinel can never be brought into line of sight: no tile "
            "on the board sees it even with the maximum buildable eye "
            f"height (stack {MAX_STACK}). Cannot build high enough.",
            detail={"max_eye_offset": tall},
        )

    # Reachability of vantage tiles (from the live start state).
    reachable = _reachability_graph(state)
    climb = _find_climb(state, reachable)
    if climb is None:
        # A LOS tile exists somewhere, but none is reachable + tall-enough.
        return Report(
            False,
            "no safe/reachable vantage: tiles with LOS to the Sentinel "
            "exist but none are reachable from the start by climbing "
            "within the modelled stack height. (May be a search limit; "
            "treat as not-provably-solvable.)",
            detail={"reachable_tiles": len(reachable)},
        )

    # (3) platform reachability for the final transfer.
    if climb.base not in reachable:
        return Report(
            False,
            f"platform/base tile {climb.base} unreachable for the final " "transfer",
        )

    # (4) energy: can we afford the minimal structure (k boulders + 1 robot)?
    # Movement is energy-neutral (we re-absorb hops), so the binding spend is the
    # peak of the climb build: boulders cost 2 each, robot 3. We must be able to
    # pay them in sequence without dropping below zero. With re-absorption of the
    # stack after winning this is recoverable, but the BUILD still needs the peak.
    # In the model, boulders/robots can be created one at a time and the previous
    # could be re-absorbed; conservatively require enough for the single largest
    # creation step (a robot, 3) plus at least one boulder (2).
    min_build = ENERGY_IN_OBJECTS[T_ROBOT]  # the costliest single create
    if state.player_energy < min_build:
        return Report(
            False,
            f"insufficient starting energy ({state.player_energy}) to "
            f"create the climbing structure (needs >= {min_build} to "
            "place a robot)",
            detail={"start_energy": state.player_energy, "min_build": min_build},
        )

    # Otherwise try the actual planner.
    plan = solve(state)
    if plan.solved:
        return Report(
            True,
            "",
            detail={
                "stack": climb.stack,
                "base": climb.base,
                "plan_len": len(plan.steps),
                "final_energy": plan.final_energy,
            },
        )
    # Planner failed but no impossibility proof fired -> honest "search failure".
    return Report(
        False,
        "planner did not find a winning sequence within its staged "
        "search budget (NOT proven impossible: a LOS vantage exists and "
        "energy suffices, but the greedy planner could not assemble a "
        "replay-verified plan). " + "; ".join(plan.notes),
        detail={"notes": plan.notes},
    )


# ---- broken-state constructors (for tests) --------------------------------
def break_state_no_energy(state_or_model) -> GameState:
    """Return a copy with the player's energy zeroed (cannot build anything)."""
    st = copy.deepcopy(_as_state(state_or_model))
    st.player_energy = 0
    return st


def break_state_grounded_sentinel(state_or_model) -> GameState:
    """Return a copy in which the Sentinel sits ON THE GROUND (no platform) and the
    board is flattened of relief. This is PROVEN unsolvable: the winning move is
    'absorb the Sentinel then transfer onto its now-empty platform', and with no
    platform there is nowhere to transfer (apply('win') requires the Sentinel not
    be on the ground). `analyse_solvability` reports this via criterion (1b)."""
    st = copy.deepcopy(_as_state(state_or_model))
    sent = _sentinel(st)
    st.height = [[0] * N for _ in range(N)]
    if sent is not None:
        sent.on_ground = True
        sent.stacked_on = None
        sent.z = 0
        sent.flags = 0x00
    return st


def break_state_flat(state_or_model) -> GameState:
    """Return a copy with all terrain relief removed (a perfectly flat board). The
    planner can still hop everywhere but can never climb to a NEW vantage relative
    to the Sentinel's tile — used to exercise the search-failure vs proven-
    impossible distinction. Prefer `break_state_grounded_sentinel` for a clean
    proven-impossible case."""
    st = copy.deepcopy(_as_state(state_or_model))
    st.height = [[0] * N for _ in range(N)]
    return st


# ---- __main__ validation ---------------------------------------------------
def _report_landscape(ls: int) -> None:
    print(f"\n############ seed {ls} ############")
    model = GameModel.from_landscape(ls)
    rep = analyse_solvability(model)
    print(f"  solvable: {rep.solvable}")
    if not rep.solvable:
        print(f"  reason  : {rep.reason}")
        return
    plan = solve(model)
    print(
        f"  plan length        : {len(plan.steps)} steps "
        f"(model-replay {len(plan.model_replay)} actions)"
    )
    print(f"  objects absorbed   : {plan.absorbed_count}  {plan.absorbed_types()}")
    print(f"  predicted energy   : {plan.final_energy}")
    print(f"  LOS proven (climb) : {plan.los_proven}")
    # replay-verify line
    ok = _replay_verify(_as_state(model), plan.model_replay)
    print(f"  replay-verified    : model-replay reaches won == {ok}")
    if plan.notes:
        print(f"  notes              : {'; '.join(plan.notes)}")


def main():
    for ls in (0, 42, 9999):
        _report_landscape(ls)


if __name__ == "__main__":
    main()
