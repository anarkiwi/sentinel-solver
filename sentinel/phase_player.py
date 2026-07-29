"""A phase-2 player: from a position of freedom, convert the board.

Time is free -- an idle frame costs only the drain a cone takes while it is on you --
so exposure is a question of WHEN, not where, and no forecast is needed: ``enemies.step``
is byte-exact, so a gap is found by running the world forward and looking.
"""

import argparse
import math

from sentinel import actions, enemies, memmap as mm, terrain
from sentinel.game import Game
from sentinel.playerbase import BOULDER_H, EYE_EPS, ROBOT_EYE, BasePlayer, _Views

WAIT_QUANTUM = 60  # frames advanced per probe while waiting for a gap
WAIT_HORIZON = 20000  # frames to look ahead for one
HOP_VERBS = ("boulder", "robot", "transfer")  # the verbs one climb is made of
ROLLOUT_ACTIONS = 200  # decision ticks a tie-breaking rollout plays before conceding
ARBITRATE_ACTIONS = 2  # decision ticks each arbitrated option plays out on its fork: a BOUND on how much of the fixed ladder's own error the score absorbs (open item 13), not a derived depth -- deeper scores worse, on and off the suite


class PhasePlayer(BasePlayer):
    """Absorb what is reachable, climb when it is not, and only ever act in a gap."""

    def __init__(self, game, verbose=False, horizon=WAIT_HORIZON, rollout=False):
        super().__init__(game, verbose=verbose)
        self.horizon = horizon
        self.waited = 0.0
        self.rollout = rollout  # a fork plays a FIXED policy: nesting explodes

    def _drained_over(self, frames):
        """Would the body lose energy standing here for ``frames``?  Simulated, exact.

        A body no enemy has any sightline to cannot be drained however the cones turn,
        so that case is answered by geometry and the simulation is skipped.
        """
        if not self._exposures(self.st, self.st.player):
            return False
        probe = self.st.clone()
        before = probe.energy
        enemies.advance_frames(probe, int(frames))
        return probe.energy < before or actions.player_dead(probe)

    def _under_fire(self):
        """Is a cone on the body NOW -- i.e. does idling cost energy?"""
        return self._drained_over(WAIT_QUANTUM)

    def _wait_for_gap(self, frames):
        """Idle until ``frames`` of quiet are available; False if none is.

        Idling is free only while unexposed.  Once a cone is on the body, waiting is
        the one thing that cannot help -- every quantum is a drain -- so the caller is
        told to act now and take the hit instead.
        """
        spent = 0
        while spent < self.horizon:
            if not self._drained_over(frames):
                return True
            if self._under_fire():
                return False
            enemies.advance_frames(self.st, WAIT_QUANTUM)
            self.frames += WAIT_QUANTUM
            self.waited += WAIT_QUANTUM
            spent += WAIT_QUANTUM
            if actions.player_dead(self.st) or self.st.energy <= 0:
                return False
        return False

    def _span(self, verb, view, tile):
        """What ``verb`` on ``tile`` will really take: its aim plus its settle.

        Priced from the view the executor will actually fire, so the gap asked for is
        the gap the action needs -- no invented duration.
        """
        return self._step_aim_frames(verb, view) + self._settle(
            verb, view, self._settle_eye(verb, tile)
        )

    def _hop_span(self, tile, k):
        """Frames a whole ``k``-boulder hop plus one act on arrival will occupy."""
        view = self._views_now().get(tile, band=True)
        if view is None:
            return math.inf
        per = {v: self._span(v, view, tile) for v in HOP_VERBS + ("absorb",)}
        return k * per["boulder"] + per["robot"] + per["transfer"] + per["absorb"]

    def _views_now(self):
        return _Views(self.st)

    def _do(self, verb, tile, views, cost=0, band=False):
        """Fire ``verb`` at ``tile``, in a gap if one can be had, at once if not.

        ``band`` buys the full pitch-band sweep, which is ~97% of this player's
        runtime, so it is spent only on named targets -- an enemy, the platform -- and
        never on bulk work that the primary plane already answers.
        """
        view = views.get(tile, band=band)
        if view is None or self.st.energy - cost < 0:
            return False
        if not self._wait_for_gap(self._span(verb, view, tile)):
            if not self._under_fire():
                return False  # no gap and no pressure: this action is not worth it
        return self._fire(verb, tile, view)

    def _twin_class(self):
        """The class a fork is built from: THIS one, so the policies arbitration
        searches are the policies that will actually be played.  Naming the base here
        made every option on a subclass be evaluated by the base's own generators.
        A player whose constructor or executor is not the simulator's overrides it."""
        return type(self)

    def _fork(self):
        """A copy of this player on a copy of the world, for trying a policy out."""
        twin = self._twin_class()(
            Game(self.st.clone()), verbose=False, horizon=self.horizon, rollout=True
        )
        twin.cursor = list(self.cursor)
        twin.last_bearing = self.last_bearing
        twin.frames = self.frames
        return twin

    @staticmethod
    def _score(player):
        """How well a rollout ended, worst to best, on facts rather than weights."""
        st = player.st
        return (
            not actions.player_dead(st) and st.energy > 0,
            -len(enemies.enemy_slots(st)),
            round(st.eye_z(), 3),
            st.energy,
        )

    def _arbitrate(self, options):
        """Run each option on a fork for a few ticks and take the one that ends best.

        The right move at an ambiguous point is board- and position-dependent -- a
        fixed order of refuel/salvage/climb wins one board and loses another -- so it
        is decided by playing each out on the byte-exact simulator instead of ranked.
        """
        if self.rollout:
            return False
        best = None
        for name in options:
            twin = self._fork()
            try:
                if not getattr(twin, name)(_Views(twin.st)):
                    continue
                for _ in range(ARBITRATE_ACTIONS):
                    if actions.player_dead(twin.st) or actions.won(twin.st):
                        break
                    twin._tick()
            except Exception:
                continue
            score = self._score(twin)
            if best is None or score > best[0]:
                best = (score, name)
        if best is None:
            return False
        return bool(getattr(self, best[1])(_Views(self.st)))

    def _harvest(self, views):
        """Top up from whatever is reachable; True when the purse actually rose.

        A policy reports whether it MOVED, and `_arbitrate` drops every option whose
        policy returned False.  `_refuel` answers a different question -- "do I now hold
        `want`", which `_strike`/`_finish` need -- and answers it False when one +1 tree
        is all there is, throwing away the only action on the board.
        """
        before = self.st.energy
        self._refuel(views, before + 2)
        return self.st.energy > before

    def _supply(self, views):
        """Phase 1a as a policy: bank, or move to where the fuel is."""
        offer = self._best_climb(views, affordable=False)
        if offer is None or offer[2] <= self.st.energy:
            return False
        return self._establish(views, offer[2])

    def _strike(self, views):
        """Absorb any enemy we can land, Sentinel last ($1B8E).

        Taking the Sentinel is the point of no return: $1B91 reads objects_flags[0]
        absolutely, so once its slot is empty EVERY later absorb is refused.  The
        endgame's robot and hyperspace toll must therefore be banked BEFORE the strike.
        """
        st = self.st
        foes = enemies.enemy_slots(st)
        endgame = 2 * mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
        for e in foes:
            if st.obj_type[e] == mm.T_SENTINEL:
                if len(foes) > 1:
                    continue
                banked = st.energy + mm.ENERGY_IN_OBJECTS[mm.T_SENTINEL]
                if banked < endgame and not self._refuel(
                    views, endgame - mm.ENERGY_IN_OBJECTS[mm.T_SENTINEL]
                ):
                    return False  # cannot afford the finish yet; gather first
            tile = tuple(st.tile_of(e))
            if terrain.top_object(st, *tile) != e or not actions.can_absorb(st, e):
                continue
            if self._do("absorb", tile, views, band=True):
                return True
        return False

    def _refuel(self, views, want, bank=False):
        """Absorb reachable objects until ``want`` is in hand, or everything if ``bank``.

        Time costs nothing in a gap and absorbing is the only income, so before
        committing to a climb it is always right to take what is going: arriving at a
        new stance broke is what kills this player, not the frames spent banking.
        """
        took = False
        for _value, tile in sorted(
            self._reclaim_targets(self.st, True), key=lambda vt: -vt[0]
        ):
            if not bank and self.st.energy >= want:
                return True
            if self.st.energy >= mm.ENERGY_MASK - 4:
                break
            if self._do("absorb", tuple(tile), views, band=True):
                took = True  # an absorb only REMOVES an occluder: the sweep stays valid
        return took if bank else self.st.energy >= want

    def _stance_base(self, tile):
        """Foot height a stack on ``tile`` builds from, or None if not stackable.

        Bare tile: a foot.  Stacked: already a robot eye, since $1F66 gives every
        create `z_frac = $E0` -- two conventions in one function, and the callers add
        `ROBOT_EYE` to both -- open item 14.
        """
        top = self._top(tile)
        if top is None:
            return terrain.tile_byte(self.st, *tile) >> 4
        otype = self.st.obj_type[top]
        if otype in (mm.T_BOULDER, mm.T_PLATFORM):
            return self._base_z(top) + (1.0 if otype == mm.T_PLATFORM else BOULDER_H)
        return None

    def _landing_holds(self, tile, k):
        """Would a body that lands on ``tile`` still be alive long enough to act?

        A gap at the tile we are LEAVING says nothing about the one we arrive on, and
        hopping into a live cone at low energy is how this player dies.  Built on a
        clone and stepped with the byte-exact enemies, so it is observed, not forecast.
        """
        probe = self.st.clone()
        for _ in range(k):
            if actions.create(probe, mm.T_BOULDER, tile) is None:
                return False
        slot = actions.create(probe, mm.T_ROBOT, tile)
        if slot is None or not actions.transfer(probe, slot):
            return False
        span = self._hop_span(tile, k)  # the whole hop plus one act, not a fragment
        if span == math.inf:
            return False
        enemies.advance_frames(probe, int(span))
        return probe.energy > 0 and not actions.player_dead(probe)

    def _fuel_near(self, tile, reach=10):
        """Absorbable objects lying near ``tile`` -- what a landing there can live on."""
        st = self.st
        n = 0
        for slot in st.occupied_slots():
            if slot == st.player or st.obj_type[slot] in mm.ENEMY_TYPES:
                continue
            if st.obj_type[slot] == mm.T_PLATFORM:
                continue
            ox, oy = st.tile_of(slot)
            if abs(ox - tile[0]) <= reach and abs(oy - tile[1]) <= reach:
                n += 1
        return n

    def _climb_candidates(self, views, affordable):
        """[(score, tile, k, cost, gain)] for every climb the ROM's own gates allow.

        Candidates rank by height per energy -- the purse is finite and only refills
        by absorbing -- and ``affordable`` restricts to what is payable now, which is
        how the caller tells establish (cannot pay) from breakout (can).
        """
        st = self.st
        my_eye = st.eye_z()
        out = []
        for tile in views.band():  # candidates need the down-look plane
            base = self._stance_base(tile)
            if base is None:
                continue
            k = max(0, math.ceil((my_eye + EYE_EPS - ROBOT_EYE - base) / BOULDER_H))
            gain = base + BOULDER_H * k + ROBOT_EYE - my_eye
            if gain <= 0 or k > 3:
                continue
            cost = 2 * k + mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
            if affordable and cost > st.energy:
                continue
            fuel = self._fuel_near(tile)
            if affordable:
                if st.energy - cost <= 0:
                    continue  # $1A00: a drain arriving at zero energy KILLS
                if st.energy - cost < mm.ENERGY_IN_OBJECTS[mm.T_ROBOT] and not fuel:
                    continue  # arriving broke with nothing to absorb is a dead end
                if not self._landing_holds(tile, k):
                    continue  # the destination's cone, not ours, decides this
            score = (gain / cost, fuel)  # height per unit of a scarce, finite purse
            out.append((score, tuple(tile), k, cost, gain))
        return out

    def _best_climb(self, views, affordable):
        """(tile, k, cost, gain) for the biggest eye gain on offer, or None."""
        cands = self._climb_candidates(views, affordable)
        if not cands:
            return None
        top = max(c[0] for c in cands)
        tied = [c for c in cands if c[0] == top]
        if len(tied) > 1 and affordable and not self.rollout:
            return self._settle_tie(tied)
        return tied[0][1:]

    def _settle_tie(self, tied):
        """Play each tied climb out to the END and keep one that wins.

        When the score cannot separate the candidates it holds no information about
        them, and no deeper VALUATION helps -- ls110's winning hop is 40 actions from
        its payoff.  What separates them is the outcome, so take the outcome: a fork
        plays the fixed ladder, which is a whole board in seconds, and ties are rare.
        """
        for cand in tied:
            _score, tile, k, _cost, _gain = cand
            twin = self._fork()
            try:
                if not twin._build_and_mount(_Views(twin.st), tile, k):
                    continue
                twin.run(max_actions=ROLLOUT_ACTIONS)
            except Exception:  # pylint: disable=broad-except
                continue
            if actions.won(twin.st):
                return cand[1:]
        return tied[0][1:]

    def _mount(self, views):
        """Stand on a robot we already own that is higher than we are.

        A transfer costs nothing, and a robot left behind by a build whose transfer
        failed is otherwise dead weight: ``_stance_base`` will not stack on a robot, so
        nothing else can ever use that tile again.

        The candidate's eye is over-stated by a whole `ROBOT_EYE` -- a robot's stored z
        IS its eye -- so this can transfer DOWNWARD (open item 14).
        """
        st = self.st
        best = None
        for slot in st.occupied_slots():
            if slot == st.player or st.obj_type[slot] != mm.T_ROBOT:
                continue
            top = terrain.top_object(st, *st.tile_of(slot))
            if top != slot:
                continue
            eye = st.eye_z(slot) + ROBOT_EYE
            if eye <= st.eye_z() + EYE_EPS:
                continue
            if not self._mount_holds(slot):
                continue  # the tile it stands on must keep us alive once we are on it
            tile = tuple(st.tile_of(slot))
            if st.energy < mm.ENERGY_IN_OBJECTS[mm.T_ROBOT] and not self._fuel_near(
                tile
            ):
                continue  # height is no use somewhere we cannot eat
            if best is None or eye > best[0]:
                best = (eye, tuple(st.tile_of(slot)))
        if best is None:
            return False
        return self._do("transfer", best[1], views, band=True)

    def _mount_holds(self, slot):
        """Would standing on ``slot`` leave us alive long enough to act there?"""
        probe = self.st.clone()
        if not actions.transfer(probe, slot):
            return False
        tile = tuple(self.st.tile_of(slot))
        view = self._views_now().get(tile, band=True)
        if view is None:
            return False
        span = self._span("transfer", view, tile) + self._span("absorb", view, tile)
        enemies.advance_frames(probe, int(span))
        return probe.energy > 0 and not actions.player_dead(probe)

    def _establish(self, views, price):
        """Phase 1a: bank energy, and stay where the fuel is.

        The board's climbs cost more than a starting purse, so the first job is stock,
        not height: harvest everything in reach and, when a neighbourhood is stripped,
        move to the densest fuel rather than to the tallest tile.
        """
        if self._refuel(views, price, bank=True):
            return True
        st = self.st
        best = None
        for tile in views.band():
            base = self._stance_base(tile)
            if base is None or tuple(tile) == tuple(st.player_xy()):
                continue
            k = 0  # a supply hop buys reach, not height
            cost = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
            if cost > st.energy:
                continue
            fuel = self._fuel_near(tile)
            if fuel <= self._fuel_near(tuple(st.player_xy())):
                continue
            if best is None or fuel > best[0]:
                best = (fuel, tuple(tile), k, cost)
        if best is None:
            return False
        return self._build_and_mount(views, best[1], best[2])

    def _breakout(self, views):
        """Phase 1b: spend the bank on the biggest climb available."""
        pick = self._best_climb(views, affordable=True)
        if pick is None:
            return False
        tile, k, _cost, _gain = pick
        return self._build_and_mount(views, tile, k)

    def _build_and_mount(self, views, tile, k):
        """Stack ``k`` boulders on ``tile``, cap with a robot, and transfer up."""
        for _ in range(k):
            if not self._do(
                "boulder", tile, views, mm.ENERGY_IN_OBJECTS[mm.T_BOULDER], band=True
            ):
                return False
            views = _Views(self.st)
        cap_view = views.get(tile, band=True)
        if cap_view is None:
            return False
        cap_span = self._span("robot", cap_view, tile) + self._span(
            "transfer", cap_view, tile
        )
        if not self._wait_for_gap(cap_span) and not self._under_fire():
            return False  # do not buy a cap we cannot climb onto
        if not self._do(
            "robot", tile, views, mm.ENERGY_IN_OBJECTS[mm.T_ROBOT], band=True
        ):
            return False
        return self._do("transfer", tile, _Views(self.st), band=True)

    def _finish(self, views):
        """Sentinel gone: stand on the platform and hyperspace off it.

        $216A spends 3 on the terminal hyperspace and $2170 KILLS on underflow, so the
        win needs that in hand first.  With every enemy absorbed nothing drains, so
        topping up is free.
        """
        st = self.st
        ptile = tuple(st.platform_xy)
        toll = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
        if actions.on_platform(st):
            if st.energy < toll and not self._refuel(views, toll):
                return False
            if st.energy < toll:
                return False
            self._hyperspace()
            return True
        top = self._top(ptile)
        standing = top is not None and st.obj_type[top] == mm.T_ROBOT
        need = toll if standing else toll + mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
        if st.energy < need and not self._refuel(views, need):
            return False  # climbing the platform broke is a dead end, not a win
        if not standing:  # $1F38 refuses a create on a tile that already carries one
            if not self._do(
                "robot", ptile, views, mm.ENERGY_IN_OBJECTS[mm.T_ROBOT], band=True
            ):
                return False
        return self._do("transfer", ptile, _Views(st), band=True)

    def _barren(self, views):
        """Whether NO action of any class exists from this stance -- gaps and timing aside.

        Every generator empty is a different state from every generator refused: the
        second is a wait, the first cannot change by waiting.
        """
        st = self.st
        band = views.band()
        if not band:
            return True  # nothing landable on either lattice: no verb has a target
        for e in enemies.enemy_slots(st):
            tile = tuple(st.tile_of(e))
            if tile in band and terrain.top_object(st, *tile) == e:
                if actions.can_absorb(st, e):
                    return False
        for _value, tile in self._reclaim_targets(st, True):
            if tuple(tile) in band:
                return False
        if self._climb_candidates(views, True):
            return False
        for slot, tile in self._robot_bodies():
            if tuple(tile) in band and st.eye_z(slot) > st.eye_z():
                return False
        return True

    def _relocate(self, views):
        """Leave a stance that offers nothing: $216A moves the body with no sightline.

        A barren stance has no income and no climb, so it loses whether or not a cone
        ever finds it -- waiting cannot change it and only position can.  The toll is 3
        ($216A) and $2170 kills on underflow, so that is the whole affordability test.
        The landing is PRNG-driven and deliberately unknowable here, so a relocation is
        never SCORED, only taken.
        """
        if self.st.energy < mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]:
            return False
        if not self._barren(views):
            return False
        self._hyperspace()
        return True

    def _tick(self):
        views = _Views(self.st)
        if self.st.is_empty(actions.SENTINEL_SLOT):
            # endgame is exclusive: climbing here spends the hyperspace toll on eye
            if self._finish(views):
                return
            want = 2 * mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
            if self.st.energy < want and self._refuel(views, want):
                return
            enemies.advance_frames(self.st, WAIT_QUANTUM)
            self.frames += WAIT_QUANTUM
            return
        if self._strike(views):  # phase 2: convert whenever a kill is on
            return
        if self.rollout:  # inside a fork: fixed order, or the nesting explodes
            for policy in (self._breakout, self._supply, self._harvest, self._mount):
                if policy(views):
                    return
        elif self._arbitrate(["_breakout", "_supply", "_harvest", "_mount"]):
            return
        if self._under_fire():  # cornered: anything is better than standing here
            for tile in list(views.primary())[:24]:
                if self._do("robot", tile, views, mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]):
                    return
        if self._relocate(views):  # nothing here to do at all: move, do not idle
            return
        enemies.advance_frames(self.st, WAIT_QUANTUM)
        self.frames += WAIT_QUANTUM


def main(argv=None):
    """Play one landscape offline, by the number a player types on the keypad."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", type=int)
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    game = Game.typed(args.landscape)
    player = PhasePlayer(game, verbose=not args.quiet)
    won = player.run(max_actions=args.max_actions)
    print(
        f"landscape {args.landscape}: {'WON' if won else 'lost'} in "
        f"{len(player.trace)} actions / {player.frames} frames, "
        f"energy {game.energy}, dead={actions.player_dead(game.state)}"
    )
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
