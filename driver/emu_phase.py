#!/usr/bin/env python3
"""The phase player on the emulated game, every lookahead run on the emulator itself.

A fork is a machine clone acting through the same keyboard driver, so a gap is found by
doing the action on a fork and looking; view selection alone stays on the model.
"""

import collections
import hashlib
import types

from driver.live_player import LivePhase
from sentinel import fleet, memmap as mm, projector
from sentinel.phase_player import _LANDING_CACHE, _LANDING_CACHE_MAX, WAIT_QUANTUM
from sentinel.playerbase import _Views

Rehearsal = collections.namedtuple("Rehearsal", "ok energy lost frames drained")
_REHEARSALS = {}
_REHEARSALS_MAX = 1 << 12


def _quiet_log(_msg):
    pass


def _view_key(view):
    if view is None:
        return None
    return (view["h_angle"], view["v_angle"], *map(int, view["cursor"]))


def _futile(rehearsal):
    """Whether firing what ``rehearsal`` played is known to be pointless: its identical
    attempt was refused outright, or it lost the body."""
    return rehearsal.lost or not (rehearsal.ok or rehearsal.drained)


def _hop_steps(tile, k):
    """The ``(verb, tile, view)`` steps of a ``k``-boulder hop, each aimed fresh."""
    tile = tuple(tile)
    return (("boulder", tile, None),) * k + (
        ("robot", tile, None),
        ("transfer", tile, None),
    )


class EmuPhase(LivePhase):
    """:class:`LivePhase` over an :class:`driver.emu.EmuMon` session, forking machine
    clones.  A rollout commits a rehearsed step by adopting the fork that played it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last = None  # (rehearsal key, the fork that played it, its Rehearsal)

    # ---- forks -----------------------------------------------------------------
    def _twin_class(self):
        return type(self)

    def _fork(self):
        """A rollout twin on a clone of this machine, standing in the same aim."""
        twin = type(self)(
            types.SimpleNamespace(bm=self.bm.clone()),
            _quiet_log,
            {"actions": [], "energy_curve": []},
            horizon=self.horizon,
            rollout=True,
        )
        twin.verbose = twin.audits = twin.kbd.advisory = False
        twin.kbd._bearing = self.kbd.committed_bearing()
        twin.acted, twin.life_lost = self.acted, self.life_lost
        twin.step_no, twin.frames = self.step_no, self.frames
        twin._refresh()
        return twin

    def _adopt(self, twin):
        """Become ``twin``: its machine, driver and the world they read."""
        self.bm, self.ex, self.kbd = twin.bm, twin.ex, twin.kbd
        self.acted, self.life_lost, self.step_no = (
            twin.acted,
            twin.life_lost,
            twin.step_no,
        )
        self.st = self.g.state = twin.st
        self._sync_aim_state()

    def _machine_key(self):
        """Digest of all a fork's outcome depends on: RAM, CPU registers and cycle, the
        monitor's owed test and held tap, and the driver's committed bearing."""
        m, mpu = self.bm.m, self.bm.m.mpu
        return (
            hashlib.blake2b(bytes(m.ram), digest_size=16).digest(),
            m.cycle,
            (mpu.a, mpu.x, mpu.y, mpu.pc, mpu.sp, mpu.p),
            self.bm._owed,
            self.bm._tap,
            self.kbd.committed_bearing(),
        )

    def _fork_jobs(self, jobs):
        """Each job on its own fresh fork, sequentially in input order (a raise as a
        :class:`fleet.Failure`): the single seam a process pool replaces."""
        out = []
        for job in jobs:
            try:
                out.append(job(self._fork()))
            except Exception as error:  # pylint: disable=broad-except
                out.append(fleet.Failure(error))
        return out

    def _on_fork(self, job):
        """``job`` on one fresh fork via :meth:`_fork_jobs`, re-raising its raise."""
        got = self._fork_jobs([job])[0]
        if isinstance(got, fleet.Failure):
            raise got.error
        return got

    # ---- outcomes ----------------------------------------------------------------
    def _alive(self):
        return not self._lost() and self.st.energy > 0

    def _band_view(self, tile):
        return _Views(self.st).get(tuple(tile), band=True)

    def _gain(self, verb, tile):
        """Energy ``verb`` on ``tile`` moves into the purse when it lands."""
        if verb in ("boulder", "robot"):
            otype = mm.T_BOULDER if verb == "boulder" else mm.T_ROBOT
            return -mm.ENERGY_IN_OBJECTS[otype]
        top = self._top(tile)
        if verb == "absorb" and top is not None:
            return mm.ENERGY_IN_OBJECTS[self.st.obj_type[top]]
        return 0

    def _rehearse(self, steps):
        """:class:`Rehearsal` of ``steps`` (``(verb, tile, view)``; a None view aims fresh
        from the band views) played back to back on a fork: whether all landed, the
        energy, death and frames left, and whether a drain took beyond their own costs.
        """
        key = (self._machine_key(), tuple((v, t, _view_key(w)) for v, t, w in steps))
        hit = _REHEARSALS.get(key)
        if hit is not None:
            return hit
        twin, out = self._on_fork(lambda twin: (twin, twin._play(steps)))
        self._last = (key, twin, out)
        return projector.memo(_REHEARSALS, key, _REHEARSALS_MAX, lambda: out)

    def _play(self, steps):
        """:meth:`_rehearse`'s body, run on the fork itself."""
        want, f0, ok = self.st.energy, self.ex.frames(), True
        for verb, tile, view in steps:
            view = view or self._band_view(tile)
            gain = self._gain(verb, tile)
            if view is None or not self._fire(verb, tile, view):
                ok = False
                break
            want += gain
        lost = self._lost()
        drained = lost or self.st.energy != want & mm.ENERGY_MASK
        return Rehearsal(ok, self.st.energy, lost, self.ex.frames() - f0, drained)

    def _gap(self, verb, tile, view):
        tile = tuple(tile)
        return lambda: not self._rehearse(((verb, tile, view),)).drained

    def _cap_gap(self, tile, view):
        tile = tuple(tile)
        steps = (("robot", tile, view), ("transfer", tile, None))
        return lambda: not self._rehearse(steps).drained

    def _idle_drains(self, frames):
        """Whether a fork idled ``frames`` on its machine loses energy or the body."""

        def idle(twin):
            before = twin.st.energy
            twin._wait(frames)
            return twin.st.energy < before or twin._lost()

        key = (self._machine_key(), "idle", int(frames))
        return projector.memo(
            _REHEARSALS, key, _REHEARSALS_MAX, lambda: self._on_fork(idle)
        )

    # ---- landings ------------------------------------------------------------------
    def _arrive(self):
        """The first act on arrival: the absorb ``_refuel`` would take first, else one
        idle quantum."""
        views = _Views(self.st)
        for _value, tile in sorted(
            self._reclaim_targets(self.st, True), key=lambda vt: -vt[0]
        ):
            view = views.get(tuple(tile), band=True)
            if view is not None:
                self._fire("absorb", tuple(tile), view)
                return
        self._wait(WAIT_QUANTUM)

    def _lands_after(self, steps):
        """Whether this fork, playing ``steps`` and then its act on arrival, is alive."""
        for verb, tile, _view in steps:
            view = self._band_view(tile)
            if view is None or not self._fire(verb, tile, view):
                return False
        self._arrive()
        return self._alive()

    def _landing_key(self, tile, k):
        return type(self), self._machine_key(), tuple(map(int, tile)), int(k)

    def _probe_verdicts(self, cands):
        got = self._fork_jobs(
            [lambda twin, c=c: twin._lands_after(_hop_steps(c[1], c[2])) for c in cands]
        )
        return [verdict is True for verdict in got]

    def _mount_holds(self, slot):
        tile = tuple(self.st.tile_of(slot))
        return projector.memo(
            _LANDING_CACHE,
            self._landing_key(tile, -1),
            _LANDING_CACHE_MAX,
            lambda: self._fork_jobs(
                [lambda twin: twin._lands_after((("transfer", tile, None),))]
            )[0]
            is True,
        )

    # ---- execution -------------------------------------------------------------------
    def _fire(self, verb, tile, view):
        """A rollout adopts the fork that just rehearsed this step; the top level fires
        for real and audits the rehearsal's prediction against what the machine did."""
        steps = ((verb, tuple(tile), view),)
        if self.rollout:
            last, self._last = self._last, None
            if last is not None and last[0] == (
                self._machine_key(),
                ((verb, tuple(tile), _view_key(view)),),
            ):
                if _futile(last[2]):
                    return False
                self._adopt(last[1])
                if last[2].ok:
                    self._log(verb, tile)
                return last[2].ok
            return super()._fire(verb, tile, view)
        want = self._rehearse(steps)
        if _futile(want):
            return False
        f0 = self.ex.frames()
        ok = super()._fire(verb, tile, view)
        got = Rehearsal(ok, self.st.energy, self._lost(), self.ex.frames() - f0, None)
        self.result.setdefault("lookahead_audit", []).append(
            [self.step_no, verb, list(tile), list(want[:4]), list(got[:4])]
        )
        if tuple(want[:4]) != tuple(got[:4]):
            self.live_log(f"    [lookahead] p{self.step_no} {verb}: {want} != {got}")
        return ok
