#!/usr/bin/env python3
"""Unit tests for scripts/game_model.py against real generated landscapes.

Validates the energy table, action preconditions, and energy bookkeeping for
absorb/create/transfer, using Py65Source.from_landscape() states.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game_model as gm
from game_model import (
    GameModel,
    Action,
    ENERGY_IN_OBJECTS,
    legal_actions,
    T_TREE,
    T_BOULDER,
    T_ROBOT,
    T_SENTINEL,
    T_PLATFORM,
)

FAILS = 0


def check(cond, msg):
    global FAILS
    if cond:
        print(f"  ok: {msg}")
    else:
        FAILS += 1
        print(f"  FAIL: {msg}")


def test_energy_table():
    print("[energy table $214F]")
    check(
        ENERGY_IN_OBJECTS == {0: 3, 1: 3, 2: 1, 3: 2, 4: 1, 5: 4, 6: 0},
        "energy_in_objects == [3,3,1,2,1,4,0]",
    )


def _landscape_demo(ls):
    print(f"[landscape {ls:04d}]")
    m = GameModel.from_landscape(ls)
    st = m.state
    _e0 = st.player_energy

    acts = legal_actions(st)
    check(len(acts) > 0, f"has legal actions ({len(acts)})")

    # no action targets a platform via absorb
    _plats = [o for o in st.objects if o.type == T_PLATFORM]
    for a in acts:
        if a.verb == "absorb":
            obj = gm.object_in_tile(st, a.a, a.b)
            check(
                obj.type != T_PLATFORM, f"absorb does not target platform @ {a.a},{a.b}"
            )
            break

    # absorb a visible tree: energy goes up by exactly the table value
    absorbs = [a for a in acts if a.verb == "absorb"]
    if absorbs:
        a = absorbs[0]
        obj = gm.object_in_tile(st, a.a, a.b)
        before = st.player_energy
        m2 = m.apply(a)
        gained = ENERGY_IN_OBJECTS[obj.type]
        check(
            m2.state.player_energy == (before + gained) & 0x3F,
            f"absorb {obj.type_name} energy {before}->{m2.state.player_energy} (+{gained})",
        )
        check(
            m2.state.object_by_slot(obj.slot) is None,
            "absorbed object removed from state",
        )
        check(st.player_energy == before, "original state unchanged (immutable)")

    # create a tree on a visible empty tile: energy drops by 1
    creates = [a for a in acts if a.verb == "create" and a.a == T_TREE]
    if creates:
        a = creates[0]
        before = st.player_energy
        m2 = m.apply(a)
        check(
            m2.state.player_energy == before - ENERGY_IN_OBJECTS[T_TREE],
            f"create tree energy {before}->{m2.state.player_energy} (-1)",
        )
        check(
            any(
                o.x == a.b and o.y == a.c and o.type == T_TREE for o in m2.state.objects
            ),
            "created tree present in new state",
        )
        # creating a boulder then absorbing it should be energy-neutral minus nothing extra
        bcreate = Action("create", T_BOULDER, a.b, a.c)
        m3 = m.apply(bcreate)
        check(
            m3.state.player_energy == before - ENERGY_IN_OBJECTS[T_BOULDER],
            f"create boulder energy {before}->{m3.state.player_energy} (-2)",
        )
        absorb_back = Action("absorb", a.b, a.c)
        m4 = m3.apply(absorb_back)
        check(
            m4.state.player_energy == before,
            f"create+absorb boulder round-trips energy back to {before}",
        )

    # create+transfer round trip: create a robot, transfer into it
    rcreates = [a for a in acts if a.verb == "create" and a.a == T_ROBOT]
    if rcreates and st.player_energy >= 3:
        a = rcreates[0]
        m2 = m.apply(a)
        # the new robot is at (a.b,a.c); transfer into it (free)
        tx = Action("transfer", a.b, a.c)
        # transfer must be legal now (LOS unchanged; robot present)
        m3 = m2.apply(tx)
        check(
            m3.state.player_energy == m2.state.player_energy, "transfer costs no energy"
        )
        np = m3.state.player
        check((np.x, np.y) == (a.b, a.c), "player relocated to the new robot tile")

    # Sentinel not directly absorbable from start (too high / occluded)
    sent_abs = any(
        a.verb == "absorb"
        and gm.object_in_tile(st, a.a, a.b)
        and gm.object_in_tile(st, a.a, a.b).type == T_SENTINEL
        for a in acts
    )
    check(not sent_abs, "Sentinel not directly absorbable from start tile")

    # energy never exceeds the 6-bit cap
    check(0 <= st.player_energy <= 0x3F, "player energy within 0..63")


def test_landscapes():
    for ls in (0, 66):
        _landscape_demo(ls)


def main():
    test_energy_table()
    for ls in (0, 42, 9999):
        _landscape_demo(ls)
    print()
    if FAILS:
        print(f"FAILED: {FAILS} checks")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
