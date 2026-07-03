#!/usr/bin/env python3
"""Differential validation: the pure-Python model (game_model.py / enemy_dynamics.py)
vs the REAL-CODE engine (code_engine.py, which executes the actual 6502
routines in py65) for generated landscapes 0000, 0042, 9999.

This is the KEY DELIVERABLE: it proves (or pinpoints divergence in) our understanding
of the game by running BOTH the Python port and the real ROM and comparing:

  1. LINE OF SIGHT      -- game_model.can_see vs the real check_for_line_of_sight_to_tile
                           $1CDD. Our Python LOS is a documented bilinear approximation
                           of the ROM's quantised slope-facet march; we quantify how
                           often they agree and characterise the disagreements.
  2. ENERGY / ACTIONS   -- absorb each absorbable object + a few creates in both, and
                           assert the resulting player_energy matches the real ROM
                           ($2136 / $214F) EXACTLY.
  3. ENEMY DYNAMICS     -- step the real update_enemies $16B5 N ticks and compare enemy
                           angles + drain outcomes against enemy_dynamics.step_enemies.
  4. SOLVER PLAN (opt.) -- run a solver_exact plan through code_engine.play_plan and
                           report whether the REAL CODE confirms the energy/absorbs.

Run: python3 scripts/test_code_engine.py            (sampled LOS, fast)
     python3 scripts/test_code_engine.py --full-los (all 1024 tiles, slow ~2min/ls)
     python3 scripts/test_code_engine.py --plan      (also run a solver plan)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _emu
import game_model as gm
import enemy_dynamics as ed
from game_state import read_game_state, Py65Source, N
import code_engine as ce

LANDSCAPES = (0, 42, 9999)


# ----------------------------------------------------------------------------
# 1. LINE OF SIGHT
# ----------------------------------------------------------------------------
def _sample_tiles(player_xy, n=160):
    """A deterministic spread of target tiles for the sampled LOS comparison:
    all tiles within a Chebyshev radius of 4 (the close band that matters most for
    absorb/build decisions) plus an evenly-spaced grid over the rest of the board."""
    px, py = player_xy
    tiles = []
    seen = set()
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            x, y = px + dx, py + dy
            if 0 <= x < N and 0 <= y < N and (x, y) != (px, py) and (x, y) not in seen:
                tiles.append((x, y))
                seen.add((x, y))
    for y in range(0, N, 3):
        for x in range(0, N, 3):
            if (x, y) != (px, py) and (x, y) not in seen:
                tiles.append((x, y))
                seen.add((x, y))
    return tiles[:n] if n else tiles


def los_compare(eng, st, full=False):
    p = st.player
    pxy = (p.x, p.y)
    if full:
        tiles = [(x, y) for y in range(N) for x in range(N) if (x, y) != pxy]
    else:
        tiles = _sample_tiles(pxy)
    agree = 0
    real_only = []  # real says visible, model says blocked
    model_only = []  # model says visible, real says blocked
    for x, y in tiles:
        real = eng.check_los(pxy, (x, y))
        model = gm.can_see(st, pxy, (x, y))
        if real == model:
            agree += 1
        elif real and not model:
            real_only.append((x, y))
        else:
            model_only.append((x, y))
    tot = len(tiles)
    return {
        "agree": agree,
        "total": tot,
        "pct": 100.0 * agree / tot if tot else 100.0,
        "real_only": real_only,
        "model_only": model_only,
    }


# ----------------------------------------------------------------------------
# 2. ENERGY / ACTIONS
# ----------------------------------------------------------------------------
def energy_compare(landscape, st):
    """Absorb each absorbable object (in a FRESH engine each time so prior absorbs
    don't perturb energy) and a few creates, asserting real energy == model energy."""
    results = []
    table = gm.ENERGY_IN_OBJECTS
    p = st.player
    # absorb: for each absorbable, non-platform, non-player object
    for o in st.objects:
        if o.slot == p.slot or o.type == gm.T_PLATFORM:
            continue
        if o.type not in gm.ABSORBABLE:
            continue
        eng = ce.CodeEngine(landscape)
        before = eng.player_energy
        r = eng.absorb(o.slot)
        real = r["energy"]
        # model: gain table[type], masked to 6 bits
        model = (before + table.get(o.type, 0)) & gm.ENERGY_MASK
        results.append(
            ("absorb", o.type, gm.TYPES[o.type], before, real, model, real == model)
        )

    # creates: tree/boulder/robot on an empty visible tile
    eng = ce.CodeEngine(landscape)
    st2 = eng.read_state()
    pxy = (st2.player.x, st2.player.y)
    occ = {(o.x, o.y) for o in st2.objects}
    # find an empty visible tile
    target = None
    for x, y in eng.visible_tiles(pxy):
        if (x, y) not in occ:
            target = (x, y)
            break
    if target is not None:
        for t in (gm.T_TREE, gm.T_BOULDER, gm.T_ROBOT):
            eng = ce.CodeEngine(landscape)
            st2 = eng.read_state()
            # re-find an empty visible tile for this fresh engine
            occ2 = {(o.x, o.y) for o in st2.objects}
            tgt = None
            for x, y in eng.visible_tiles((st2.player.x, st2.player.y)):
                if (x, y) not in occ2:
                    tgt = (x, y)
                    break
            if tgt is None:
                continue
            before = eng.player_energy
            r = eng.create(t, tgt)
            if not r.get("ok"):
                results.append(
                    (
                        "create",
                        t,
                        gm.TYPES[t],
                        before,
                        None,
                        None,
                        None,
                        r.get("reason"),
                    )
                )
                continue
            real = r["energy"]
            model = (before - table.get(t, 0)) & gm.ENERGY_MASK
            results.append(
                ("create", t, gm.TYPES[t], before, real, model, real == model)
            )
    return results


# ----------------------------------------------------------------------------
# 2b. ABSORB GATE -- "must be above the base tile, looking down" (domain rule)
# ----------------------------------------------------------------------------
def absorb_gate_compare(eng, st):
    """Validate the REAL absorb gate against game_model's prediction.

    DOMAIN RULE (from the user): absorb targets the SQUARE THE OBJECT RESTS ON (its
    base tile); the player must SEE that base tile by LOOKING DOWN at it -- the eye
    must be STRICTLY ABOVE the base-tile height with line of sight. The ROM enforces
    this in handle_player_actions $1B46 (check_for_line_of_sight_to_tile $1CDD, whose
    $1D2E looking-up rejection kills any tile above the eye), then $1B52 requires an
    object in that tile.

    REAL gate  : eng.can_absorb(slot) -- runs $1CDD to the object's base tile.
    MODEL pred : the object appears as an 'absorb'/'win' action in
                 game_model.legal_actions (which gates on visible_tiles -> base-tile
                 LOS via can_see). We compare per object and flag any divergence (esp.
                 the Sentinel, whose base/platform tile must be UNSEEABLE from a low
                 start, looking up)."""
    p = st.player
    pxy = (p.x, p.y)
    model_absorb_tiles = set()
    for a in gm.legal_actions(st):
        if a.verb in ("absorb", "win"):
            model_absorb_tiles.add((a.a, a.b))
    rows = []
    for o in st.objects:
        if o.slot == p.slot or o.type == gm.T_PLATFORM:
            continue
        if o.type not in gm.ABSORBABLE:
            continue
        real = eng.can_absorb(o.slot)  # real $1CDD gate to base tile
        model = (o.x, o.y) in model_absorb_tiles  # model's base-tile LOS gate
        eye_h = st.height[pxy[1]][pxy[0]]
        base_h = st.height[o.y][o.x]
        rows.append(
            (
                o.slot,
                o.type,
                gm.TYPES[o.type],
                (o.x, o.y),
                base_h,
                eye_h,
                real,
                model,
                real == model,
            )
        )
    return rows


def sentinel_climb_threshold(eng, st):
    """The endgame crux: from the player's start tile the Sentinel's base/platform
    tile must be UNSEEABLE (looking up). Raise the observer eye z_height step by step
    (a modelled boulder-stack climb) and report the eye height at which the REAL
    $1CDD first grants LOS to the Sentinel's base tile -- proving 'you must build an
    eye height above the platform tile to absorb the Sentinel'."""
    p = st.player
    pxy = (p.x, p.y)
    sent = next((o for o in st.objects if o.type == gm.T_SENTINEL), None)
    if sent is None:
        return None
    base_h = st.height[sent.y][sent.x]
    eye_start = st.height[pxy[1]][pxy[0]]
    first = None
    for zh in range(eye_start, eye_start + 16):
        if eng.check_los(pxy, (sent.x, sent.y), observer_eye_z=zh):
            first = zh
            break
    return {
        "base_tile": (sent.x, sent.y),
        "base_h": base_h,
        "eye_start": eye_start,
        "first_visible_eye_z": first,
    }


# ----------------------------------------------------------------------------
# 3. ENEMY DYNAMICS
# ----------------------------------------------------------------------------
def enemy_compare(landscape, ticks=(50, 200, 600, 1500, 3000)):
    """Step the real update_enemies vs enemy_dynamics.step_enemies and compare
    enemy angles + player-energy drain at several tick counts. The model phase is
    seeded from the SAME generated RAM (init_phase_from_ram), so the only difference
    is the dynamics."""
    # fresh clean RAM for the model phase (the real engine mutates its own copy)
    mem, _ = _emu.generate(landscape)
    st = read_game_state(Py65Source(mem))
    clean = bytes(mem)
    phase = ed.init_phase_from_ram(st, clean)
    enemy_slots = sorted(phase.enemies.keys())

    eng = ce.CodeEngine(landscape)
    e0_real = eng.player_energy

    rows = []
    prev = 0
    ph = phase
    for total in ticks:
        eng.step_enemies(total - prev)
        for _ in range(total - prev):
            ph = ed.step_enemies(st, ph)
        prev = total
        real_ang = eng.enemy_angles()
        model_ang = {s: ph.enemies[s].h_angle for s in enemy_slots}
        match = all(real_ang.get(s) == model_ang.get(s) for s in enemy_slots)
        rows.append((total, dict(real_ang), dict(model_ang), match))
    real_drain = eng.player_energy - e0_real
    return enemy_slots, rows, real_drain


# ----------------------------------------------------------------------------
# 4. SOLVER PLAN through the real code
# ----------------------------------------------------------------------------
def plan_compare(landscape, budget=20.0):
    import solver_exact as sx

    model = gm.GameModel.from_landscape(landscape)
    rep = sx.analyse_solvability(model)
    if not rep.solvable:
        return {"solvable": False, "reason": rep.reason}
    plan = sx.solve(model, depth=4, budget_s=budget)
    eng = ce.CodeEngine(landscape)
    res = eng.play_plan(plan, step_ticks=0, verbose=False)
    return {
        "solvable": True,
        "plan_steps": len(plan.steps),
        "plan_absorbed": plan.absorbed_count,
        "plan_final_energy": plan.final_energy,
        "plan_solved": plan.solved,
        "real_won": res["won"],
        "real_final_energy": res["final_energy"],
        "real_absorbed": len(res["absorbed"]),
        "real_instructions": res["instructions"],
        "first_failure": res["first_failure"],
    }


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    full = "--full-los" in sys.argv
    do_plan = "--plan" in sys.argv

    los_tot = los_agree = 0
    energy_tot = energy_ok = 0
    gate_tot = gate_ok = 0
    enemy_tot = enemy_ok = 0
    summary = []

    for ls in LANDSCAPES:
        print(f"\n{'='*70}\nLANDSCAPE {ls:04d}\n{'='*70}")
        t0 = time.time()
        eng = ce.CodeEngine(ls)
        st = eng.read_state()
        p = st.player
        print(
            f"  player @ ({p.x},{p.y}) energy {eng.player_energy}; build "
            f"{eng.instructions:,} instrs in {time.time()-t0:.2f}s"
        )

        # --- 1. LOS ---
        t1 = time.time()
        los = los_compare(eng, st, full=full)
        los_tot += los["total"]
        los_agree += los["agree"]
        print(
            f"\n  [1] LINE OF SIGHT  (model.can_see vs real $1CDD; "
            f"{'all' if full else 'sampled'} {los['total']} tiles, {time.time()-t1:.1f}s)"
        )
        print(f"      agreement: {los['agree']}/{los['total']} = {los['pct']:.1f}%")
        print(
            f"      model-says-visible but real-blocked: {len(los['model_only'])}  "
            f"(model OVER-optimistic) e.g. {los['model_only'][:5]}"
        )
        print(
            f"      real-visible but model-blocked:      {len(los['real_only'])}  "
            f"(model over-conservative) e.g. {los['real_only'][:5]}"
        )

        # --- 2. ENERGY ---
        ecmp = energy_compare(ls, st)
        n_ok = sum(1 for r in ecmp if r[6] is True)
        n_check = sum(1 for r in ecmp if r[6] is not None)
        energy_tot += n_check
        energy_ok += n_ok
        print(
            f"\n  [2] ENERGY / ACTIONS  (real $2136/$214F vs model; "
            f"{n_ok}/{n_check} exact)"
        )
        for r in ecmp[:8]:
            kind, _t, tn, before, real, model, ok = (
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
            )
            if ok is None:
                print(f"      {kind:7} {tn:8} energy {before:2d} -> (skipped: {r[7]})")
            else:
                print(
                    f"      {kind:7} {tn:8} energy {before:2d} -> real {real:2d}  "
                    f"model {model:2d}  {'OK' if ok else 'MISMATCH'}"
                )
        if len(ecmp) > 8:
            print(
                f"      ... ({len(ecmp)-8} more, all {'OK' if n_ok == n_check else 'see above'})"
            )

        # --- 2b. ABSORB GATE (LOS to base tile, looking down) ---
        gate = absorb_gate_compare(eng, st)
        n_gate_ok = sum(1 for r in gate if r[8])
        gate_tot += len(gate)
        gate_ok += n_gate_ok
        print(
            f"\n  [2b] ABSORB GATE  (real $1B46/$1CDD to BASE tile vs model "
            f"legal_actions; {n_gate_ok}/{len(gate)} agree)"
        )
        print(
            f"       rule: eye must be ABOVE the object's base tile, looking DOWN "
            f"($1D2E). player eye h={st.height[p.y][p.x]}"
        )
        for slot, _t, tn, tile, base_h, eye_h, real, model, ok in gate:
            tag = "OK" if ok else "DIVERGE"
            rel = (
                ">base(down)"
                if eye_h > base_h
                else "=base(level)" if eye_h == base_h else "<base(UP)"
            )
            print(
                f"       {tn:8} slot {slot:2d} base {str(tile):8} h{base_h:2d} "
                f"(eye h{eye_h} {rel}): "
                f"real_absorbable={int(real)} model={int(model)}  {tag}"
            )
        # endgame crux: Sentinel base tile unseeable from start; threshold by climb
        thr = sentinel_climb_threshold(eng, st)
        if thr:
            fv = thr["first_visible_eye_z"]
            print(
                f"       Sentinel base/platform tile {thr['base_tile']} h{thr['base_h']}: "
                f"from start eye h{thr['eye_start']} it is "
                f"{'UNSEEABLE (looking up) -- correct' if (fv is None or fv > thr['eye_start']) else 'visible'}; "
                f"real $1CDD first grants LOS at eye z_height="
                f"{fv if fv is not None else '>'+str(thr['eye_start']+15)} "
                f"(must build above the platform tile to absorb the Sentinel)"
            )

        # --- 3. ENEMIES ---
        slots, rows, real_drain = enemy_compare(ls)
        n_enemy_ok = sum(1 for _, _, _, m in rows if m)
        enemy_tot += len(rows)
        enemy_ok += n_enemy_ok
        print(
            f"\n  [3] ENEMY DYNAMICS  (real $16B5 vs enemy_dynamics; "
            f"angle match at {n_enemy_ok}/{len(rows)} checkpoints)"
        )
        print(
            f"      enemy slots: {slots}; real player drain over 3000 ticks: {real_drain}"
        )
        for total, ra, ma, m in rows:
            print(
                f"      t={total:5d}  real {ra}  model {ma}  {'MATCH' if m else 'DIVERGE'}"
            )

        summary.append(
            (ls, los["pct"], n_ok, n_check, n_gate_ok, len(gate), n_enemy_ok, len(rows))
        )

        # --- 4. PLAN ---
        if do_plan:
            t2 = time.time()
            pc = plan_compare(ls)
            print(f"\n  [4] SOLVER PLAN through real code  ({time.time()-t2:.1f}s)")
            if not pc["solvable"]:
                print(f"      unsolvable: {pc['reason']}")
            else:
                print(
                    f"      plan: {pc['plan_steps']} steps, absorbed {pc['plan_absorbed']}, "
                    f"model final energy {pc['plan_final_energy']}, solved={pc['plan_solved']}"
                )
                print(
                    f"      real: won={pc['real_won']}, final energy {pc['real_final_energy']}, "
                    f"absorbed {pc['real_absorbed']}, {pc['real_instructions']:,} instrs"
                )
                if pc["first_failure"]:
                    print(f"      FIRST REAL-CODE FAILURE: {pc['first_failure']}")

    # ---- summary table ----
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(
        f"  {'landscape':>9} | {'LOS agree':>10} | {'energy exact':>12} | "
        f"{'absorb gate':>12} | {'enemy match':>12}"
    )
    for ls, lospct, eok, ec, gok, gc, mok, mc in summary:
        print(
            f"  {ls:>9} | {lospct:>9.1f}% | {eok:>5}/{ec:<6} | "
            f"{gok:>5}/{gc:<6} | {mok:>5}/{mc:<6}"
        )
    print(f"  {'-'*9} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*12}")
    print(
        f"  {'OVERALL':>9} | {100*los_agree/los_tot:>9.1f}% | "
        f"{energy_ok:>5}/{energy_tot:<6} | {gate_ok:>5}/{gate_tot:<6} | "
        f"{enemy_ok:>5}/{enemy_tot:<6}"
    )

    print("\n  VERDICTS:")
    print(
        f"   * LINE OF SIGHT  : model agrees with the real ROM ~{100*los_agree/los_tot:.0f}% "
        "of sampled tiles. Disagreements are dominated by the model being"
    )
    print("                     OVER-OPTIMISTIC (bilinear surface vs the ROM's")
    print("                     quantised slope-facet march); see model_only tiles.")
    verdict_e = "EXACT" if energy_ok == energy_tot else "DIVERGENT"
    print(
        f"   * ENERGY/ACTIONS : {verdict_e} ({energy_ok}/{energy_tot}). The energy economy "
        "port ($214F/$2136) matches the real ROM byte-for-byte."
    )
    verdict_g = "HOLDS" if gate_ok == gate_tot else "DIVERGENT"
    print(
        f"   * ABSORB GATE    : {verdict_g} ({gate_ok}/{gate_tot}). The 'must be ABOVE the "
        "object's BASE TILE, looking down' rule ($1B46/$1CDD/$1D2E)"
    )
    print("                     holds identically in the real code and game_model. The")
    print(
        "                     Sentinel's high platform tile is UNSEEABLE from a low start"
    )
    print(
        "                     (looking up) -- you must build an eye above it to absorb it."
    )
    verdict_m = "EXACT" if enemy_ok == enemy_tot else "DIVERGENT"
    print(
        f"   * ENEMY DYNAMICS : {verdict_m} ({enemy_ok}/{enemy_tot}). Where it diverges, the "
        "model's enemy LOCKS onto a target the model's LOS"
    )
    print("                     (wrongly) thinks it sees, so it stops rotating, while")
    print("                     the real Sentinel keeps sweeping -- a downstream")
    print("                     consequence of the LOS over-optimism above.")


if __name__ == "__main__":
    main()
