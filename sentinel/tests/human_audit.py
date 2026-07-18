"""Model-vs-human ground-truth audit over the recorded human-win fixtures.

Per step it reconstructs the PRE-action ``State`` and records the human's ground truth
beside our model's energy/geometry/enemy/exposure/aim/verdict values, flagging every
disagreement.  :func:`audit_fixture` runs any fixture; :func:`main` writes the artifacts.
"""

import json
import math
import os
import types

from sentinel import energy, enemies, los, memmap as mm
from sentinel.playerbase import BasePlayer, FOV_HALF, FOV_MARGIN, HOP_FRAMES
from sentinel.test_human_win_logs import state_from_event, _load, FIXTURES

_FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "human_wins")
_HALF = FOV_HALF + FOV_MARGIN


def _load_truth(name):
    """The replayed enemy-phase ground truth for ``name`` (``<name>_truth.json``,
    written by ``driver.replay_human``), as ``{step_i: {slot: enemy_dict}}`` over
    reproduced steps only, or ``None`` when the fixture is absent."""
    path = os.path.join(_FIX_DIR, name[:-5] + "_truth.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        truth = json.load(fh)
    out = {}
    for s in truth.get("steps", []):
        rep = s.get("replay", {})
        if rep.get("diverged_since") or not rep.get("matched_recording", True):
            continue  # only the clean pre-divergence prefix is on the human's line
        out[s["i"]] = {e["slot"]: e for e in s["enemies"]}
    return out


def _apply_truth(st, phase):
    """Overwrite the reconstructed baseline enemy facing + rotation/drain cooldowns
    with the replayed true values for every enemy slot present in ``phase``."""
    for slot, e in phase.items():
        st.obj_h_angle[slot] = e["h_angle"] & 0xFF
        st.obj_v_angle[slot] = e["v_angle"] & 0xFF
        st.mem[mm.ROTATION_SPEED_TABLE + slot] = e["rot_step"] & 0xFF
        st.mem[mm.ENEMIES_ROTATION_COOLDOWN + slot] = e["rot_cooldown"] & 0xFF
        st.mem[mm.ENEMIES_DRAINING_COOLDOWN + slot] = e["drain_cooldown"] & 0xFF
        st.mem[mm.ENEMIES_UPDATE_COOLDOWN + slot] = e["update_cooldown"] & 0xFF


_NOTES = (
    "Enemy h_angle / rotation cooldowns are NOT recorded in the fixture; every "
    "reconstructed enemy facing is the landscape.generate baseline, not the true "
    "mid-game facing. Exposure/gaze/breach verdicts run against baseline facings. "
    "On a human WIN every placement survived, so any model REJECT/BREACH is a "
    "model false-positive against ground truth.",
    "energy note 'tree_spawn' = a create of a TREE the human paid nothing for: an "
    "enemy-discharge-spawned tree ($1A5D) mis-kept as a player build. 'drain' = the "
    "human lost energy to an enemy drain between actions (real exposure).",
)


def _signed(b):
    return b - 256 if b >= 128 else b


def _rd(x, n=1):
    return None if x is None else (math.inf if x == math.inf else round(x, n))


def _energy_step(ev, nxt):
    """Pure action-cost energy model vs the human's next recorded energy."""
    pre = ev["energy"]
    verb, ot = ev["verb"], ev["otype"]
    cost = {"create": -energy.value(ot), "absorb": energy.value(ot), "transfer": 0}[
        verb
    ]
    model_post = pre + cost
    human_next = None if nxt is None else nxt["energy"]
    if human_next is None:
        note = "terminal"
    elif model_post == human_next:
        note = "ok"
    elif verb == "create" and ot == mm.T_TREE and human_next == pre:
        note = "tree_spawn"
    elif human_next < model_post:
        note = "drain"
    else:
        note = "cost_mismatch"
    return {
        "pre": pre,
        "action_cost": cost,
        "model_post": model_post,
        "human_next": human_next,
        "agree": note in ("ok", "terminal"),
        "note": note,
    }


def _enemies_block(bp, st, exposed):
    """Per-enemy geometry + rotation primitives, with the toward-target exposure."""
    exp = {e: (ah, full) for e, ah, full in exposed}
    out = []
    for e in enemies.enemy_slots(st):
        facing = st.obj_h_angle[e]
        ah, full = exp.get(e, (None, None))
        out.append(
            {
                "slot": int(e),
                "type": mm.TYPES[st.obj_type[e]],
                "tile": list(st.tile_of(e)),
                "h_angle": int(facing),
                "v_angle": int(st.obj_v_angle[e]),
                "rot_step": _signed(st.mem[mm.ROTATION_SPEED_TABLE + e]),
                "rot_cooldown": int(st.mem[mm.ENEMIES_ROTATION_COOLDOWN + e]),
                "drain_cooldown": int(st.mem[mm.ENEMIES_DRAINING_COOLDOWN + e]),
                "target_angle_hi": None if ah is None else int(ah),
                "target_full_sight": full,
                "in_cone_now": None if ah is None else bp._in_cone(ah, facing, _HALF),
            }
        )
    return out


def _audit_step(i, ev, evs, seed, truth=None):
    """One step's full model-vs-human record with a per-step disagreement list.
    With ``truth`` (replayed enemy phase) the reconstructed enemy facings/cooldowns
    are the TRUE mid-game values, else the ``landscape.generate`` baseline."""
    st = state_from_event(ev, seed)
    phase = None if truth is None else truth.get(i)
    if phase:
        _apply_truth(st, phase)
    bp = BasePlayer(types.SimpleNamespace(state=st), audit=True)
    verb, otype, tgt = ev["verb"], ev["otype"], tuple(ev["target"])
    pl = ev["player"]
    own_tile = tgt == (pl["x"], pl["y"])

    exposed_t = bp._exposing_enemies(tgt)
    seen_now = bp._seen_now(exposed_t)
    gaze = bp._gaze_window(tgt, exposed=exposed_t)
    tree_near = bp._tree_near(tgt)
    player_window = bp._player_window()
    sees_target = bp._sees_tile(tgt)

    view = los.landable_view_targeted(st, tgt, st.player, eye_z=pl["z"])
    aim_frames = bp._aim_frames(view) if view else None
    view_matches = None
    if view is not None:
        view_matches = view["h_angle"] == pl["hang"] and view["v_angle"] == pl["vang"]

    gate_allow = None
    if verb in ("create", "transfer") and view is not None:
        need = bp._settle("transfer", view) if verb == "transfer" else HOP_FRAMES
        gate_allow = (not seen_now) and gaze >= aim_frames + need

    fverb = {"create": "boulder" if otype == mm.T_BOULDER else "robot"}.get(verb, verb)
    fire_ok = None
    breaches = []
    if view is not None:
        fire_ok = bp._fire(fverb, tgt, view)
        breaches = [
            {"verb": b[1], "tile": list(b[2]), "seen_by": b[3]} for b in bp.breaches
        ]

    en = _energy_step(ev, evs[i + 1] if i + 1 < len(evs) else None)

    dis = []
    if en["note"] in ("tree_spawn", "drain", "cost_mismatch"):
        dis.append("energy:" + en["note"])
    if view is None and not own_tile:
        dis.append("no_landable_view")
    if gate_allow is False:
        dis.append("gate_reject")
    if fire_ok is False:
        dis.append("fire_fail")
    if breaches:
        dis.append("account_breach")

    return {
        "i": i,
        "verb": verb,
        "otype": otype,
        "otype_name": mm.TYPES[otype],
        "target": list(tgt),
        "human": {
            "tile": [pl["x"], pl["y"]],
            "eye_z": _rd(pl["z"] + pl["zf"] / 256.0, 3),
            "h_angle": pl["hang"],
            "v_angle": pl["vang"],
            "cursor": list(ev["cursor"]),
            "energy": ev["energy"],
            "do_los": ev["do_los"],
        },
        "model_geom": {
            "player_tile": list(st.player_xy()),
            "eye_z": _rd(bp._my_eye(), 3),
            "h_angle": int(st.obj_h_angle[st.player]),
            "v_angle": int(st.obj_v_angle[st.player]),
        },
        "energy": en,
        "enemy_facings_source": "replay_truth" if phase else "generate_baseline",
        "enemies": _enemies_block(bp, st, exposed_t),
        "exposure_target": {
            "n_exposed": len(exposed_t),
            "n_full": sum(1 for _, _, f in exposed_t if f),
            "seen_now": seen_now,
            "gaze_window": _rd(gaze),
            "tree_near": tree_near,
        },
        "exposure_player": {
            "player_window": _rd(player_window),
            "own_tile_transfer": own_tile,
        },
        "aim": {
            "sees_target": sees_target,
            "has_view": view is not None,
            "view": view,
            "aim_frames": _rd(aim_frames),
            "view_matches_human_facing": view_matches,
        },
        "verdict": {"gate_allow": gate_allow, "fire_ok": fire_ok, "breaches": breaches},
        "disagreements": dis,
    }


def _summarise(steps):
    """Aggregate per-dimension agreement and the disagreement step-sets by code."""
    codes = {}
    for s in steps:
        for c in s["disagreements"]:
            codes.setdefault(c.split(":")[0], []).append(s["i"])
    notes = {}
    for s in steps:
        notes.setdefault(s["energy"]["note"], []).append(s["i"])
    return {
        "n_steps": len(steps),
        "energy_model_agree": sum(1 for s in steps if s["energy"]["agree"]),
        "energy_notes": dict(sorted(notes.items())),
        "landable_view_agree": sum(1 for s in steps if s["aim"]["has_view"]),
        "own_tile_transfers": [
            s["i"] for s in steps if s["exposure_player"]["own_tile_transfer"]
        ],
        "view_facing_matches": sum(
            1 for s in steps if s["aim"]["view_matches_human_facing"]
        ),
        "n_steps_with_disagreement": sum(1 for s in steps if s["disagreements"]),
        "disagreement_steps_by_code": {k: sorted(v) for k, v in sorted(codes.items())},
    }


def audit_fixture(name):
    """The full model-vs-human audit dict for one ``human_wins`` fixture."""
    data = _load(name)
    seed = data["landscape"]
    evs = data["events"]
    truth = _load_truth(name)
    steps = [_audit_step(i, ev, evs, seed, truth) for i, ev in enumerate(evs)]
    notes = list(_NOTES)
    if truth:
        notes.insert(
            0,
            f"{len(truth)} steps use TRUE replayed enemy facings/cooldowns "
            "(enemy_facings_source=replay_truth); the rest fall back to baseline.",
        )
    return {
        "fixture": name,
        "seed": seed,
        "entered_code": data["entered_code"],
        "n_events": len(evs),
        "enemy_truth_steps": 0 if truth is None else len(truth),
        "notes": notes,
        "summary": _summarise(steps),
        "steps": steps,
    }


def artifact_path(name):
    return os.path.join(_FIX_DIR, name[:-5] + "_audit.json")


def _default(o):
    if o == math.inf:
        return "inf"
    raise TypeError(repr(o))


def main():
    for name in FIXTURES:
        audit = audit_fixture(name)
        with open(artifact_path(name), "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=1, default=_default)
        s = audit["summary"]
        counts = {k: len(v) for k, v in s["disagreement_steps_by_code"].items()}
        print(
            f"{name}: energy {s['energy_model_agree']}/{s['n_steps']} "
            f"view {s['landable_view_agree']}/{s['n_steps']} "
            f"disagree_steps={s['n_steps_with_disagreement']} codes={counts}"
        )


if __name__ == "__main__":
    main()
