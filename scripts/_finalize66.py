import sys, json, time

sys.path.insert(0, "scripts")
import climb_greedy
import climb_greedy_validate as V

t0 = time.time()
# 1) native keyboard-lattice plan (A1/A3/A5/A6/B): every emitted view is on the
#    keyboard-reachable lattice (h%8==0, v%4==1, v in the [$CD..$35] pan band).
g = climb_greedy.plan_greedy(66, verbose=False, toward_plat=True, near_plat_radius=2)
print(
    "native: won", g.native_won, "steps", len(g.steps), "energy", g.energy, flush=True
)

# 2) honest ROM validation with blocklist-replan (A4) + enemy stress (A6): a step that
#    can only pass via the non-keyboard-reproducible poke rescue is FAILED and replanned.
won, steps = V.solve(66, verbose=True, toward_plat=True, near_plat_radius=2)
print("ROM $0CDE won:", won, "in", round(time.time() - t0, 1), "s", flush=True)

out_steps = steps if won else g.steps  # emit the ROM-clean plan, else the native one
out = {
    "landscape": 66,
    "won": bool(won),
    "native_won": bool(g.native_won),
    "toward_plat": True,
    "final_player": list(g.player_xy()),
    "eye": g.eye,
    "energy": g.energy,
    "steps": out_steps,
}
json.dump(out, open("out/kbd_greedy_0066.json", "w"), indent=0)
print(
    "wrote out/kbd_greedy_0066.json",
    "(ROM-VALIDATED WIN)" if won else "(native-only; strict ROM keyboard-gate not won)",
    flush=True,
)
