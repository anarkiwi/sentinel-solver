import sys, json, time

sys.path.insert(0, "scripts")
import climb_greedy
import climb_greedy_validate as V

t0 = time.time()
g = climb_greedy.plan_greedy(0, verbose=False, toward_plat=True, near_plat_radius=2)
print(
    "native: won", g.native_won, "steps", len(g.steps), "energy", g.energy, flush=True
)

won, steps = V.solve(0, verbose=True, toward_plat=True, near_plat_radius=2)
print("ROM $0CDE won:", won, "in", round(time.time() - t0, 1), "s", flush=True)

out_steps = steps if won else g.steps
out = {
    "landscape": 0,
    "won": bool(won),
    "native_won": bool(g.native_won),
    "toward_plat": True,
    "final_player": list(g.player_xy()),
    "eye": g.eye,
    "energy": g.energy,
    "steps": out_steps,
}
json.dump(out, open("out/kbd_greedy_0000.json", "w"), indent=0)
print(
    "wrote out/kbd_greedy_0000.json",
    "(ROM-VALIDATED WIN)" if won else "(native-only; strict ROM keyboard-gate not won)",
    flush=True,
)
