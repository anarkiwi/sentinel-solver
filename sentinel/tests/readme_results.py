"""Render the README results block from the committed result files.

``docs/results/search.json`` maps each suite landscape to the search player's emulated
record and ``docs/results/live.json`` lists the live runs; only the README text between
the results markers is rewritten (``python -m sentinel.tests.readme_results``).
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SEARCH = os.path.join(ROOT, "docs", "results", "search.json")
LIVE = os.path.join(ROOT, "docs", "results", "live.json")
README = os.path.join(ROOT, "README.md")
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"
COMMAND = "python -m driver.emu_suite --out docs/results/search.json --readme"
MODES = {"live": "", "frozen": " (enemies frozen)"}
PLAYERS = ("search", "phase")
RETIRED = "† Played by the phase player, since retired; its media are kept."


def load(path):
    """The JSON document at ``path``."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def live_cell(runs):
    """The live column for one board's runs; a retired player's run is marked."""
    cells = []
    for run in runs:
        text = f"**{run['actions']} actions**{MODES[run['mode']]}"
        if run["energy"] is not None:
            text += f", energy {run['energy']}"
        if run["media"]:
            text = f"[{text}]({run['media']})"
        cells.append(text + ("" if run["player"] == "search" else "†"))
    return "<br>".join(cells) or "—"


def render(search, live):
    """The README block body for the result documents."""
    head = ["landscape", "enemies", "search actions", "search energy", "live"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in sorted(search, key=int):
        rec = search[key]
        runs = [r for r in live if r["landscape"] == rec["landscape"]]
        won = "" if rec["won"] else " (not won)"
        cells = [key, str(rec["enemies"]), f"{rec['actions']}{won}", str(rec["energy"])]
        rows.append("| " + " | ".join(cells + [live_cell(runs)]) + " |")
    text = "\n".join(rows) + "\n"
    if any(r["player"] != "search" for r in live):
        text += f"\n{RETIRED}\n"
    return text + f"\nRegenerate with `{COMMAND}`.\n"


def block(search, live):
    """README text from marker to marker, markers included."""
    return f"{BEGIN}\n{render(search, live)}{END}"


def splice(readme, new):
    """``readme`` with its marked block replaced by ``new``."""
    head, rest = readme.split(BEGIN, 1)
    return head + new + rest.split(END, 1)[1]


def rewrite(path=README):
    """Rewrite the marked block of the README at ``path`` from the committed data."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(splice(text, block(load(SEARCH), load(LIVE))))


if __name__ == "__main__":
    rewrite()
    sys.exit(0)
