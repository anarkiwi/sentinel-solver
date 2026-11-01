"""Render the README results block from the committed result files:
``python -m sentinel.tests.readme_results`` (or ``trace_suite --readme``).

``docs/results/offline.json`` maps each suite landscape to its offline record and
``docs/results/live.json`` lists the live runs; only the text between the
``<!-- results:begin -->`` and ``<!-- results:end -->`` markers in README.md is rewritten.
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OFFLINE = os.path.join(ROOT, "docs", "results", "offline.json")
LIVE = os.path.join(ROOT, "docs", "results", "live.json")
README = os.path.join(ROOT, "README.md")
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"
COMMAND = (
    "python -m sentinel.tests.trace_suite --out docs/results/offline.json --readme"
)
MODES = {"live": "", "frozen": " (enemies frozen)"}


def load(path):
    """The JSON document at ``path``."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def stale(run, offline):
    """Whether ``run`` was played with a model other than the current offline one."""
    board = offline.get(str(run["landscape"]))
    return run["digest"] is None or board is None or run["digest"] != board["digest"]


def live_cell(runs, offline):
    """The live column for one board's runs."""
    cells = []
    for run in runs:
        text = f"**{run['actions']} actions**{MODES[run['mode']]}"
        if run["energy"] is not None:
            text += f", energy {run['energy']}"
        if run["media"]:
            text = f"[{text}]({run['media']})"
        cells.append(text + ("†" if stale(run, offline) else ""))
    return "<br>".join(cells) or "—"


def render(offline, live):
    """The README block body for the two result documents."""
    rows = [
        "| landscape | enemies | offline actions | offline energy | live |",
        "|---|---|---|---|---|",
    ]
    for key in sorted(offline, key=int):
        rec = offline[key]
        runs = [r for r in live if r["landscape"] == rec["landscape"]]
        won = "" if rec["won"] else " (not won)"
        rows.append(
            f"| {key} | {rec['enemies']} | {rec['actions']}{won} | {rec['energy']} "
            f"| {live_cell(runs, offline)} |"
        )
    text = "\n".join(rows) + "\n"
    if any(stale(r, offline) for r in live):
        text += (
            "\n† Live run recorded against an older model: its trace digest is unrecorded "
            "or differs from the current offline trace for that board.\n"
        )
    return text + f"\nRegenerate with `{COMMAND}`.\n"


def block(offline, live):
    """README text from marker to marker, markers included."""
    return f"{BEGIN}\n{render(offline, live)}{END}"


def splice(readme, new):
    """``readme`` with its marked block replaced by ``new``."""
    head, rest = readme.split(BEGIN, 1)
    return head + new + rest.split(END, 1)[1]


def rewrite(path=README):
    """Rewrite the marked block of the README at ``path`` from the committed data."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(splice(text, block(load(OFFLINE), load(LIVE))))


if __name__ == "__main__":
    rewrite()
    sys.exit(0)
