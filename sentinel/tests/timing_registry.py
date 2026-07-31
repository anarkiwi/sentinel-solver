"""Provenance registry for timing constants in ``sentinel/`` and ``driver/``.

DERIVED = recomputable from a ROM primitive (evidence: test_timing_derivations.py);
MEASURED = checked against a committed fixture (evidence: test_settle_accuracy.py);
UNVALIDATED = debt, pinned. Discovery parses source with ``ast``; never imports it.
"""

import ast
import pathlib

DERIVED = "DERIVED"
MEASURED = "MEASURED"
UNVALIDATED = "UNVALIDATED"
CLASSES = (DERIVED, MEASURED, UNVALIDATED)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("sentinel", "driver")
EXCLUDE_DIRS = (ROOT / "sentinel" / "tests",)
TEST_SOURCE_GLOBS = ("sentinel/tests/test_*.py", "driver/test_*.py")

# Constant NAME substrings marking a timing/frame-cost quantity.
NAME_PATTERNS = (
    "_FRAMES",
    "_CYCLES",
    "_TICKS",
    "_SECONDS",
    "SIGMA",
    "TIMEOUT",
    "_RU_",
    "_DELAY",
    "_PERIOD",
    "_SCROLL",
    "SETTLE",
    "COOLDOWN",
    "_RATE",
    "_MS",
    "_HZ",
    "REDRAW",
    "STEPS_PER_",
    "DITHER",
    "TUNE",
    "_WAIT",
    "_STALL",
    "_SPAWN",
    "_ARM",
    "_RAMP",
    "_MASK",
)

# Keyword arguments whose numeric default is a timing/budget knob.
KWARG_NAMES = frozenset(
    {
        "timeout",
        "hold",
        "settle",
        "period",
        "delay",
        "max_steps",
        "chunk",
        "passes",
        "attempts",
    }
)

# Comment words asserting that evidence exists.
PROVENANCE_CLAIM_WORDS = ("measured", "validated")


def _plain_number(node):
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _numeric(node):
    """True if ``node`` evaluates to a number or a container of numbers."""
    if isinstance(node, ast.Constant):
        if _plain_number(node):
            return True
        if not isinstance(node.value, str):
            return False
        try:
            float(node.value)
        except ValueError:
            return False
        return True
    if isinstance(node, (ast.Name, ast.Attribute)):
        return True
    if isinstance(node, ast.UnaryOp):
        return _numeric(node.operand)
    if isinstance(node, ast.BinOp):
        return _numeric(node.left) and _numeric(node.right)
    if isinstance(node, ast.Dict):
        return bool(node.values) and all(_numeric(v) for v in node.values)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(_numeric(e) for e in node.elts)
    if isinstance(node, ast.Call):
        # float()/int()/os.environ.get(NAME, "<number>") wrappers.
        return any(_numeric(a) for a in node.args)
    return False


def _source_files():
    for name in SOURCE_DIRS:
        for path in sorted((ROOT / name).rglob("*.py")):
            if path.name.startswith("test_"):
                continue
            if any(d in path.parents for d in EXCLUDE_DIRS):
                continue
            yield path


def _dotted(path):
    return str(path.relative_to(ROOT)).replace("/", ".")[: -len(".py")]


def _comment_index(source):
    """Map line number to (comment text, is_own_line)."""
    out = {}
    for lineno, line in enumerate(source.splitlines(), start=1):
        hint = line.find("#")
        if hint < 0:
            continue
        head = line[:hint]
        if head.count('"') % 2 or head.count("'") % 2:
            continue
        out[lineno] = (line[hint + 1 :].strip(), not head.strip())
    return out


def _provenance_text(comments, lineno):
    """Trailing comment plus the contiguous own-line comment block above it."""
    block = []
    above = lineno - 1
    while above in comments and comments[above][1]:
        block.append(comments[above][0])
        above -= 1
    parts = list(reversed(block))
    if lineno in comments:
        parts.append(comments[lineno][0])
    return " ".join(parts)


def _kwarg_defaults(func):
    args = func.args
    positional = args.posonlyargs + args.args
    paired = list(
        zip(positional[len(positional) - len(args.defaults) :], args.defaults)
    )
    return paired + list(zip(args.kwonlyargs, args.kw_defaults))


def discover():
    """Scan shipped source; return name -> {"module", "lineno", "comment"}."""
    found = {}

    def record(name, module, lineno, comment):
        if name in found:
            raise AssertionError(
                f"duplicate timing constant {name!r} in {found[name]['module']} "
                f"and {module}; registry keys must be unique"
            )
        found[name] = {"module": module, "lineno": lineno, "comment": comment}

    for path in _source_files():
        source = path.read_text()
        tree = ast.parse(source)
        module = _dotted(path)
        comments = _comment_index(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if node.value is None or not _numeric(node.value):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and any(
                    p in target.id for p in NAME_PATTERNS
                ):
                    record(
                        target.id,
                        module,
                        node.lineno,
                        _provenance_text(comments, node.lineno),
                    )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg, default in _kwarg_defaults(node):
                if (
                    arg.arg in KWARG_NAMES
                    and default is not None
                    and _plain_number(default)
                ):
                    record(
                        f"{node.name}.{arg.arg}",
                        module,
                        node.lineno,
                        _provenance_text(comments, node.lineno),
                    )
    return found


def entry(module, provenance, note, evidence=None):
    """Build a registry value."""
    return {"module": module, "class": provenance, "evidence": evidence, "note": note}


def _u(module, note):
    return entry(module, UNVALIDATED, note)


_PRIMITIVE = "test_derived_constant_matches_primitive"
_SETTLE_FIT = "test_create_settle_prediction_is_accurate"
_PAN_FIT = "test_pan_notch_cost_matches_the_measured_plot"


def _d(module, note):
    return entry(module, DERIVED, note, _PRIMITIVE)


_AC = "sentinel.actioncost"
_PB = "sentinel.playerbase"
_PR = "sentinel.projector"
_PN = "sentinel.pancost"
_EN = "sentinel.enemies"
_ENJ = "sentinel.enemies_jit"
_PC = "sentinel.passcost"
_BL = "sentinel.badline"
_MM = "sentinel.memmap"
_LOS = "sentinel.los"
_KBD = "driver.kbd_aim"
_CORE = "driver.core"
_TICK_EVIDENCE = "test_the_cooldown_tick_prices_every_live_130c_sample"
_BODY_ORACLE = "test_the_body_cost_model_matches_the_roms_own_16e6_cycle_count"


def _tick(note):
    """A counted $130C branch, checked against the live cycle-exact samples."""
    return entry(_PC, DERIVED, note, _TICK_EVIDENCE)


_ROW = "ray-march/sweep iteration cap; unmeasured"
_RELOAD = "ROM cooldown reload value; no derivation test"
_REDRAW = (
    "$1F9F update_object_on_screen around the $209B screen-span query, counted "
    "instruction by instruction against the ROM's own $209B/$1F9F"
)
_GUARD = "wall-clock guard; unmeasured"

REGISTRY = {
    "FRAME_TICKS": _u(_AC, "unit scalar; no test pins it to a ROM primitive"),
    "DITHER_FRAMES": _d(_AC, "977904 dither cycles / projector.FRAME_CYCLES"),
    "VIEWPOINT_REPLOT_FRAMES": _d(_AC, "TUNE_TRANSFER_FRAMES + SETTLE_FIXED_FRAMES"),
    "POST_ACTION_REPLOT_FRAMES": entry(
        _AC,
        MEASURED,
        "validated only inside the create settle sum vs frozen_ls42_audit.json (<5 f)",
        _SETTLE_FIT,
    ),
    "SETTLE": entry(
        _AC,
        MEASURED,
        "create settle within 5 f of frozen_ls42_audit.json; absorb bias is xfailed",
        _SETTLE_FIT,
    ),
    "_CLEAR_CYCLES_H": _d(_PN, "$3912 store-loop cycle count"),
    "_CLEAR_CYCLES_V": _d(_PN, "$38AD store-loop cycle count"),
    "CLEAR_FRAMES": entry(
        _PN,
        MEASURED,
        "within 1 f of the py65-measured clear subtree in golden_pan_cost.json",
        _PAN_FIT,
    ),
    "H_SCROLL": _u(_PB, "pan scroll step; no derivation test"),
    "V_SCROLL": _u(_PB, "pan scroll step; no derivation test"),
    "TOGGLE_FRAMES": entry(
        _PB,
        MEASURED,
        "inside the live_aim_subframes.json measured toggle range; envelope is loose "
        "(min <= 12 <= max), not an error bound",
        "test_charged_toggle_matches_the_measured_pair",
    ),
    "ROT_PERIOD_FRAMES": _d(_PB, "ROTATION_COOLDOWN_RELOAD x UNIT_FRAMES ($1813)"),
    "MEANIE_SPAWN_FRAMES": _d(_PB, "UPDATE_COOLDOWN_MEANIE_MADE x UNIT_FRAMES ($1869)"),
    "TAP_FRAMES": _u(_PB, "key tap hold; no derivation test"),
    "UTURN_FRAMES": entry(
        _PB,
        MEASURED,
        "pooled live n=9 (live_ls42_hops.json p1 + live_ls335_uturns.json), mean "
        "76.6, samples 33..180; a central value over a wide spread, not a bound, "
        "and not yet derived from the tap_action scan/settle structure",
        "test_uturn_is_charged_as_an_action_tap_not_a_keystroke",
    ),
    "UNIT_FRAMES": _d(_PB, "3 x 256 / COOLDOWN_BRESENHAM_STEP gate+Bresenham divider"),
    "CURSOR_RAMP": _d(_PB, "popcount of the $11E0 CURSOR_REPEAT_MASK"),
    "CURSOR_REPEAT_MASK": _u(_PB, "cursor repeat mask; no derivation test"),
    "HOP_FRAMES": entry(
        _PB,
        MEASURED,
        "under both live ls42 hops (745, 879 f) and within 25%; its claimed SETTLE "
        "composition still does not add up (separately xfailed)",
        "test_hop_frames_brackets_the_measured_hops",
    ),
    "SAFE_FRAMES": _u(_PB, "post-action safety margin; unmeasured"),
    "WAIT_FRAMES": _u(_PB, "idle wait quantum; unmeasured"),
    "DRAIN_DELAY": _d(_PB, "DRAINING_COOLDOWN_RELOAD x UNIT_FRAMES ($1835)"),
    "REVOLUTION_FRAMES": _u(
        _PB,
        "14 x ROT_PERIOD_FRAMES covers a 256-unit turn at the +-20 $1813 step, but "
        "a stalled enemy skips its rotate and sweeps as little as 80 units in one "
        "period, so the calendar it bounds is optimistic (strict xfail: "
        "test_revolution_frames_is_a_full_cone_revolution)",
    ),
    "MEANIE_ARM_FRAMES": _d(_PB, "$171B half-turn x $173A rounds x UNIT_FRAMES"),
    "FRAME_CYCLES": _d(_PR, "PAL frame cycle count 19656"),
    "PAL_FRAME_CYCLES": _d(_PC, "PAL 6569: 312 raster lines x 63 cycles"),
    "LINE_CYCLES": entry(
        _BL,
        DERIVED,
        "PAL_FRAME_CYCLES / 312; the badline windows sit 8 lines apart on it",
        "test_the_badline_window_is_the_25_lines_of_a_pal_frame",
    ),
    "WRITE_CYCLES": entry(
        _BL,
        DERIVED,
        "the NMOS 6510's own write cycles per opcode; a run is at most two, which is "
        "why a badline never steals 40, and 4824 live badlines are each BADLINE_STEAL "
        "less the run at their BA window",
        "test_every_live_badline_steal_is_the_derived_one",
    ),
    "IRQ_CYCLES": entry(
        _PC,
        MEASURED,
        "BADLINE_FRAME + the four short raster interrupts + IRQ_BODY. The handler is "
        "COUNTED off the image (KERNAL banked out, $FFC2/$FFC5 are the game's own RAM, "
        "so py65 walks it); only the VIC-II DMA steal is hardware, and BADLINE_FRAME is "
        "the frozen-clock $1289 rate on three boards to under a pass in 50000 "
        "(fixtures/live_pass_cycles.json frozen_idle_rate)",
        "test_the_frozen_frame_budget_reproduces_the_live_idle_pass_count",
    ),
    "FOREGROUND_CYCLES": _d(_PC, "PAL_FRAME_CYCLES - IRQ_CYCLES"),
    "REDRAW_CALL": _d(_PC, _REDRAW),
    "REDRAW_NONE": _d(_PC, _REDRAW),
    "REDRAW_PLOT_ENTRY": entry(
        _PC,
        DERIVED,
        "$1FA2 BCS not taken. Past it $1F9F re-points the camera at the object and "
        "replots its <=20-column strip through $1FFC JSR $2625, which "
        "projector.strip_replot_frames prices at that shifted camera -- exactly "
        "under RENDER_COST_BACKEND=py65, else through the render_cost proxy",
        "test_the_object_screen_span_is_exact_against_the_roms_own_209b",
    ),
    "_REDRAW_CALL": _d(_ENJ, "jit alias of passcost.REDRAW_CALL"),
    "_REDRAW_NONE": _d(_ENJ, "jit alias of passcost.REDRAW_NONE"),
    "_REDRAW_PLOT_ENTRY": _d(_ENJ, "jit alias of passcost.REDRAW_PLOT_ENTRY"),
    "PARTIAL_ARM": entry(
        _PC,
        DERIVED,
        "$17D5..$17DE: the JSR $1973 that re-arms a meanie hunt on a head-only "
        "player, counted instruction by instruction against the per-round $16E6 oracle",
        _BODY_ORACLE,
    ),
    "TARGET_WAIT": entry(
        _PC,
        DERIVED,
        "$1833/$183D: target_object's exit while the draining cooldown still counts "
        "down, counted against the per-round $16E6 oracle",
        _BODY_ORACLE,
    ),
    "TUNE_ROTATE": entry(
        _PC,
        DERIVED,
        "$3470 ends in JMP $FFF1, which the image carries as JMP $8D81 in the game's "
        "own RAM, so the tune is counted on the image; 323 also matches the live "
        "rotation measurement in fixtures/live_pass_cycles.json",
        "test_the_tune_cost_is_counted_off_the_image",
    ),
    "TUNE_DRAIN": entry(
        _PC,
        DERIVED,
        "$1A1D LDA #$05: the drain's tune walks a longer $ACxx descriptor than the "
        "rotation's, counted on the image the same way",
        "test_the_tune_cost_is_counted_off_the_image",
    ),
    "_TUNE_DRAIN": _d(_ENJ, "jit alias of passcost.TUNE_DRAIN"),
    "_PARTIAL_ARM": _d(_ENJ, "jit alias of passcost.PARTIAL_ARM"),
    "_TARGET_WAIT": _d(_ENJ, "jit alias of passcost.TARGET_WAIT"),
    "COOLDOWN_TICK_NO_CARRY": _tick("$130C to the $1315 BCC and RTS"),
    "COOLDOWN_TICK_GATE": _tick("+ the $1317 read and the $1331 $0C50 decrement"),
    "COOLDOWN_TICK_WALK": _tick("$131C entry + the $132B reload + RTS"),
    "COOLDOWN_TICK_BYTE_STICK": _tick("$131E LDA/CMP/BCC + $1328 DEX/BPL"),
    "COOLDOWN_TICK_BYTE_DEC": _tick("+ the $1325 DEC, less the taken BCC"),
    "COOLDOWN_TICK_LAST": _tick("$0C20 leaves by the $1329 BPL not taken, 1 short"),
    "_COOLDOWN_TICK_LAST": _d(_ENJ, "jit alias of passcost.COOLDOWN_TICK_LAST"),
    "_COOLDOWN_TICK_NO_CARRY": _d(_ENJ, "jit alias of passcost.COOLDOWN_TICK_NO_CARRY"),
    "_COOLDOWN_TICK_GATE": _d(_ENJ, "jit alias of passcost.COOLDOWN_TICK_GATE"),
    "_COOLDOWN_TICK_WALK": _d(_ENJ, "jit alias of passcost.COOLDOWN_TICK_WALK"),
    "_COOLDOWN_TICK_BYTE_STICK": _d(_ENJ, "jit alias of COOLDOWN_TICK_BYTE_STICK"),
    "_COOLDOWN_TICK_BYTE_DEC": _d(_ENJ, "jit alias of passcost.COOLDOWN_TICK_BYTE_DEC"),
    "_FOREGROUND_CYCLES": _d(_ENJ, "jit alias of passcost.FOREGROUND_CYCLES"),
    "BASE_CYCLES": _u(_PR, "plot_world base cycles; no fixture"),
    "SETTLE_FIXED_FRAMES": _u(_PR, "fixed settle base; no fixture"),
    "TUNE_TRANSFER_FRAMES": _u(_PR, "transfer tune wait; unmeasured"),
    "UPDATE_COOLDOWN_SCAN": _u(_EN, _RELOAD),
    "UPDATE_COOLDOWN_DRAIN": _u(_EN, _RELOAD),
    "UPDATE_COOLDOWN_MEANIE_ROTATE": _u(_EN, _RELOAD),
    "UPDATE_COOLDOWN_MEANIE_MADE": _u(_EN, _RELOAD),
    "ROTATION_COOLDOWN_RELOAD": _u(_EN, _RELOAD),
    "DRAINING_COOLDOWN_RELOAD": _u(_EN, _RELOAD),
    "COOLDOWN_STICK": _u(_EN, "cooldown stick threshold; no derivation test"),
    "_COOLDOWN_STICK": _d(_ENJ, "jit alias of enemies.COOLDOWN_STICK"),
    "_ENERGY_MASK": _d(_ENJ, "jit alias of memmap.ENERGY_MASK; a bit mask"),
    "ENEMIES_DRAINING_COOLDOWN": _u(_MM, "ROM address, not a duration"),
    "ENEMIES_ROTATION_COOLDOWN": _u(_MM, "ROM address, not a duration"),
    "ENEMIES_UPDATE_COOLDOWN": _u(_MM, "ROM address, not a duration"),
    "COOLDOWN_GATE": _u(_MM, "ROM address of the 1-in-3 gate"),
    "COOLDOWN_BRESENHAM": _u(_MM, "ROM address of the Bresenham accumulator"),
    "COOLDOWN_BRESENHAM_STEP": _u(_MM, "ROM Bresenham step; no derivation test"),
    "ENERGY_MASK": _u(_MM, "bit mask, not a duration"),
    "_MASK_TABLE": _u("sentinel.landscape", "bit mask table, not a duration"),
    "_march_python.max_steps": _u(_LOS, _ROW),
    "_march_jit.max_steps": _u(_LOS, _ROW),
    "check_for_line_of_sight_to_tile.max_steps": _u(_LOS, _ROW),
    "aim_target.max_steps": _u(_LOS, _ROW),
    "landable_views.max_steps": _u(_LOS, _ROW),
    "landable_sweep_with_centres.max_steps": _u(_LOS, _ROW),
    "landable_view.max_steps": _u(_LOS, _ROW),
    "landable_view_targeted.max_steps": _u(_LOS, _ROW),
    "can_see_object.max_steps": _u("sentinel.relative", _ROW),
    "_PAN_STALL_FRAMES": _d(_KBD, "playerbase.H_SCROLL + playerbase.V_SCROLL"),
    "_PAN_MAX_FRAMES": _u(
        _KBD, "below a full 464-frame pan; xfail in test_pan_max_covers_full_pan"
    ),
    "_RU_COMMIT": _u(_KBD, "ROM read-under trap byte; no derivation test"),
    "_SCAN_WAIT_PASSES": _u(_KBD, "scan settle passes; unmeasured"),
    "_run_to_scan.timeout": _u(_KBD, _GUARD),
    "tap.hold": _u(_CORE, "keypress hold frames; unmeasured"),
    "tap.settle": _u(_CORE, "post-keypress settle frames; unmeasured"),
    "load_snapshot.timeout": _u("driver.boot", _GUARD),
}

# Pinned debt as a count; lower it when a constant is validated away, never raise it.
UNVALIDATED_CEILING = 42

# Pinned constants whose source comment advertises evidence that does not exist.
KNOWN_FALSE_PROVENANCE_COMMENTS = frozenset()
