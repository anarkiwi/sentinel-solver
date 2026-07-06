#!/usr/bin/env python3
"""Generate a landscape with the ROM's OWN routines inside live asid-vice and enter
the real first-person play loop -- bypassing the obfuscated "SECRET ENTRY CODE?" gate.

WHY: the code-check at $14DC-$14F2 computes the jump to play_landscape ($35A4) from
the validation result + objects_flags, so naive patching crashes. Instead we mirror
play_setup ($1A97) exactly, minus the preview/title plotting and minus the code gate:

  $1149 reset_game_state
  $33ED seed_prnd_from_landscape_number (X=lo, Y=hi)  -> also stores $0CFD/$0CFE/$0C52
  $2ACC generate_landscape, STOP at $2B21 (GENERATE_END terrain-build end; the render
        tail desyncs the prnd and changes the tree count -- see _emu.GENERATE_END)
  $1420 set_palette_and_initialise_enemies (Sentinel + sentries + landscape palette)
  $1450 initialise_player_and_trees -- called with $0C71 bit7 CLEAR so its $14AF
        `BIT $0C71; BPL` takes the PREVIEW path (build validation table, RTS to guard)
        instead of the obfuscated leave-to-play jump. This is byte-for-byte the same
        object/prnd path code_engine + _emu use, so the resulting board matches
        Py65Source.from_landscape(ls).
  set $141F = $7F (viewpoint_perspective; REQUIRED for in-play LOS geometry $13FF)
  set $0C71 bit7 (play_game_after_generation -> in-play semantics)
  ensure $0CDE = 0 (player_has_hyperspaced clear; play_landscape entry checks it)
  JMP $35A4 (play_landscape) -- the real interactive loop; VICE has a real VIC-II/IRQ
        so it renders and accepts keys.

Each ROM routine is invoked JSR-style via a tiny `JSR addr ; JMP self` stub planted in
free RAM, exactly like Executor.probe: push nothing, set PC=stub, run_until_pc(jmp_self).
generate's early stop ($2B21) uses run_until_pc on $2B21 directly.

ROM addresses: play_setup $1A97, $14AA gate.
"""

import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# ---- ROM routine addresses ----
R_RESET = 0x1149  # reset_game_state
R_SEED = 0x33ED  # seed_prnd_from_landscape_number (X=lo, Y=hi)
R_GENERATE = 0x2ACC  # generate_landscape
GENERATE_END = 0x2B21  # terrain-build end (stop before preview-render tail)
R_INIT_ENEMIES = (
    0x1420  # set_palette_and_initialise_enemies (Sentinel/sentries+palette)
)
R_INIT_PLAYER = 0x1450  # initialise_player_and_trees
PLAY_LANDSCAPE = 0x35A4  # play_landscape (real first-person loop)

A_PLAY_FLAG = 0x0C71  # play_game_after_generation (bit7)
A_VIEWPOINT = 0x141F  # viewpoint_perspective: 0=preview, $7F=in-play ($13FF)
A_HYPERSPACED = 0x0CDE  # player_has_hyperspaced (bit7) / landscape-complete (bit6)

# scratch RAM for the JSR stub. The tape buffer / free zp-area $033C is unused at the
# title/code screen (it is in _emu LOADED but holds no live play state pre-play). Use a
# region clear of the Executor's later LOS stub ($02A0) so nothing is clobbered.
STUB = 0x0334


# While the Sentinel waits for input at the "LANDSCAPE NUMBER?" / code screen it spins
# in its keyboard-matrix scanner ($8CF9, scan_keyboard_matrix), hit continuously there
# -- a reliable place to STOP the CPU before injecting (verified: live PC samples sit
# in $8CF9-$8D68; run_until_pc($8CF9) halts immediately).
TITLE_HALT = 0x8CF9


def _halt(bm, addr=TITLE_HALT, timeout=8.0):
    """Bring the CPU to a known HALTED state so a subsequent registers_set(PC) sticks.
    `bm.halted()` only disables auto-resume -- it does NOT stop a running CPU, so a PC
    we set would be immediately overwritten by the live game loop. Stop the CPU with an
    EXEC checkpoint at `addr` (an address the current loop executes every iteration)."""
    bm.run_until_pc(addr, timeout=timeout)


def _stub_call(bm, addr, a=0, x=0, y=0, timeout=8.0, stop_pc=None):
    """JSR-call ROM `addr` (A/X/Y set) and run until it returns. Precondition: the CPU
    is ALREADY HALTED (the caller halts once via _halt; after each call the CPU sits at
    the `JMP self` guard, still halted). Plant `JSR addr ; JMP self` at STUB, set PC=STUB
    and the A/X/Y regs (sticks because the CPU is halted -- done with auto-resume off so
    no live frame runs between the pokes), then run_until_pc the JMP-self (the routine's
    RTS lands on it). `stop_pc` halts generate at $2B21 before its preview-render tail.
    """
    jsr = bytes([0x20, addr & 0xFF, (addr >> 8) & 0xFF])
    jmp_self = STUB + len(jsr)
    code = jsr + bytes([0x4C, jmp_self & 0xFF, (jmp_self >> 8) & 0xFF])
    with bm.halted():
        bm.mem_set(STUB, code)
        # SP near top of stack so the JSR has room; regs A/X/Y; FLAGS=$20 (clear D/I).
        bm.registers_set(
            {0: a & 0xFF, 1: x & 0xFF, 2: y & 0xFF, 3: STUB, 4: 0xFD, 5: 0x20}
        )
    target = stop_pc if stop_pc is not None else jmp_self
    bm.run_until_pc(target, timeout=timeout)


# signature bytes of the loaded game in RAM, used to detect that the multi-stage
# tape load has finished (the routines we call are resident). $35A4 holds
# `A5 0B 85` (play_landscape: LDA player_object / STA) once loaded.
SIG_ADDR = 0x35A4
SIG_BYTES = bytes([0xA5, 0x0B, 0x85])


def wait_for_load(bm, log=print, total=80.0, poll=2.0):
    """Poll RAM until the game is resident (SIG_BYTES present at SIG_ADDR).
    The tape load is multi-stage and its timing under warp varies; polling a
    signature is more reliable than a fixed sleep. Returns True if loaded."""
    import time as _t

    deadline = _t.time() + total
    while _t.time() < deadline:
        try:
            if bytes(bm.mem_get(SIG_ADDR, SIG_ADDR + 2)) == SIG_BYTES:
                log(f"  load complete (sig at ${SIG_ADDR:04x})")
                return True
        except Exception:
            pass
        _t.sleep(poll)
    return False


def dump_snapshot(bm, path, save_roms=True, save_disks=False):
    """DUMP (binmon opcode 0x41): save a full VICE snapshot to `path` (container
    side). Body = save_roms(u8) save_disks(u8) name_len(u8) name. save_roms=True so
    UNDUMP restores a self-contained machine. The installed BinMon has no wrapper, so
    we build the body and use bm.call directly (asid-vice monitor_binary.c:1047)."""
    pb = path.encode("ascii")
    body = bytes([1 if save_roms else 0, 1 if save_disks else 0, len(pb)]) + pb
    bm.call(0x41, body, timeout=20.0)


def undump_snapshot(bm, path):
    """UNDUMP (binmon opcode 0x42): restore a VICE snapshot from `path` (container
    side). Body = name_len(u8) name. Returns the restored PC (monitor_binary.c:1071)."""
    pb = path.encode("ascii")
    body = bytes([len(pb)]) + pb
    resp = bm.call(0x42, body, timeout=20.0)
    if len(resp.body) >= 2:
        return resp.body[0] | (resp.body[1] << 8)
    return None


def generate_and_enter(bm, landscape, log=print, settle=1.0):
    """Run the play_setup mirror in live VICE for `landscape`, then JMP into
    play_landscape ($35A4). Returns when the interactive loop has been entered.
    Caller is responsible for having booted the tape far enough that the ROM +
    KERNAL are resident (the title / code screen is fine)."""
    lo, hi = landscape & 0xFF, (landscape >> 8) & 0xFF
    log(f"  gen_enter ls{landscape}: reset/seed/generate/enemies/player ...")
    # HALT the CPU ONCE at the title key-scan loop; thereafter each _stub_call leaves it
    # halted at the JMP-self guard, so no live frame runs between steps. All pokes use
    # auto-resume OFF (bm.halted()) so the halted state is preserved throughout.
    _halt(bm)
    with bm.halted():
        _stub_call(bm, R_RESET)
        _stub_call(bm, R_SEED, x=lo, y=hi)
        # play flag CLEAR while initialise_player_and_trees runs so $14AF takes the
        # preview path (build table, clean RTS) -- not the obfuscated leave-to-play jump.
        bm.mem_set(A_PLAY_FLAG, bytes([bm.mem_get(A_PLAY_FLAG, A_PLAY_FLAG)[0] & 0x7F]))
        _stub_call(bm, R_GENERATE, stop_pc=GENERATE_END, timeout=20.0)
        _stub_call(bm, R_INIT_ENEMIES)
        _stub_call(bm, R_INIT_PLAYER)
        # in-play state, then JMP into the real play loop.
        bm.mem_set(A_VIEWPOINT, bytes([0x7F]))  # $141F = $7F (in-play LOS geometry)
        bm.mem_set(
            A_PLAY_FLAG, bytes([bm.mem_get(A_PLAY_FLAG, A_PLAY_FLAG)[0] | 0x80])
        )  # play semantics
        bm.mem_set(A_HYPERSPACED, bytes([0x00]))  # not hyperspaced / not complete
        log(f"  gen_enter: JMP play_landscape ${PLAY_LANDSCAPE:04x}")
        bm.registers_set({3: PLAY_LANDSCAPE, 4: 0xFD, 5: 0x20})
    bm.exit()  # resume the CPU into the play loop
    time.sleep(settle)
