"""Which instruction a charged cost term's cycle offset names, and what it writes.

A ``passcost`` term is a contiguous ROM run counted off the disassembly, so an offset
into one identifies an instruction, and with it the write cycles a VIC-II BA window
catches there (:mod:`sentinel.badline`).  The 6510 tables are jennings' own.
"""

from jennings.devices.mpu6502 import MPU

from sentinel import passcost

MODE = tuple(mode for _, mode in MPU.disassemble)
MNEMONIC = tuple(name for name, _ in MPU.disassemble)
DEC_ABS_CYCLES = 6  # jennings' table reads 3; the machine takes 6, as $0E/$2E/$4E do
OP_CYCLES = tuple(
    DEC_ABS_CYCLES if op == 0xCE else c for op, c in enumerate(MPU.cycletime)
)
_MODE_LENGTH = {
    "imp": 1,
    "acc": 1,
    "imm": 2,
    "zpg": 2,
    "zpx": 2,
    "zpy": 2,
    "rel": 2,
    "inx": 2,
    "iny": 2,
    "abs": 3,
    "abx": 3,
    "aby": 3,
    "ind": 3,
}
OP_LENGTH = tuple(_MODE_LENGTH[mode] for mode in MODE)

_RMW = frozenset(
    ("ASL", "LSR", "ROL", "ROR", "INC", "DEC", "SLO", "RLA", "SRE", "RRA", "DCP", "ISC")
)
_STORE = frozenset(("STA", "STX", "STY", "SAX", "SHA", "SHX", "SHY", "SHS"))
_NO_MEMORY = frozenset(("imp", "acc", "imm", "rel", "ind"))
_BRANCH = frozenset(("BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ"))


def _write_cycles(op):
    """Cycle indices, from the opcode fetch, at which opcode ``op`` drives a write."""
    name, mode, cycles = MNEMONIC[op], MODE[op], OP_CYCLES[op]
    if name in ("PHA", "PHP"):
        return (cycles - 1,)
    if name == "JSR":
        return (3, 4)  # the two return-address pushes, before the target fetch
    if mode in _NO_MEMORY:
        return ()
    if name in _STORE:
        return (cycles - 1,)
    if name in _RMW:  # NMOS: the dummy write of the old byte, then the new one
        return (cycles - 2, cycles - 1)
    return ()


OP_WRITE_CYCLES = tuple(_write_cycles(op) for op in range(256))


def write_run(op, index):
    """Consecutive write cycles of ``op`` starting at cycle ``index`` from its fetch."""
    cycles = OP_WRITE_CYCLES[op]
    run = 0
    while index + run in cycles:
        run += 1
    return run


def walk(image, pc, cycles, branches=()):
    """``(offset, pc, op)`` per instruction of the run at ``pc``, out to ``cycles``.

    ``branches`` is consumed one flag per conditional branch -- the decision the cost
    term charging this run already made; exhausted, a branch falls through.
    """
    out = []
    taken = iter(branches)
    stack = []
    offset = 0
    while offset < cycles:
        op = image[pc]
        out.append((offset, pc, op))
        name = MNEMONIC[op]
        offset += OP_CYCLES[op]
        nxt = (pc + OP_LENGTH[op]) & 0xFFFF
        if name in _BRANCH:
            if next(taken, False):
                dest = image[(pc + 1) & 0xFFFF]
                dest = (nxt + (dest - 256 if dest > 127 else dest)) & 0xFFFF
                offset += 1 + (1 if (dest >> 8) != (nxt >> 8) else 0)
                nxt = dest
        elif name == "JMP":
            nxt = image[(pc + 1) & 0xFFFF] | (image[(pc + 2) & 0xFFFF] << 8)
            if MODE[op] == "ind":
                nxt = image[nxt] | (image[(nxt + 1) & 0xFFFF] << 8)
        elif name == "JSR":
            stack.append(nxt)
            nxt = image[(pc + 1) & 0xFFFF] | (image[(pc + 2) & 0xFFFF] << 8)
        elif name == "RTS":
            if not stack:
                break
            nxt = stack.pop()
        elif name in ("RTI", "JAM", "BRK"):
            break
        pc = nxt
    return tuple(out)


def instruction_at(image, pc, offset, branches=()):
    """``(offset, pc, op)`` of the instruction covering cycle ``offset`` of the run."""
    run = walk(image, pc, offset + 1, branches)
    return run[-1] if run else None


def steal_at(image, pc, offset, branches=()):
    """The badline steal of a BA window ``offset`` cycles into the run at ``pc``."""
    run = walk(image, pc, offset + 1, branches)
    if not run:
        return None
    return passcost.BADLINE_STEAL - write_run(run[-1][2], offset - run[-1][0])
