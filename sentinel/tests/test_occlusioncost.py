"""$245B occlusion-raytrace pricing: occlusioncost equals the live 6502 count.

The oracle gate runs the real populate_tile_visibility_bit_table in jennings
from generated boards (play posture) and asserts the model is cycle-exact,
including with the player relocated across the board.
"""

import pytest

from sentinel import landscape, occlusioncost, terrain
from sentinel.game import Game
from sentinel.state import State
from sentinel.tests import oracle

# typed board -> successive relocation targets (flat bare tiles, edges included)
BOARDS = {42: ((0, 0), (17, 21)), 335: ((28, 16), (25, 0)), 9795: ((0, 18), (28, 31))}


def _measure(cpu, mem):
    """One real $245B pass in the play posture; returns its cycle delta."""
    mem[0x0CDE] = 0
    mem[0x0CCE] = 0x80
    mem[0x0CDF] = mem[0x0C73] = 0
    ret = 0xFFF0
    mem[ret] = 0x60
    img = oracle._rom_image()
    mem[0xFFF1:0x10000] = img[0xFFF1:0x10000]
    sp = cpu.sp
    mem[0x0100 + sp] = (ret - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (ret - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = 0x245B
    start = cpu.processorCycles
    for _ in range(6_000_000):
        if cpu.pc == ret:
            break
        cpu.step()
    assert cpu.pc == ret
    return cpu.processorCycles - start


def _relocate(mem, nx, ny):
    """Move the player object to flat bare tile (nx, ny), restoring the old tile."""
    st = State(mem)
    p = st.player
    ox, oy = st.obj_x[p], st.obj_y[p]
    assert terrain.tile_byte(st, ox, oy) == 0xC0 | p
    terrain.set_tile_byte(st, ox, oy, st.obj_z_height[p] << 4)
    b = terrain.tile_byte(st, nx, ny)
    assert b < 0xC0 and not b & 0x0F
    terrain.set_tile_byte(st, nx, ny, 0xC0 | p)
    st.obj_x[p], st.obj_y[p], st.obj_z_height[p] = nx, ny, b >> 4


@pytest.mark.oracle
@pytest.mark.parametrize("typed", sorted(BOARDS))
def test_model_is_cycle_exact_on_live_245b(typed):
    """Model == measured cycles at the start position and two relocations."""
    cpu, mem, _ = oracle.generate_machine(landscape.seed_for(typed))
    game = Game.typed(typed)
    assert bytes(mem[0x0400:0x0800]) == bytes(game.state.mem[0x0400:0x0800])
    for pos in (None,) + BOARDS[typed]:
        if pos is not None:
            _relocate(mem, *pos)
            _relocate(game.state.mem, *pos)
        got = _measure(cpu, mem)
        want = occlusioncost.occlusion_cycles(game.state)
        assert want == got, f"ls{typed} obs={pos}: model {want} != live {got}"


def test_scaffolding_constants_are_instruction_sums():
    """Each derived constant equals the cycle sum of its instruction sequence."""
    assert occlusioncost.ENTRY == sum((4, 2, 2, 3, 2))  # $245B..$2464
    assert occlusioncost.ZERO_FILL == 128 * (5 + 2 + 3) - 1  # $2466..$246A
    # $24ED..$2500 head: JSR+$352C ($352C..$355D) + reads + JSR+$1ECC ($1ECC..$1EEE)
    assert 35 == 6 + sum((4, 2, 4, 2, 2, 2, 2, 2, 3, 6))
    assert 59 == 6 + sum((2, 3, 3, 3, 2, 3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 6))
    assert occlusioncost.TILE_HEAD == 35 + sum((2, 3, 2, 3, 3, 5, 2, 3, 3)) + 59
    # $2532..$2541 normalise lap: ASL/ROL zp x6 + LSR $17 + ASL A + BCC taken
    assert occlusioncost.NORM_LAP == sum((5, 5, 5, 5, 5, 5, 5, 2, 3))
    # $257A..$25AA march lap: seven zp add triples + CMP (zp),Y + BCC/DEX/BNE
    lap = sum((3, 2, 3, 3, 2, 3, 2)) + sum((3, 2, 3, 3, 3, 3, 3))
    lap += sum((3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 5))
    assert lap == 72
    assert occlusioncost.LAP_FULL == lap + 2 + 2 + 3
    assert occlusioncost.LAP_OCCL == lap + 3 + 2  # BCC taken + $25AF SEC
    assert occlusioncost.LAP_EXHAUST == lap + 2 + 2 + 2 + 2 + 3  # $25AC CLC+BCC
    # $25B0..$25C0 tile tail; last tile takes BMI + RTS instead of BNE + JMP
    assert occlusioncost.TILE_TAIL == sum((3, 3, 2, 2, 2, 5, 5, 2, 3))
    assert occlusioncost.TILE_TAIL_LAST == sum((3, 3, 2, 2, 2, 5, 5, 3, 6))
    # $249A..$24D3 merge fixed part (incl DEX + BPL taken)
    merge = sum((3, 2, 2, 2, 2, 2, 3, 2, 3)) + sum((2, 2, 2, 2, 4, 2, 3))
    merge += sum((4, 4, 4, 4, 3, 4)) + sum((3, 3, 4, 3, 3, 5)) + 2 + 3
    assert occlusioncost.MERGE_TAIL == merge


def test_model_runs_without_an_image():
    """The model prices a pure-sim board (no emulator, no ROM fixture reads)."""
    state = Game.typed(42).state
    cycles = occlusioncost.occlusion_cycles(state)
    assert 500_000 < cycles < 6_000_000
    assert cycles == occlusioncost.occlusion_cycles(state)
