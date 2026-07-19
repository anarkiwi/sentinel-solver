"""Unit tests for the foundation driver's pure reads and operation verification.
No VICE/Docker: a fake monitor backs a 64KB byte image, and the keyboard fire is
faked to mutate it, so the aim->fire->verify logic is exercised on its own."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import core  # noqa: E402


class FakeBM:
    """A monitor over an in-memory 64KB image (all object slots start empty)."""

    def __init__(self):
        self.mem = bytearray(0x10000)
        for s in range(64):
            self.mem[core.A_FLAGS + s] = 0x80  # bit7 == empty slot

    def mem_get(self, a, b):
        return bytes(self.mem[a : b + 1])

    def set_obj(self, slot, x, y, otype, flags=0x00):
        self.mem[core.A_FLAGS + slot] = flags
        self.mem[core.A_X + slot] = x
        self.mem[core.A_Y + slot] = y
        self.mem[core.A_TYPE + slot] = otype

    def clear_obj(self, slot):
        self.mem[core.A_FLAGS + slot] = 0x80

    def set_player(self, slot, x, y):
        self.mem[core.A_SLOT] = slot
        self.mem[core.A_X + slot] = x
        self.mem[core.A_Y + slot] = y

    def set_energy(self, e):
        self.mem[core.A_ENERGY] = e


def _driver(bm):
    d = core.SentinelDriver(bm, log=lambda *a: None)
    d.aim = lambda tile, want_centre=False: True  # pretend the sights always land
    return d


def test_energy_is_six_bit():
    bm = FakeBM()
    bm.set_energy(0x4A)  # 0x4A & 0x3F == 0x0A
    assert core.energy(bm) == 0x0A


def test_player_tile():
    bm = FakeBM()
    bm.set_player(62, 8, 17)
    assert core.player_tile(bm) == (8, 17)


def test_object_in_tile_resolves_stack_head():
    bm = FakeBM()
    bm.set_obj(5, 3, 4, core.T_BOULDER, flags=0x00)  # boulder on the ground
    bm.set_obj(9, 3, 4, core.T_ROBOT, flags=0x40 | 5)  # robot stacked ON slot 5
    assert set(core.slots_at_tile(bm, 3, 4)) == {
        (5, core.T_BOULDER),
        (9, core.T_ROBOT),
    }
    # the topmost is the robot: slot 5 supports slot 9, so 9 is the stack head.
    assert core.object_in_tile(bm, 3, 4) == (9, core.T_ROBOT)
    assert core.object_in_tile(bm, 0, 0) is None


def test_create_returns_new_slot_on_verified_object():
    bm = FakeBM()
    bm.set_player(62, 8, 17)
    bm.set_energy(10)
    d = _driver(bm)
    d.kbd.tap_action = lambda key, **k: bm.set_obj(10, 3, 4, core.T_BOULDER)
    assert d.create(core.T_BOULDER, (3, 4)) == 10


def test_create_none_when_object_never_appears():
    bm = FakeBM()
    bm.set_player(62, 8, 17)
    bm.set_energy(10)
    d = _driver(bm)
    d.kbd.tap_action = lambda key, **k: None  # fire does nothing
    assert d.create(core.T_BOULDER, (3, 4)) is None


def test_absorb_nothing_there_is_true():
    assert _driver(FakeBM()).absorb((3, 4)) is True


def test_absorb_true_when_object_removed():
    bm = FakeBM()
    bm.set_obj(7, 3, 4, core.T_TREE)
    bm.set_energy(5)
    d = _driver(bm)
    d.kbd.tap_action = lambda key, **k: bm.clear_obj(7)
    assert d.absorb((3, 4)) is True


def test_absorb_false_when_object_persists():
    bm = FakeBM()
    bm.set_obj(7, 3, 4, core.T_TREE)
    d = _driver(bm)
    d.kbd.tap_action = lambda key, **k: None
    assert d.absorb((3, 4)) is False


def test_transfer_true_when_player_moves():
    bm = FakeBM()
    bm.set_player(62, 8, 17)
    bm.set_obj(40, 3, 4, core.T_ROBOT)
    d = _driver(bm)
    d.kbd.tap_action = lambda key, **k: bm.set_player(40, 3, 4)
    assert d.transfer((3, 4)) is True


def test_won_reads_landscape_complete_bit():
    bm = FakeBM()
    assert _driver(bm).won() is False
    bm.mem[core.A_LANDSCAPE_DONE] = 0x40
    assert _driver(bm).won() is True


def test_landscape_from_digits_keeps_the_high_bcd_byte():
    """The seed is the whole packed-BCD code, not just its low byte: the PRNG is
    seeded from both bytes ($0C7B/$0C7C), so dropping the leading pair selects a
    different landscape for any code above 0099."""
    assert core.landscape_from_digits("0042") == 0x0042
    assert core.landscape_from_digits("0335") == 0x0335
    assert core.landscape_from_digits("2024") == 0x2024


def test_landscape_from_digits_inverts_enter_landscape():
    for seed in (0x0000, 0x0042, 0x0335, 0x2024, 0x9999):
        assert core.landscape_from_digits(f"{seed:04x}") == seed


def test_landscape_from_digits_matches_prng_seed_bytes():
    from sentinel import prng

    seed = core.landscape_from_digits("0335")
    assert prng.seed_state(seed)[:2] == [0x35, 0x03]
