"""The PRNG port reproduces the real ROM stream (prnd $31CA).

The golden vectors in ``golden_prng.json`` were captured from the actual 6502
routine driven under py65 (seed after the reset that does ``INC $0C7D``), so this
test proves bit-exact parity with the game with no emulator in the loop.
"""

import json
import os

from sentinel.prng import Prng, seed_state

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_prng.json")


def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


def test_golden_stream_matches_rom():
    for ls, g in _golden().items():
        prng = Prng(g["seed_state"])
        outputs = [prng.next() for _ in range(len(g["outputs"]))]
        assert outputs == g["outputs"], f"landscape {ls} output stream diverged"
        assert prng.s == g["final_state"], f"landscape {ls} final state diverged"


def test_seed_sets_low_and_high_bytes():
    # seed_prnd_from_landscape_number $33ED: state[0]=lo, state[1]=hi, rest 0.
    assert seed_state(0x270F) == [0x0F, 0x27, 0, 0, 0]
    assert seed_state(42) == [42, 0, 0, 0, 0]


def test_shuffle_is_deterministic():
    a = Prng([1, 2, 3, 4, 5])
    b = Prng([1, 2, 3, 4, 5])
    for _ in range(100):
        assert a.next() == b.next()
    assert a.s == b.s


def test_all_zero_state_is_a_fixed_point():
    # A degenerate all-zero LFSR stays zero; the game avoids it via INC $0C7D
    # in reset_game_state before seeding landscape 0000.
    prng = Prng([0, 0, 0, 0, 0])
    assert all(prng.next() == 0 for _ in range(64))
    assert prng.s == [0, 0, 0, 0, 0]


def test_load_store_roundtrip():
    mem = bytearray(0x10000)
    Prng([9, 8, 7, 6, 5]).store(mem)
    assert list(mem[0x0C7B:0x0C80]) == [9, 8, 7, 6, 5]
    assert Prng().load(mem).s == [9, 8, 7, 6, 5]
