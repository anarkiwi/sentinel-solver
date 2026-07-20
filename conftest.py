import os
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "out", "sentinel_stage2.bin")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "oracle: test drives the real 6502 code (needs out/sentinel_stage2.bin)",
    )


def pytest_collection_modifyitems(config, items):
    # Only `oracle` tests (differential against the real 6502) need the ROM fixture.
    if os.path.exists(IMG):
        return
    no_img = pytest.mark.skip(reason="needs out/sentinel_stage2.bin fixture")
    for item in items:
        if item.get_closest_marker("oracle"):
            item.add_marker(no_img)


@pytest.fixture(scope="session")
def board_image():
    """``(landscape_number, at_entry) -> bytes``: the 64 KB board image, generated once
    per key and memoised immutably. ``at_entry`` applies ``Game.new``'s entry scalars
    (cursor, cooldown gate, enemies-frozen flag)."""
    from sentinel import landscape
    from sentinel.game import Game

    cache = {}

    def image(number=0, at_entry=False):
        key = (number, at_entry)
        if key not in cache:
            st = Game.new(number).state if at_entry else landscape.generate(number)
            cache[key] = bytes(st.mem)
        return cache[key]

    return image


@pytest.fixture
def new_state(board_image):
    """``(landscape_number) -> State``: a fresh mutable copy of the cached board, so
    no test can leak mutations into another."""
    from sentinel.state import State

    return lambda number=0: State(bytearray(board_image(number)))


@pytest.fixture
def new_game(board_image):
    """``(landscape_number) -> Game`` at the ROM's at-entry state, over a fresh copy
    of the cached board (the cheap equivalent of ``Game.new``)."""
    from sentinel.game import Game
    from sentinel.state import State

    return lambda number=0: Game(State(bytearray(board_image(number, True))))
