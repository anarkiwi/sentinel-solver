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
