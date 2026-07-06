import os
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "out", "sentinel_stage2.bin")
TAP = os.path.join(ROOT, "sentinel-gold.tap")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "oracle: test drives the real 6502 code (needs out/sentinel_stage2.bin)",
    )


def pytest_collection_modifyitems(config, items):
    have_img = os.path.exists(IMG)
    have_tap = os.path.exists(TAP)
    have_docker = os.system("docker info >/dev/null 2>&1") == 0
    no_img = pytest.mark.skip(reason="needs out/sentinel_stage2.bin fixture")
    for item in items:
        if "test_video_record" in item.nodeid:
            if not (have_tap and have_docker):
                item.add_marker(pytest.mark.skip(reason="needs tape fixture + docker"))
        elif item.nodeid.startswith("sentinel" + os.sep) or item.nodeid.startswith(
            "sentinel/"
        ):
            # The standalone package runs without the ROM image; only tests that
            # differentially validate against the real 6502 code (marked `oracle`)
            # need the fixture.
            if not have_img and item.get_closest_marker("oracle"):
                item.add_marker(no_img)
        elif not have_img:
            # Legacy scripts/ tests all reach the emulator; skip when it's absent.
            item.add_marker(no_img)
