import os
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "out", "sentinel_stage2.bin")
TAP = os.path.join(ROOT, "sentinel-gold.tap")


def pytest_collection_modifyitems(config, items):
    have_img = os.path.exists(IMG)
    have_tap = os.path.exists(TAP)
    have_docker = os.system("docker info >/dev/null 2>&1") == 0
    for item in items:
        if "test_video_record" in item.nodeid and not (have_tap and have_docker):
            item.add_marker(pytest.mark.skip(reason="needs tape fixture + docker"))
        elif not have_img:
            item.add_marker(pytest.mark.skip(reason="needs out/sentinel_stage2.bin fixture"))
