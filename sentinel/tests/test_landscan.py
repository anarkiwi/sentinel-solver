"""The landscape analyzer: features are read from the board, not assumed."""

from sentinel import landscan
from sentinel.game import Game


def test_features_match_the_generated_board():
    row = landscan.features(0)
    st = Game.typed(0).state
    assert row["code"] == 0
    assert row["enemies"] >= 1
    assert row["start_eye"] == round(st.eye_z(), 3)
    assert row["start_energy"] == st.energy
    assert 0 <= row["flat_tiles"] <= 32 * 32
    assert row["roughness"] > 0


def test_distance_is_zero_against_itself():
    row = landscan.features(0)
    assert landscan.distance(row, row) == 0.0


def test_a_rougher_board_is_further_away():
    ref = landscan.features(0)
    near = dict(ref, roughness=ref["roughness"] * 1.05)
    far = dict(ref, roughness=ref["roughness"] * 2.0)
    assert landscan.distance(ref, near) < landscan.distance(ref, far)
