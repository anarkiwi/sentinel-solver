"""Tests for los.sweep_with_centres."""

from sentinel import los
from sentinel.game import Game


def test_sweep_with_centres():
    g = Game.new(0)
    eye_z = int(g.state.eye_z())
    views, centres = los.sweep_with_centres(g.state, g.state.player, eye_z)

    assert isinstance(views, dict)
    assert isinstance(centres, dict)
    assert set(views) == set(centres)

    for view in views.values():
        assert "h_angle" in view
        assert "v_angle" in view
        assert "cursor" in view

    for centre in centres.values():
        assert isinstance(centre, int)
        assert 0 <= centre <= 255

    assert set(views) == set(los.visible_tiles(g.state, g.state.player, eye_z))
