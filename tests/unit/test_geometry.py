import math

import pytest

from app.logic import geometry


SQUARE = [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1},
          {"x": 0.9, "y": 0.9}, {"x": 0.1, "y": 0.9}]


def test_polygon_membership_is_orientation_independent_and_boundary_outside():
    for polygon in (SQUARE, list(reversed(SQUARE))):
        assert geometry.point_in_polygon((0.5, 0.5), polygon)
        assert not geometry.point_in_polygon((0.1, 0.5), polygon)
        assert not geometry.point_in_polygon((0.1, 0.1), polygon)
        assert not geometry.point_in_polygon((0.95, 0.5), polygon)


def test_validation_returns_an_independent_normalized_polygon():
    assert hasattr(geometry, "validate_polygon"), "polygon validation is missing"
    result = geometry.validate_polygon(SQUARE)
    assert result == SQUARE
    assert result is not SQUARE
    assert result[0] is not SQUARE[0]


@pytest.mark.parametrize("points", [
    SQUARE[:2],
    SQUARE + [SQUARE[0]],
    [{"x": 0, "y": 0}, {"x": 0.5, "y": 0.5}, {"x": 1, "y": 1}],
    [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}, {"x": 1, "y": 0}],
    [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 0.8}, {"x": 1, "y": 0}],
    [{"x": 0, "y": 0}, {"x": math.nan, "y": 1}, {"x": 1, "y": 0}],
    [{"x": 0, "y": 0}, {"x": math.inf, "y": 1}, {"x": 1, "y": 0}],
    [{"x": -0.01, "y": 0}, {"x": 1, "y": 1}, {"x": 1, "y": 0}],
    [{"x": 0, "y": 0}, {"x": 1, "y": 1.01}, {"x": 1, "y": 0}],
    [{"x": 0, "y": 0}, {"x": True, "y": 1}, {"x": 1, "y": 0}],
])
def test_validation_rejects_invalid_polygons(points):
    assert hasattr(geometry, "validate_polygon"), "polygon validation is missing"
    with pytest.raises(ValueError):
        geometry.validate_polygon(points)


def test_validation_accepts_concave_polygon_and_both_orientations():
    assert hasattr(geometry, "validate_polygon"), "polygon validation is missing"
    points = [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0.5, "y": 0.5},
              {"x": 1, "y": 1}, {"x": 0, "y": 1}]
    assert geometry.validate_polygon(points) == points
    assert geometry.validate_polygon(points[::-1]) == points[::-1]
