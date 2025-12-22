"""
Tests for Point value object.
"""
from math import isclose
from dataclasses import FrozenInstanceError
import pytest

from src.shared_kernel.value_objects import Point

class TestPoint:
    """Unit tests for Point value object."""

    # Initialization and equality tests
    def test_create_point_with_coordinates(self):
        """Test creating a Point with specific coordinates."""
        p = Point(x=3.5, y=4.7)

        assert p.x == 3.5
        assert p.y == 4.7

    def test_equality_by_value(self):
        """Two points with the same coordinates should be equal."""
        p1 = Point(x=1.0, y=2.0)
        p2 = Point(x=1.0, y=2.0)

        assert p1 == p2

    def test_hashability(self):
        """Point should be hashable and usable in sets/dicts."""
        p1 = Point(x=1.0, y=2.0)
        unique_points = {p1, Point(x=1.0, y=2.0)}

        assert len(unique_points) == 1
        assert p1 in unique_points

    # Imutabulity test
    def test_is_immutable(self):
        """Point attributes should be immutable."""
        p = Point(x=1.0, y=2.0)

        with pytest.raises(FrozenInstanceError):
            p.x = 3.0

    # Geometric method tests
    def test_distance_to_zero(self):
        """Distance from a point to itself should be zero."""
        p = Point(x=5.0, y=5.0)

        assert p.distance_to(p) == 0.0

    def test_distance_to_another_point(self):
        """Test distance calculation between two points."""
        p1 = Point(x=0.0, y=0.0)
        p2 = Point(x=3.0, y=4.0)

        distance = p1.distance_to(p2)

        assert isclose(distance, 5.0)  # 3-4-5 triangle

    def test_distance_commutativity(self):
        """Distance A->B should equal distance B->A."""
        p1 = Point(x=1.0, y=2.0)
        p2 = Point(x=4.0, y=6.0)

        dist1 = p1.distance_to(p2)
        dist2 = p2.distance_to(p1)

        assert dist1 == dist2

    # Conversion method tests
    def test_as_tuple(self):
        """Test conversion of Point to tuple."""
        p = Point(x=2.5, y=7.5)

        result = p.as_tuple()

        assert result == (2.5, 7.5)

    def test_repr_string(self):
        """Test the string representation of Point."""
        p = Point(x=1.0, y=2.0)

        repr_str = repr(p)

        assert repr_str == "Point(x=1.0, y=2.0)"