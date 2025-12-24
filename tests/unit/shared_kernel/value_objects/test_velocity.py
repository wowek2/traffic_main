"""
Tests for Velocity value object.
"""
from dataclasses import FrozenInstanceError
from math import isclose, pi

import pytest
from src.shared_kernel.value_objects import Velocity


class TestVelocity:
    """Unit tests for Velocity value object."""

    # Basic tests
    def test_create_velocity(self):
        """Test creating a Velocity with specific components."""
        v = Velocity(dx=3.0, dy=4.0)

        assert v.dx == 3.0
        assert v.dy == 4.0

    def test_equality(self):
        """Two velocities with the same components should be equal."""
        v1 = Velocity(dx=1.0, dy=2.0)
        v2 = Velocity(dx=1.0, dy=2.0)

        assert v1 == v2

    def test_hashability(self):
        """Velocity should be hashable and usable in sets/dicts."""
        v1 = Velocity(dx=1.0, dy=2.0)
        unique_velocities = {v1, Velocity(dx=1.0, dy=2.0)}

        assert len(unique_velocities) == 1
        assert v1 in unique_velocities

    def test_is_immutable(self):
        """Velocity attributes should be immutable."""
        v = Velocity(dx=1.0, dy=2.0)

        with pytest.raises(FrozenInstanceError):
            v.dx = 3.0 # type: ignore

    # Calculation tests
    def test_speed_calculation(self):
        """Test speed (magnitude) calculation of the velocity vector."""
        v = Velocity(dx=3.0, dy=4.0)

        speed = v.speed

        assert isclose(speed, 5.0)

    def test_speed_negative_components(self):
        """Speed should be non-negative even with negative components."""
        v = Velocity(dx=-3.0, dy=-4.0)

        speed = v.speed

        assert isclose(speed, 5.0)

    def test_zero_velocity(self):
        """Speed of zero velocity should be zero."""
        v = Velocity(dx=0.0, dy=0.0)

        speed = v.speed

        assert isclose(speed, 0.0)

    def test_angle_direction(self):
        """Test angle (direction) calculation of the velocity vector."""
        v_right = Velocity(dx=1.0, dy=0.0)
        assert v_right.angle == 0.0

        v_up = Velocity(dx=0.0, dy=1.0)
        assert isclose(v_up.angle, pi / 2)

        v_left = Velocity(dx=-1.0, dy=0.0)
        assert isclose(v_left.angle, pi)

        v_down = Velocity(dx=0.0, dy=-1.0)
        assert isclose(v_down.angle, -pi / 2)

    def test_angle_45_degrees(self):
        """Test angle calculation for 45 degrees vector."""
        v = Velocity(dx=1.0, dy=1.0)

        angle = v.angle

        assert isclose(angle, pi / 4)
