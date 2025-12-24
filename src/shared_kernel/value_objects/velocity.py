from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt


@dataclass(frozen=True, slots=True)
class Velocity:
    """
    Immutable velocity vector in 2D space.
    Attributes:
        vx (float): Velocity component in x-direction.
        vy (float): Velocity component in y-direction.
    """
    dx: float
    dy: float

    @property
    def speed(self) -> float:
        """Calculate the speed (magnitude) of the velocity vector.
        Returns:
            Speed as a float.
        """
        return sqrt(self.dx**2 + self.dy**2)

    @property
    def angle(self) -> float:
        """Calculate the angle (direction) of the velocity vector in radians.
        Returns:
            Angle in radians as a float.
        """
        return atan2(self.dy, self.dx)
