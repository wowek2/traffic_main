from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True, slots=True)
class Point:
    """
    Immutable point in 2D space.
    """
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Calculate the Euclidean distance to another point.
        Args:
            other (Point): Another point.
        Returns:
            Euclidean distance as a float.
        """
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def as_tuple(self) -> tuple[float, float]:
        """Return the point as a tuple (x, y).
        Returns:
            Tuple of (x, y) coordinates.
        """
        return (self.x, self.y)
