from dataclasses import dataclass
from typing import Self

from src.shared_kernel.exceptions import DomainError


class InvalidBoundingBoxError(DomainError):
    """Raised when bounding box coordinates violate invariants."""
    pass

@dataclass(frozen=True)
class BoundingBox:
    """Imutable bounding box with validation.
    Attributes:
        x1 (float): Left edge x-coordinate.
        y1 (float): Top edge y-coordinate.
        x2 (float): Right edge x-coordinate.
        y2 (float): Bottom edge y-coordinate.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        """Validate bounding box invariants after initialization."""
        if self.x1 >= self.x2:
            raise InvalidBoundingBoxError(
                f"x1 ({self.x1}) must be less than x2 ({self.x2})"
            )
        if self.y1 >= self.y2:
            raise InvalidBoundingBoxError(
                f"y1 ({self.y1}) must be less than y2 ({self.y2})"
            )

        if self.x1 < 0:
            raise InvalidBoundingBoxError(
                f"x1 ({self.x1}) must be non-negative"
            )
        if self.y1 < 0:
            raise InvalidBoundingBoxError(
                f"y1 ({self.y1}) must be non-negative"
            )

    """
    Factory methods
    """
    @classmethod
    def from_coordinates(cls, coords: tuple[float, float, float, float]) -> Self:
        """Create BoundingBox from a tuple of coordinates.
        Args:
            coords (BBoxTuple): Tuple of (x1, y1, x2, y2).
        Returns:
            New BoundingBox instance.
        """
        x1, y1, x2, y2 = coords
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    "May add more factory methods in the future"

    """
    Properties
    """
    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point (x, y) of the bounding box."""
        center_x = (self.x1 + self.x2) / 2
        center_y = (self.y1 + self.y2) / 2
        return (center_x, center_y)

    @property
    def top_left(self) -> tuple[float, float]:
        """Top-left corner (x1, y1) of the bounding box."""
        return (self.x1, self.y1)

    @property
    def top_right(self) -> tuple[float, float]:
        """Top-right corner (x2, y1) of the bounding box."""
        return (self.x2, self.y1)

    @property
    def bottom_left(self) -> tuple[float, float]:
        """Bottom-left corner (x1, y2) of the bounding box."""
        return (self.x1, self.y2)

    """
    Conversion methods
    """
    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert BoundingBox to a tuple of coordinates.
        Returns:
            Tuple of (x1, y1, x2, y2).
        """
        return (self.x1, self.y1, self.x2, self.y2)

    """ May add more conversion methods in the future """

    """
    Geometric operations
    """
    def iou(self, other: Self) -> float:
        """Calculate Intersection over Union (IoU) with another bounding box.

        IoU = Area of Intersection / Area of Union

        Args:
            other (BoundingBox): Another bounding box to compare with.
        Returns:
            IoU value in range [0.0, 1.0].
            - 0.0 means no overlap.
            - 1.0 means identical boxes.
        """
        # Calculate intersection coordinates
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        # Check if there's no intersection
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        # Calculate areas
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - intersection_area

        return intersection_area / union_area

    def intersection(self, other: Self) -> Self | None:
        """Calculate the intersection bounding box with another bounding box.
        Args:
            other (BoundingBox): Another bounding box.
        Returns:
            BoundingBox representing the intersection area, or None if no intersection.
        """
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        # Check for no intersection
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return None  # No intersection

        return BoundingBox(x1=inter_x1, y1=inter_y1, x2=inter_x2, y2=inter_y2)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if the bounding box contains a given point.
        Args:
            x (float): x-coordinate of the point.
            y (float): y-coordinate of the point.
        Returns:
            True if the point is inside the bounding box, False otherwise.
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def contains_box(self, other: Self) -> bool:
        """Check if the bounding box fully contains another bounding box.
        Args:
            other (BoundingBox): Another bounding box.
        Returns:
            True if this bounding box contains the other, False otherwise.
        """
        return (
            self.x1 <= other.x1 and
            self.y1 <= other.y1 and
            self.x2 >= other.x2 and
            self.y2 >= other.y2
        )

    def overlaps(self, other: Self) -> bool:
        """Check if the bounding box overlaps with another bounding box.
        Args:
            other (BoundingBox): Another bounding box.
        Returns:
            True if the bounding boxes overlap, False otherwise.
        """
        return not (
            self.x2 <= other.x1 or
            self.x1 >= other.x2 or
            self.y2 <= other.y1 or
            self.y1 >= other.y2
        )

    """
    Transformation methods
    """
    def translate(self, dx: float, dy: float) -> Self:
        """Translate the bounding box by (dx, dy).
        Args:
            dx (float): Offset in x-direction.
            dy (float): Offset in y-direction.
        Returns:
            New translated BoundingBox instance.
        """
        return BoundingBox(
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy
        )

    def scale(self, factor: float) -> Self:
        """ Return new bounding box scaled by a factor from its center.
        Args:
            factor (float): Scaling factor. >1 to enlarge, <1 to shrink.
        Returns:
            New scaled BoundingBox instance.
        """
        center_x, center_y = self.center
        new_half_width = (self.width * factor) / 2
        new_half_height = (self.height * factor) / 2

        return BoundingBox(
            x1=max(0, center_x - new_half_width),
            y1=max(0, center_y - new_half_height),
            x2=center_x + new_half_width,
            y2=center_y + new_half_height
        )

    """
    Distance methods
    """
    def center_distance(self, other: Self) -> float:
        """Calculate Euclidean distance between centers of two boxes.
        Args:
            other: Another bounding box.
        Returns:
            Euclidean distance between the centers.
        """
        center_x1, center_y1 = self.center
        center_x2, center_y2 = other.center

        return ((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2) ** 0.5

    def edge_distance(self, other: Self) -> float:
        """Calculate the minimum distance between the edges of two boxes.
        Args:
            other: Another bounding box.
        Returns:
            Minimum distance between the edges. 0 if they overlap.
        """
        if self.overlaps(other):
            return 0.0

        dx = max(other.x1 - self.x2, self.x1 - other.x2, 0)
        dy = max(other.y1 - self.y2, self.y1 - other.y2, 0)

        return (dx ** 2 + dy ** 2) ** 0.5
