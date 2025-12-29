from dataclasses import dataclass

from shared_kernel.semantic_model.labels import SemanticClass
from shared_kernel.value_objects import BoundingBox


@dataclass(frozen=True, slots=True)
class Detection:
    """
    Represents detected object in a frame.

    Attributes:
        bbox (BoundingBox): Spatial location.
        class_label: what is it?
        confidence: Model certanity (0.0 - 1.0)
        track_id (str | None): Optional identifier if tracking is active.
    """
    bbox: BoundingBox
    class_label: SemanticClass
    confidence: float
    track_id: str | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence {self.confidence} must be between 0.0 and 1.0"
                )

    @property
    def area(self) -> float:
        return self.bbox.area

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center.as_tuple()
