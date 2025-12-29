from __future__ import annotations

from enum import Enum, unique


@unique
class SemanticClass(Enum):
    """
    Buisness domain definition of detectable objects.
    Decoupled from specific model class.
    """

    # vechicles
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"

    # pedastrians
    PEDASTRIAN = "pedastrian"

    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> SemanticClass:
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN
