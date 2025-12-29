from shared_kernel.semantic_model.labels import SemanticClass


class YoloClassMapper:
    """Maps YOLO class IDs (int) to Domain SemanticClass."""

    _MAPPING = {
        0: SemanticClass.PEDASTRIAN,
        1: SemanticClass.BICYCLE,
        2: SemanticClass.CAR,
        3: SemanticClass.MOTORCYCLE,
        4: SemanticClass.BUS,
        5: SemanticClass.TRUCK
    }

    @classmethod
    def map_id(cls, class_id: int) -> SemanticClass:
        return cls._MAPPING.get(class_id, SemanticClass.UNKNOWN)
