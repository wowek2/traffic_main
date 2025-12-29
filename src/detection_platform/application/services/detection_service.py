from collections.abc import Sequence
from typing import Any

from shared_kernel.detection_core.domain.detection import Detection
from shared_kernel.detection_core.ports.detector_port import DetectorPort
from shared_kernel.result_monad import Result
from shared_kernel.semantic_model.labels import SemanticClass


class DetectionService:
    """
    Application Service for handling object detection workflow.
    Orchestrates the flow between input data and the Domain Detector Port.
    """

    def __init__(self, detector: DetectorPort) -> None:
        self._detector = detector

    def detect_objects(
        self,
        frame: Any,
        filter_classes: Sequence[SemanticClass] | None = None
    ) -> Result[Sequence[Detection], str]:
        """
        Executes detection logic on a single frame.

        Args:
            frame: Input image frame (infrastructure agnostic type).
            filter_classes: Optional list of SemanticClasses to filter results.

        Returns:
            Result monad containing sequence of Detections or error string.
        """
        # logic buisness here in the future
        return self._detector.detect(frame, filter_classes)