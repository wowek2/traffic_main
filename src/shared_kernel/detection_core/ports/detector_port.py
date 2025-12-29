from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from shared_kernel.detection_core.domain.detection import Detection
from shared_kernel.result_monad import Result
from shared_kernel.semantic_model.labels import SemanticClass
from shared_kernel.exceptions import DomainError


class DetectorPort(ABC):
    """
    Abstract port for object detection.
    Infrastructure adapters will import this using specific ML frameworks.
    """

    @abstractmethod
    def detect(
        self,
        frame: Any, # Abstracted image (could be numpy arr in implementation)
        filter_classes: Sequence[SemanticClass] | None = None
    ) -> Result[Sequence[Detection], DomainError]:
        """
        Perform detection on the given frame.

        Args:
            frame: Input frame.
            filter_classes: Optional list of classes to include. If None return All.
        Returns:
            Result containing list of Detections or erro message.
        """
        ...
