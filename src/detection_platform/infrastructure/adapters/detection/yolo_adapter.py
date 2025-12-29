from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
from shared_kernel.detection_core.domain.detection import Detection
from shared_kernel.detection_core.ports.detector_port import DetectorPort
from shared_kernel.result_monad import Err, Ok, Result
from shared_kernel.semantic_model.labels import SemanticClass
from shared_kernel.value_objects import BoundingBox
from ultralytics import YOLO  # type: ignore[attr-defined]

from .mappers import YoloClassMapper


class YoloAdapter(DetectorPort):
    def __init__(self, model_path: str, confidence_threshold: float = 0.5) -> None:
        try:
            self._model = YOLO(model_path)
            self._conf_threshold = confidence_threshold
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}") from e

    def detect(
        self,
        frame: Any,
        filter_classes: Sequence[SemanticClass] | None = None
    ) -> Result[Sequence[Detection], str]:

        if not isinstance(frame, np.ndarray):
             return Err("YoloAdapter requires numpy array as frame input")

        try:
            results = self._model.predict(frame, conf=self._conf_threshold, verbose=False)

            domain_detections: list[Detection] = []

            for result in results:
                # Jeśli nic nie wykryto, boxes może być None
                if result.boxes is None:
                    continue

                boxes_iter = cast("Iterable[Any]", result.boxes)

                for box in boxes_iter:
                    coords = box.xyxy[0].tolist()

                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    semantic_label = YoloClassMapper.map_id(cls_id)

                    if filter_classes and semantic_label not in filter_classes:
                        continue

                    bbox_result = BoundingBox.create(
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3]
                    )

                    if bbox_result.is_err():
                        continue

                    detection = Detection(
                        bbox=bbox_result.unwrap(),
                        class_label=semantic_label,
                        confidence=conf,
                        track_id=None
                    )
                    domain_detections.append(detection)

            return Ok(domain_detections)

        except Exception as e:
            return Err(f"Detection infrastructure error: {e!s}")
