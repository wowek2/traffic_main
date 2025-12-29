import argparse
import sys

import cv2  # type: ignore
from detection_platform.application.services.detection_service import DetectionService
from detection_platform.infrastructure.adapters.detection.yolo_adapter import YoloAdapter


def run_detection(source: str | int, model_path: str) -> None:
    # 1. Infrastructure Setup (Adapter)
    try:
        print(f"Loading model from: {model_path}")
        adapter = YoloAdapter(model_path=model_path)
    except Exception as e:
        print(f"Error initializing infrastructure: {e}")
        sys.exit(1)

    # 2. Application Setup (Dependency Injection)
    service = DetectionService(detector=adapter)

    # 3. Execution (Simple Video Loop - Presentation Logic)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open video source: {source}")
        sys.exit(1)

    print(f"Starting detection on: {source} (Press 'q' to exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = service.detect_objects(frame)

        if result.is_ok():
            detections = result.unwrap()
            
            for det in detections:
                bbox = det.bbox
                x1, y1 = int(bbox.x1), int(bbox.y1)
                x2, y2 = int(bbox.x2), int(bbox.y2)

                label_text = f"{det.class_label.value}: {det.confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"Detection error: {result.unwrap_err()}")

        cv2.imshow('Detection Platform CLI', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def app() -> None:
    """CLI Entrypoint defined in pyproject.toml"""
    parser = argparse.ArgumentParser(description="Modular Detection Platform CLI")

    parser.add_argument("source", type=str, help="Path to video file or camera index (0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model weights")

    args = parser.parse_args()

    # Obsługa "0" jako kamery internetowej (konwersja str -> int dla OpenCV)
    source: str | int = 0 if args.source == "0" else args.source

    run_detection(source, args.model)


if __name__ == "__main__":
    app()