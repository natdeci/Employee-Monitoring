from ultralytics import YOLO

class YOLOInference:
    def __init__(self, model_path, confidence_threshold=0.7):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_people(self, frame):
        results = self.model(frame)
        return [
            map(int, box.xyxy[0])
            for box in results[0].boxes
            if int(box.cls[0]) == 0 and box.conf[0] > self.confidence_threshold
        ]
