import cv2
import yaml
from ultralytics import YOLO


class YOLODetector:
    def __init__(
        self,
        model_path="models/yolov8/train/weights/best.pt",
        config_path="config/yolo_config.yaml"
    ):
        self.model = YOLO(model_path)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.class_names = cfg["classes"]["names"]

    def detect(self, image, conf_threshold=0.25, iou_threshold=0.45):
        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("Ảnh không hợp lệ")

        results = self.model.predict(
            image,
            imgsz=416,          # 🔥 GIỐNG LÚC TRAIN
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 4),
                    "class_id": cls,
                    "class_name": self.class_names[cls]
                })

        return {
            "detections": detections,
            "num_detections": len(detections),
            "image_shape": image.shape[:2]
        }
