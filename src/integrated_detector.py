"""
Integrated detector combining YOLO for detection and CNN for classification refinement
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.cnn_classifier import CNNClassifier
import yaml

class IntegratedDetector:
    def __init__(self, yolo_model_path="models/yolov8/train/weights/best.pt", 
                 cnn_model_path="models/cnn/best_model.h5",
                 config_path="config/yolo_config.yaml"):
        """Khởi tạo integrated detector"""
        
        # Load YOLO detector
        self.yolo_detector = YOLO(yolo_model_path)
        
        # Load CNN classifier
        self.cnn_classifier = CNNClassifier(52)
        self.cnn_classifier.load(cnn_model_path)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['classes']['names']
    
    def detect_with_refinement(self, image, yolo_conf_threshold=0.3, 
                             cnn_conf_threshold=0.7, top_k=5):
        """
        Phát hiện và phân loại biển báo với refinement bằng CNN
        
        Args:
            image: numpy array hoặc đường dẫn ảnh
            yolo_conf_threshold: ngưỡng confidence cho YOLO
            cnn_conf_threshold: ngưỡng confidence cho CNN
            top_k: số lượng top predictions để xem xét
            
        Returns:
            refined_detections: danh sách detection đã được refinement
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO detection
        yolo_results = self.yolo_detector.predict(
            image,
            conf=yolo_conf_threshold,
            iou=0.45,
            verbose=False
        )
        
        refined_detections = []
        
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                yolo_conf = float(box.conf[0])
                yolo_cls = int(box.cls[0])
                
                # Crop detected region for CNN classification
                cropped = image[int(y1):int(y2), int(x1):int(x2)]
                if cropped.size == 0:
                    continue
                
                # CNN classification refinement
                cnn_result = self.cnn_classifier.predict(cropped)
                cnn_class_id = cnn_result['class_id']
                cnn_confidence = cnn_result['confidence']
                top5_probs = cnn_result['all_probabilities']
                
                # Get top-k predictions
                top_k_indices = np.argsort(top5_probs)[-top_k:][::-1]
                top_k_predictions = [
                    {
                        'class_id': int(idx),
                        'class_name': self.class_names.get(idx, f"class_{idx}"),
                        'confidence': float(top5_probs[idx])
                    }
                    for idx in top_k_indices
                ]
                
                # Decide final classification
                if cnn_confidence >= cnn_conf_threshold:
                    # Use CNN prediction if confident
                    final_class_id = cnn_class_id
                    final_confidence = cnn_confidence * yolo_conf  # Combined confidence
                    refinement_source = "CNN"
                else:
                    # Fall back to YOLO prediction
                    final_class_id = yolo_cls
                    final_confidence = yolo_conf
                    refinement_source = "YOLO"
                
                refined_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'final_class_id': final_class_id,
                    'final_class_name': self.class_names.get(final_class_id, f"class_{final_class_id}"),
                    'final_confidence': final_confidence,
                    'yolo_class_id': yolo_cls,
                    'yolo_confidence': yolo_conf,
                    'cnn_class_id': cnn_class_id,
                    'cnn_confidence': cnn_confidence,
                    'refinement_source': refinement_source,
                    'top_k_predictions': top_k_predictions
                })
        
        return refined_detections
    
    def visualize_detections(self, image, detections, save_path=None):
        """Vẽ bounding boxes với thông tin refinement"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['final_class_name']
            conf = det['final_confidence']
            source = det['refinement_source']
            
            # Màu sắc khác nhau cho các source
            if source == "CNN":
                color = (0, 255, 0)  # Xanh lá - CNN refinement
            else:
                color = (255, 0, 0)  # Đỏ - YOLO fallback
            
            # Vẽ bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label với thông tin source
            label_text = f"{label}: {conf:.2f} ({source})"
            cv2.putText(image_copy, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
        
        return image_copy

# Test function
def test_integrated_detector():
    """Test integrated detector"""
    detector = IntegratedDetector()
    
    # Test với ảnh mẫu (cần có file test image)
    try:
        detections = detector.detect_with_refinement("test_image.jpg")
        print(f"Found {len(detections)} detections")
        
        for i, det in enumerate(detections):
            print(f"\nDetection {i+1}:")
            print(f"  Final: {det['final_class_name']} ({det['final_confidence']:.2f})")
            print(f"  YOLO: {det['yolo_class_id']} ({det['yolo_confidence']:.2f})")
            print(f"  CNN: {det['cnn_class_id']} ({det['cnn_confidence']:.2f})")
            print(f"  Source: {det['refinement_source']}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        print("Please provide a test image or use your own image path")

if __name__ == "__main__":
    test_integrated_detector()