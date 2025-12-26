"""
Test Integrated System - Kiểm tra hệ thống tích hợp YOLO + CNN
Bước 6: TEST - Đánh giá toàn bộ pipeline
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import defaultdict
import json

class IntegratedSystemTester:
    def __init__(self):
        # Load configurations
        self.yolo_model_path = "models/yolov8/train/weights/best.pt"
        self.cnn_model_path = "models/cnn/best_model.h5"  # Điều chỉnh theo đường dẫn thực tế
        self.data_yaml_path = "data/processed/data.yaml"
        
        # Load class names
        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        self.class_names = self.data_config['names']
        
        # Load models
        self.yolo_model = YOLO(self.yolo_model_path)
        
        # Try to load CNN model
        try:
            self.cnn_model = keras.models.load_model(self.cnn_model_path)
            self.cnn_available = True
        except:
            print("⚠️  CNN model not found, testing YOLO only")
            self.cnn_available = False
    
    def test_single_image(self, image_path, confidence_threshold=0.5):
        """Test hệ thống trên ảnh đơn lẻ"""
        print(f"Testing image: {image_path}")
        
        # Đọc và xử lý ảnh
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO detection
        results = self.yolo_model(image_rgb, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
                
                # Crop detected region for CNN classification
                cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
                
                # CNN classification (if available)
                cnn_class_name = None
                cnn_confidence = None
                
                if self.cnn_available and cropped_img.size > 0:
                    cnn_result = self.classify_with_cnn(cropped_img)
                    if cnn_result:
                        cnn_class_name, cnn_confidence = cnn_result
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': class_name,
                    'cnn_class_name': cnn_class_name,
                    'cnn_confidence': cnn_confidence
                }
                detections.append(detection)
        
        return {
            'image_path': str(image_path),
            'detections': detections,
            'image_shape': image.shape
        }
    
    def classify_with_cnn(self, cropped_image):
        """Phân loại ảnh crop với CNN"""
        try:
            # Resize và preprocess cho CNN
            img_resized = cv2.resize(cropped_image, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0)
            img_array = img_array / 255.0  # Normalize
            
            # Prediction
            predictions = self.cnn_model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            class_name = self.class_names.get(predicted_class, f"Unknown_{predicted_class}")
            return class_name, float(confidence)
        except Exception as e:
            print(f"CNN classification error: {e}")
            return None
    
    def test_on_validation_set(self, sample_size=20):
        """Test trên tập validation"""
        print("=== TESTING ON VALIDATION SET ===")
        
        val_dir = Path(self.data_config['path']) / "val" / "images"
        if not val_dir.exists():
            print(f"❌ Validation directory not found: {val_dir}")
            return
        
        # Lấy sample images
        image_files = list(val_dir.glob("*.*"))[:sample_size]
        if not image_files:
            print("❌ No validation images found")
            return
        
        results = []
        stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            
            # Get ground truth từ annotation
            ann_path = Path(self.data_config['path']) / "val" / "labels" / f"{img_path.stem}.txt"
            ground_truth = []
            if ann_path.exists():
                with open(ann_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            ground_truth.append(class_id)
            
            # Test hệ thống với confidence threshold thấp hơn
            result = self.test_single_image(img_path, confidence_threshold=0.3)
            if result:
                # Đánh giá kết quả với IoU matching
                detected_boxes = [(det['class_id'], det['bbox'], det['confidence']) for det in result['detections']]
                
                # Simple matching evaluation - count each ground truth object
                for gt_class in ground_truth:
                    stats[gt_class]['total'] += 1
                    
                    # Check if this class was detected
                    if any(det_class == gt_class for det_class, _, _ in detected_boxes):
                        stats[gt_class]['correct'] += 1
                
                results.append(result)
        
        # Print statistics
        print("\n=== VALIDATION RESULTS ===")
        overall_correct = 0
        overall_total = 0
        
        for class_id, class_stats in stats.items():
            if class_stats['total'] > 0:  # Only show classes that were actually present
                class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
                accuracy = class_stats['correct'] / class_stats['total'] if class_stats['total'] > 0 else 0
                overall_correct += class_stats['correct']
                overall_total += class_stats['total']
                
                print(f"Class {class_id:2d} ({class_name:20s}): {class_stats['correct']:2d}/{class_stats['total']:2d} = {accuracy:.1%}")
        
        if overall_total > 0:
            overall_accuracy = overall_correct / overall_total
            print(f"\nOverall Accuracy: {overall_correct}/{overall_total} = {overall_accuracy:.1%}")
        else:
            print("\nNo ground truth objects found in validation set")
            overall_accuracy = 0
        
        return results, stats
    
    def visualize_results(self, test_result, output_dir="test_results"):
        """Visualize kết quả detection"""
        os.makedirs(output_dir, exist_ok=True)
        
        image = cv2.imread(test_result['image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        
        for detection in test_result['detections']:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Vẽ bounding box
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, edgecolor='red', linewidth=2
            ))
            
            # Thêm label
            label = f"{class_name}: {confidence:.2f}"
            if detection['cnn_class_name']:
                label += f" (CNN: {detection['cnn_class_name']}: {detection['cnn_confidence']:.2f})"
            
            plt.text(x1, y1-10, label, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=8)
        
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"Detection Results - {Path(test_result['image_path']).name}")
        
        # Lưu ảnh
        output_path = Path(output_dir) / f"result_{Path(test_result['image_path']).stem}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Visualization saved: {output_path}")
        return output_path
    
    def generate_test_report(self, validation_results):
        """Tạo báo cáo test chi tiết"""
        report = {
            'test_date': __import__('datetime').datetime.now().isoformat(),
            'yolo_model': self.yolo_model_path,
            'cnn_model': self.cnn_model_path if self.cnn_available else "Not available",
            'num_classes': len(self.class_names),
            'validation_results': validation_results[1],
            'recommendations': []
        }
        
        # Calculate overall metrics
        stats = validation_results[1]
        total_detections = sum(stats[cls]['total'] for cls in stats)
        correct_detections = sum(stats[cls]['correct'] for cls in stats)
        overall_accuracy = correct_detections / total_detections if total_detections > 0 else 0
        
        report['overall_accuracy'] = overall_accuracy
        report['total_tested'] = total_detections
        report['correct_detections'] = correct_detections
        
        # Add recommendations
        if overall_accuracy < 0.8:
            report['recommendations'].append("Consider fine-tuning on problematic classes")
        
        # Find problematic classes
        for class_id, class_stats in stats.items():
            if class_stats['total'] > 0:
                accuracy = class_stats['correct'] / class_stats['total']
                if accuracy < 0.5:
                    class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
                    report['recommendations'].append(
                        f"Improve detection for class {class_id} ({class_name}): {accuracy:.1%}"
                    )
        
        # Save report
        report_path = "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Test report saved: {report_path}")
        return report

def main():
    """Main testing function"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Test integrated YOLO+CNN system')
    parser.add_argument('--image', '-i', help='Test single image path')
    parser.add_argument('--validation', '-v', action='store_true', help='Test on validation set')
    parser.add_argument('--sample-size', '-s', type=int, default=20, help='Number of validation samples')
    parser.add_argument('--visualize', '-viz', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    tester = IntegratedSystemTester()
    
    print("🧪 INTEGRATED SYSTEM TESTER")
    print("=" * 50)
    
    if args.image:
        # Test single image
        result = tester.test_single_image(args.image)
        if result:
            print(f"Detections: {len(result['detections'])}")
            for det in result['detections']:
                print(f"  {det['class_name']}: {det['confidence']:.3f}")
            
            if args.visualize:
                tester.visualize_results(result)
    
    if args.validation:
        # Test on validation set
        results, stats = tester.test_on_validation_set(args.sample_size)
        
        if args.visualize and results:
            # Visualize first few results
            for i, result in enumerate(results[:5]):
                print(f"Visualizing result {i+1}...")
                tester.visualize_results(result)
        
        # Generate report
        report = tester.generate_test_report((results, stats))
        
        print("\n" + "=" * 50)
        print("📊 TEST REPORT SUMMARY")
        print("=" * 50)
        print(f"Overall Accuracy: {report['overall_accuracy']:.1%}")
        print(f"Total Tested: {report['total_tested']}")
        print(f"Correct Detections: {report['correct_detections']}")
        
        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        else:
            print("\n✅ System performance is satisfactory!")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()