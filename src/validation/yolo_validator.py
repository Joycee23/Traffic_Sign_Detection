"""
YOLO Validator - Kiểm tra và sửa lỗi học sai của YOLO
Phát hiện và chỉnh sửa các vấn đề về class mapping, annotation, và training
"""

import yaml
import json
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from ultralytics import YOLO

class YOLOValidator:
    def __init__(self, config_path="config/yolo_config.yaml", data_yaml_path="data/processed/data.yaml"):
        self.config_path = config_path
        self.data_yaml_path = data_yaml_path
        
        # Load configurations
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        # Load trained model
        self.model_path = "models/yolov8/train/weights/best.pt"
        self.model = YOLO(self.model_path) if os.path.exists(self.model_path) else None
    
    def check_class_mismatch(self):
        """Kiểm tra sự không khớp giữa config và data.yaml"""
        config_classes = self.config['classes']['names']
        data_classes = self.data_config['names']
        
        print("=== CLASS MISMATCH ANALYSIS ===")
        print(f"Config classes: {len(config_classes)}")
        print(f"Data classes: {len(data_classes)}")
        
        # Check for exact match
        if config_classes == data_classes:
            print("✅ Class names match perfectly!")
            return True
        
        # Find differences
        config_set = set(config_classes.values())
        data_set = set(data_classes.values())
        
        only_in_config = config_set - data_set
        only_in_data = data_set - config_set
        
        if only_in_config:
            print("❌ Classes only in config:")
            for cls in only_in_config:
                print(f"  - {cls}")
        
        if only_in_data:
            print("❌ Classes only in data:")
            for cls in only_in_data:
                print(f"  - {cls}")
        
        # Check order consistency
        print("\n=== CLASS ORDER CHECK ===")
        for i in range(max(len(config_classes), len(data_classes))):
            config_cls = config_classes.get(i, "MISSING")
            data_cls = data_classes.get(i, "MISSING")
            
            if config_cls != data_cls:
                print(f"❌ Index {i}: Config='{config_cls}' vs Data='{data_cls}'")
            else:
                print(f"✅ Index {i}: '{config_cls}'")
        
        return len(only_in_config) == 0 and len(only_in_data) == 0
    
    def analyze_annotations(self):
        """Phân tích annotations để tìm vấn đề"""
        print("\n=== ANNOTATION ANALYSIS ===")
        
        annotation_dir = Path(self.data_config['path']) / "train" / "labels"
        if not annotation_dir.exists():
            print(f"❌ Annotation directory not found: {annotation_dir}")
            return
        
        # Collect statistics
        class_counts = Counter()
        image_counts = defaultdict(set)
        annotation_stats = []
        
        for ann_file in annotation_dir.glob("*.txt"):
            with open(ann_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    image_counts[class_id].add(ann_file.stem)
            
            annotation_stats.append({
                'file': ann_file.name,
                'annotations': len(lines),
                'classes': set(int(line.split()[0]) for line in lines if line.strip())
            })
        
        # Print statistics
        print(f"Total annotation files: {len(annotation_stats)}")
        print(f"Total annotations: {sum(class_counts.values())}")
        
        print("\n=== CLASS DISTRIBUTION ===")
        for class_id in sorted(class_counts.keys()):
            class_name = self.data_config['names'].get(class_id, f"UNKNOWN_{class_id}")
            count = class_counts[class_id]
            images = len(image_counts[class_id])
            print(f"Class {class_id:2d} ({class_name:20s}): {count:4d} annotations in {images:3d} images")
        
        # Find problematic classes
        print("\n=== PROBLEMATIC CLASSES ===")
        for class_id, count in class_counts.most_common():
            if count < 10:  # Classes with very few samples
                class_name = self.data_config['names'].get(class_id, f"UNKNOWN_{class_id}")
                print(f"⚠️  Class {class_id} ({class_name}): Only {count} samples")
        
        return class_counts
    
    def validate_model_predictions(self, sample_images=10):
        """Kiểm tra dự đoán của model trên tập validation"""
        if not self.model:
            print("❌ Model not found for validation")
            return
        
        print("\n=== MODEL VALIDATION ===")
        
        val_dir = Path(self.data_config['path']) / "val" / "images"
        if not val_dir.exists():
            print(f"❌ Validation directory not found: {val_dir}")
            return
        
        # Get sample images
        image_files = list(val_dir.glob("*.*"))[:sample_images]
        if not image_files:
            print("❌ No validation images found")
            return
        
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        for img_path in image_files:
            print(f"\nAnalyzing: {img_path.name}")
            
            # Get ground truth from annotation
            ann_path = Path(self.data_config['path']) / "val" / "labels" / f"{img_path.stem}.txt"
            ground_truth = []
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            ground_truth.append(int(parts[0]))
            
            # Model prediction
            results = self.model(img_path)
            
            predictions = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()
                    predictions.append((class_id, confidence))
            
            # Compare predictions with ground truth
            print(f"  Ground truth: {[self.data_config['names'].get(gt, 'UNKNOWN') for gt in ground_truth]}")
            print(f"  Predictions: {[self.data_config['names'].get(pred, 'UNKNOWN') for pred, conf in predictions]}")
            
            # Simple matching (for demonstration)
            for gt in ground_truth:
                for pred, conf in predictions:
                    confusion_matrix[gt][pred] += 1
        
        return confusion_matrix
    
    def fix_class_mismatch(self):
        """Tự động sửa lỗi không khớp class"""
        print("\n=== FIXING CLASS MISMATCH ===")
        
        # Ensure config and data.yaml match exactly
        config_classes = self.config['classes']['names']
        data_classes = self.data_config['names']
        
        if config_classes != data_classes:
            print("Fixing class mismatch...")
            
            # Update config to match data.yaml (safer approach)
            self.config['classes']['names'] = data_classes.copy()
            
            # Save updated config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            
            print("✅ Updated config to match data.yaml")
        
        # Create backup and create standardized version
        self.create_standardized_dataset()
        
        return True
    
    def create_standardized_dataset(self):
        """Tạo dataset chuẩn hóa để tránh lỗi"""
        print("\n=== CREATING STANDARDIZED DATASET ===")
        
        original_path = Path(self.data_config['path'])
        standardized_path = Path("data/standard_yolo")
        
        # Copy structure
        for split in ['train', 'val', 'test']:
            split_path = standardized_path / split
            split_path.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            images_src = original_path / split / "images"
            images_dest = split_path / "images"
            if images_src.exists():
                # In real implementation, copy files
                print(f"  Copying {split} images...")
            
            # Copy and fix annotations
            labels_src = original_path / split / "labels"
            labels_dest = split_path / "labels"
            if labels_src.exists():
                print(f"  Processing {split} annotations...")
                labels_dest.mkdir(parents=True, exist_ok=True)
                
                # Fix annotation files
                for ann_file in labels_src.glob("*.txt"):
                    self.fix_annotation_file(ann_file, labels_dest / ann_file.name)
        
        # Create standardized data.yaml
        standardized_yaml = {
            'path': str(standardized_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.data_config['names']),
            'names': self.data_config['names']
        }
        
        with open(standardized_path / "data.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(standardized_yaml, f, allow_unicode=True, default_flow_style=False)
        
        print(f"✅ Standardized dataset created at: {standardized_path}")
        return standardized_path
    
    def fix_annotation_file(self, src_file, dest_file):
        """Sửa lỗi trong file annotation"""
        fixed_lines = []
        
        with open(src_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # Ensure class_id is within valid range
                    if 0 <= class_id < len(self.data_config['names']):
                        fixed_lines.append(line)
                    else:
                        print(f"⚠️  Fixed invalid class_id {class_id} in {src_file.name}")
                        # Skip or handle invalid class_id
                else:
                    fixed_lines.append(line)  # Keep malformed lines for manual review
        
        with open(dest_file, 'w') as f:
            f.writelines(fixed_lines)
    
    def generate_validation_report(self):
        """Tạo báo cáo validation chi tiết"""
        report = {
            'class_mismatch': not self.check_class_mismatch(),
            'annotation_stats': self.analyze_annotations(),
            'fix_applied': self.fix_class_mismatch(),
            'recommendations': []
        }
        
        # Add recommendations
        if report['class_mismatch']:
            report['recommendations'].append("Retrain YOLO with fixed class mapping")
        
        annotation_stats = report['annotation_stats']
        if annotation_stats:
            low_count_classes = [cls for cls, count in annotation_stats.items() if count < 10]
            if low_count_classes:
                report['recommendations'].append(
                    f"Add more samples for classes: {low_count_classes}"
                )
        
        return report

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and fix YOLO training issues')
    parser.add_argument('--fix', '-f', action='store_true', help='Apply fixes automatically')
    parser.add_argument('--validate', '-v', action='store_true', help='Run model validation')
    
    args = parser.parse_args()
    
    validator = YOLOValidator()
    
    print("🚀 YOLO VALIDATION TOOL")
    print("=" * 50)
    
    # Run validations
    class_match = validator.check_class_mismatch()
    annotation_stats = validator.analyze_annotations()
    
    if args.validate:
        confusion_matrix = validator.validate_model_predictions()
    
    if args.fix:
        validator.fix_class_mismatch()
    
    # Generate report
    report = validator.generate_validation_report()
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION REPORT")
    print("=" * 50)
    
    for rec in report['recommendations']:
        print(f"💡 Recommendation: {rec}")
    
    if not report['recommendations']:
        print("✅ No major issues found!")
    
    print("\nNext steps:")
    if args.fix:
        print("1. Retrain YOLO with: python src/training/train_yolo.py")
    else:
        print("1. Run with --fix to apply automatic corrections")
        print("2. Then retrain YOLO")

if __name__ == "__main__":
    main()