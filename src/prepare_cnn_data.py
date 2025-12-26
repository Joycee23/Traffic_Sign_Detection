"""
Script to prepare CNN training data from YOLO format
Converts YOLO dataset to class-based directory structure for CNN classification
"""

import os
import shutil
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm

def convert_yolo_to_cnn(yolo_data_path, cnn_data_path, data_yaml_path="data/processed/data.yaml"):
    """
    Convert YOLO dataset to CNN format
    
    Args:
        yolo_data_path: Path to YOLO dataset
        cnn_data_path: Path to save CNN dataset
        data_yaml_path: Path to data.yaml file with class names
    """
    
    # Load data.yaml for class names
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    
    # Create CNN directory structure
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_cnn_path = Path(cnn_data_path) / split
        split_yolo_path = Path(yolo_data_path) / split
        
        # Create class directories
        for class_id, class_name in class_names.items():
            class_dir = split_cnn_path / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        images_path = split_yolo_path / 'images'
        labels_path = split_yolo_path / 'labels'
        
        if not images_path.exists():
            print(f"Warning: {images_path} does not exist, skipping {split}")
            continue
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        print(f"Processing {split} split: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Get corresponding label file
            label_path = labels_path / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Read labels and process each detection
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert normalized coordinates to pixels
                    img_height, img_width = image.shape[:2]
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    # Crop the detected region
                    cropped = image[y1:y2, x1:x2]
                    
                    if cropped.size == 0:
                        continue
                    
                    # Save cropped image to class directory
                    class_dir = split_cnn_path / str(class_id)
                    save_path = class_dir / f"{img_path.stem}_{i}{img_path.suffix}"
                    
                    cv2.imwrite(str(save_path), cropped)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error processing {label_path}: {e}")
                    continue

def main():
    """Main function"""
    yolo_data_path = "data/processed"
    cnn_data_path = "data/cnn_processed"
    
    print("Converting YOLO dataset to CNN format...")
    convert_yolo_to_cnn(yolo_data_path, cnn_data_path, "data/processed/data.yaml")
    print("Conversion completed!")

if __name__ == "__main__":
    main()