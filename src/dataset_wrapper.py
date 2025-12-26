"""
Dataset Wrapper for automatic conversion
Step 2: Auto convert dataset to standardized format without touching original data
"""

import os
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class DatasetWrapper:
    def __init__(self, 
                 original_data_dir="data/processed",
                 temp_data_dir="data/temp_standard",
                 mapping_path="data/mappings/simplified_mapping.yaml"):
        
        self.original_data_dir = Path(original_data_dir)
        self.temp_data_dir = Path(temp_data_dir)
        self.mapping_path = mapping_path
        
        # Load mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = yaml.safe_load(f)
        
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        
    def create_temp_structure(self):
        """Create temporary dataset structure"""
        print("Creating temporary dataset structure...")
        
        # Remove existing temp directory
        if self.temp_data_dir.exists():
            shutil.rmtree(self.temp_data_dir)
        
        # Create directories
        splits = ['train', 'val', 'test']
        for split in splits:
            images_dir = self.temp_data_dir / split / 'images'
            labels_dir = self.temp_data_dir / split / 'labels'
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Temporary structure created at: {self.temp_data_dir}")
    
    def convert_labels(self, original_label_path, temp_label_path):
        """Convert label files using mapping"""
        try:
            with open(original_label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    original_class = int(parts[0])
                    if original_class in self.mapping:
                        # Convert class ID using mapping
                        parts[0] = str(original_class)  # Keep same class ID, just standardize names
                        converted_lines.append(' '.join(parts))
            
            # Write converted labels
            with open(temp_label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(converted_lines))
                
        except Exception as e:
            print(f"Error converting {original_label_path}: {e}")
    
    def copy_and_convert_images(self, original_split, temp_split):
        """Copy images and convert corresponding labels"""
        original_images_dir = self.original_data_dir / original_split / 'images'
        original_labels_dir = self.original_data_dir / original_split / 'labels'
        
        temp_images_dir = self.temp_data_dir / temp_split / 'images'
        temp_labels_dir = self.temp_data_dir / temp_split / 'labels'
        
        if not original_images_dir.exists():
            print(f"Warning: {original_images_dir} does not exist")
            return
        
        # Get all image files
        image_files = list(original_images_dir.glob('*.jpg')) + list(original_images_dir.glob('*.png'))
        
        print(f"Processing {original_split} split: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"Converting {original_split}"):
            # Copy image
            temp_img_path = temp_images_dir / img_path.name
            shutil.copy2(img_path, temp_img_path)
            
            # Convert corresponding label
            label_path = original_labels_dir / f"{img_path.stem}.txt"
            temp_label_path = temp_labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                self.convert_labels(label_path, temp_label_path)
    
    def generate_standard_data_yaml(self):
        """Generate standard data.yaml for the temp dataset"""
        data_yaml_path = self.temp_data_dir / "data.yaml"
        
        standard_data = {
            'names': self.mapping,
            'nc': len(self.mapping),
            'path': str(self.temp_data_dir),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images'
        }
        
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(standard_data, f, allow_unicode=True)
        
        print(f"Standard data.yaml created at: {data_yaml_path}")
        return data_yaml_path
    
    def create_dataset_stats(self):
        """Create dataset statistics"""
        stats = {}
        splits = ['train', 'val', 'test']
        
        for split in splits:
            labels_dir = self.temp_data_dir / split / 'labels'
            if labels_dir.exists():
                label_files = list(labels_dir.glob('*.txt'))
                total_objects = 0
                class_counts = {class_id: 0 for class_id in self.mapping.keys()}
                
                for label_file in label_files:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    total_objects += len(lines)
                    
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id in class_counts:
                                class_counts[class_id] += 1
                
                stats[split] = {
                    'num_images': len(list((self.temp_data_dir / split / 'images').glob('*.*'))),
                    'num_labels': len(label_files),
                    'total_objects': total_objects,
                    'class_distribution': class_counts
                }
        
        # Save stats
        stats_path = self.temp_data_dir / "dataset_stats.yaml"
        with open(stats_path, 'w', encoding='utf-8') as f:
            yaml.dump(stats, f, allow_unicode=True)
        
        print(f"Dataset statistics saved to: {stats_path}")
        return stats
    
    def run_conversion(self):
        """Run complete dataset conversion"""
        print("=== DATASET WRAPPER - AUTO CONVERSION ===")
        print("Step 1: Creating temporary structure...")
        self.create_temp_structure()
        
        print("Step 2: Converting training data...")
        self.copy_and_convert_images('train', 'train')
        
        print("Step 3: Converting validation data...")  
        self.copy_and_convert_images('val', 'val')
        
        print("Step 4: Converting test data...")
        self.copy_and_convert_images('test', 'test')
        
        print("Step 5: Generating standard data.yaml...")
        self.generate_standard_data_yaml()
        
        print("Step 6: Creating dataset statistics...")
        self.create_dataset_stats()
        
        print("=== CONVERSION COMPLETED ===")
        print(f"Temporary dataset ready at: {self.temp_data_dir}")
        print("Original dataset remains untouched!")

def main():
    """Main function"""
    wrapper = DatasetWrapper()
    wrapper.run_conversion()

if __name__ == "__main__":
    main()