"""
Auto mapping system for traffic sign dataset
Step 1: Create consistent class mapping with standardized names
"""

import yaml
import os
from pathlib import Path
import shutil
from collections import defaultdict
import json

class AutoMapper:
    def __init__(self, data_yaml_path="data/processed/data.yaml"):
        self.data_yaml_path = data_yaml_path
        self.original_classes = {}
        self.mapped_classes = {}
        self.class_groups = defaultdict(list)
        
    def load_original_classes(self):
        """Load original classes from data.yaml"""
        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        self.original_classes = data_config['names']
        print(f"Loaded {len(self.original_classes)} original classes")
        return self.original_classes
    
    def analyze_class_patterns(self):
        """Analyze class names to identify patterns"""
        pattern_categories = {
            'W': 'warning_signs',
            'P': 'prohibition_signs', 
            'R': 'regulation_signs',
            'I': 'information_signs',
            'S': 'supplementary_signs',
            'B': 'additional_signs',
            'Camera': 'camera_signs'
        }
        
        for class_id, class_name in self.original_classes.items():
            # Detect pattern category
            category = 'other'
            for prefix, cat_name in pattern_categories.items():
                if class_name.startswith(prefix):
                    category = cat_name
                    break
            
            self.class_groups[category].append((class_id, class_name))
        
        print("Class analysis completed:")
        for category, classes in self.class_groups.items():
            print(f"  {category}: {len(classes)} classes")
    
    def create_standardized_mapping(self):
        """Create standardized mapping with consistent names"""
        mapping = {}
        reverse_mapping = {}
        
        for class_id, class_name in self.original_classes.items():
            # Create standardized name
            std_name = class_name.replace('*', '_').replace(' ', '_').replace('/', '_')
            std_name = f"class_{class_id:02d}_{std_name}"
            
            mapping[class_id] = {
                'original_name': class_name,
                'standardized_name': std_name,
                'short_name': f"c{class_id:02d}"
            }
            reverse_mapping[std_name] = class_id
        
        self.mapped_classes = mapping
        return mapping
    
    def generate_mapping_files(self, output_dir="data/mappings"):
        """Generate mapping files for future use"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mapping as YAML
        mapping_data = {
            'original_to_standard': self.mapped_classes,
            'standard_to_original': {v['standardized_name']: k for k, v in self.mapped_classes.items()}
        }
        
        with open(f"{output_dir}/class_mapping.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(mapping_data, f, allow_unicode=True)
        
        # Save mapping as JSON
        with open(f"{output_dir}/class_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        # Save simplified mapping for training
        simplified_mapping = {
            class_id: info['standardized_name'] 
            for class_id, info in self.mapped_classes.items()
        }
        
        with open(f"{output_dir}/simplified_mapping.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(simplified_mapping, f, allow_unicode=True)
        
        print(f"Mapping files saved to {output_dir}/")
        return simplified_mapping
    
    def create_standard_data_yaml(self, output_path="data/standard/data.yaml"):
        """Create standardized data.yaml file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        standard_data = {
            'names': {class_id: info['standardized_name'] 
                     for class_id, info in self.mapped_classes.items()},
            'nc': len(self.mapped_classes),
            'path': os.path.dirname(output_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images'
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(standard_data, f, allow_unicode=True)
        
        print(f"Standard data.yaml created at: {output_path}")
        return standard_data
    
    def run_full_mapping(self):
        """Run complete mapping process"""
        print("=== AUTO MAPPING SYSTEM ===")
        print("Step 1: Loading original classes...")
        self.load_original_classes()
        
        print("Step 2: Analyzing class patterns...")
        self.analyze_class_patterns()
        
        print("Step 3: Creating standardized mapping...")
        self.create_standardized_mapping()
        
        print("Step 4: Generating mapping files...")
        self.generate_mapping_files()
        
        print("Step 5: Creating standard data.yaml...")
        self.create_standard_data_yaml()
        
        print("=== MAPPING COMPLETED ===")
        return self.mapped_classes

def main():
    """Main function"""
    mapper = AutoMapper()
    mapping = mapper.run_full_mapping()
    
    print("\nMapping preview (first 10 classes):")
    for class_id in range(min(10, len(mapping))):
        info = mapping[class_id]
        print(f"  {class_id:2d}: {info['original_name']} -> {info['standardized_name']}")

if __name__ == "__main__":
    main()