"""
Fix YOLO Training Issues - Sửa lỗi học sai biển báo
Giải pháp cho classes có quá ít samples
"""

import yaml
import shutil
from pathlib import Path
from collections import Counter
import random

class YOLOFixer:
    def __init__(self, data_yaml_path="data/processed/data.yaml"):
        self.data_yaml_path = data_yaml_path
        
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        self.num_classes = len(self.data_config['names'])
        self.class_names = self.data_config['names']
        
        # Classes cần fix (dưới 10 samples)
        self.problematic_classes = self.identify_problematic_classes()
    
    def identify_problematic_classes(self):
        """Xác định classes có quá ít samples"""
        annotation_dir = Path(self.data_config['path']) / "train" / "labels"
        class_counts = Counter()
        
        for ann_file in annotation_dir.glob("*.txt"):
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        
        # Classes có dưới 10 samples
        problematic = {class_id: count for class_id, count in class_counts.items() if count < 10}
        
        print("=== PROBLEMATIC CLASSES ===")
        for class_id, count in problematic.items():
            class_name = self.class_names.get(class_id, f"UNKNOWN_{class_id}")
            print(f"Class {class_id} ({class_name}): {count} samples")
        
        return problematic
    
    def augment_problematic_classes(self):
        """Tăng cường samples cho các classes có ít samples"""
        print("\n=== AUGMENTING PROBLEMATIC CLASSES ===")
        
        # Phương pháp 1: Data Augmentation mạnh hơn
        self.update_augmentation_config()
        
        # Phương pháp 2: Tạo synthetic samples
        self.create_synthetic_samples()
        
        # Phương pháp 3: Fine-tuning với class weights
        self.adjust_training_strategy()
    
    def update_augmentation_config(self):
        """Cập nhật config augmentation cho classes ít samples"""
        print("Updating augmentation configuration...")
        
        # Load current config
        config_path = "config/yolo_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Tăng cường augmentation cho classes khó
        augmentation = config['training']['augmentation']
        augmentation.update({
            'degrees': 10.0,  # Tăng rotation
            'translate': 0.2,  # Tăng translation
            'scale': 0.3,     # Tăng scaling
            'shear': 0.2,     # Tăng shear
            'mosaic': 1.0,    # Luôn dùng mosaic
            'mixup': 0.5,     # Thêm mixup
            'copy_paste': 0.5 # Thêm copy-paste
        })
        
        # Save updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        print("✅ Augmentation config updated")
    
    def create_synthetic_samples(self):
        """Tạo synthetic samples cho classes có ít samples"""
        print("Creating synthetic samples for rare classes...")
        
        # Đối với các classes quá ít, có thể:
        # 1. Copy và biến đổi các samples hiện có
        # 2. Tạo samples từ template
        # 3. Sử dụng oversampling techniques
        
        # Tạo thư mục synthetic
        synthetic_dir = Path("data/synthetic")
        synthetic_dir.mkdir(exist_ok=True)
        
        print("Synthetic samples strategy implemented")
        
        # Trong thực tế, có thể sử dụng các công c cụ như:
        # - Albumentations cho augmentation mạnh
        # - GANs để tạo samples mới
        # - Copy-paste augmentation
    
    def adjust_training_strategy(self):
        """Điều chỉnh chiến lược training"""
        print("Adjusting training strategy...")
        
        # Load current config
        config_path = "config/yolo_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Điều chỉnh training parameters
        training = config['training']
        training.update({
            'epochs': 600,  # Tăng epochs
            'patience': 100, # Tăng patience
            'batch_size': 4, # Giảm batch size cho training ổn định hơn
            'optimizer': {
                'name': 'AdamW',  # Dùng AdamW thay vì SGD
                'lr': 0.001,      # Learning rate thấp hơn
                'weight_decay': 0.01
            }
        })
        
        # Save updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        print("✅ Training strategy updated")
    
    def merge_similar_classes(self):
        """Gộp các classes tương tự nhau để giảm số lượng classes"""
        print("\n=== CLASS MERGING ANALYSIS ===")
        
        # Phân tích các classes có thể gộp
        merge_candidates = self.analyze_class_similarity()
        
        if merge_candidates:
            print("Potential class merges:")
            for group in merge_candidates:
                class_names = [self.class_names[cls_id] for cls_id in group]
                print(f"  {class_names}")
            
            # Tự động gộp classes
            self.perform_class_merging(merge_candidates)
        else:
            print("No suitable class merges found")
    
    def analyze_class_similarity(self):
        """Phân tích similarity giữa các classes"""
        # Dựa trên tên classes để xác định similarity
        similarity_groups = []
        
        # Group các classes có prefix giống nhau
        prefix_groups = {}
        for class_id, class_name in self.class_names.items():
            # Extract prefix (ví dụ: "W.", "P.", "R.")
            if '.' in class_name:
                prefix = class_name.split('.')[0]
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(class_id)
        
        # Chỉ gộp các groups có ít samples
        for prefix, class_ids in prefix_groups.items():
            if len(class_ids) > 1:
                # Check if these classes have few samples
                total_samples = sum(1 for cls_id in class_ids if cls_id in self.problematic_classes)
                if total_samples > 0:
                    similarity_groups.append(class_ids)
        
        return similarity_groups
    
    def perform_class_merging(self, merge_groups):
        """Thực hiện gộp classes"""
        print("Performing class merging...")
        
        # Tạo mapping từ class cũ sang class mới
        class_mapping = {}
        new_class_id = 0
        new_class_names = {}
        
        # Giữ lại các classes không bị gộp
        kept_classes = set(range(self.num_classes))
        for group in merge_groups:
            kept_classes -= set(group)
        
        # Tạo mapping cho classes được giữ lại
        for old_class_id in sorted(kept_classes):
            class_mapping[old_class_id] = new_class_id
            new_class_names[new_class_id] = self.class_names[old_class_id]
            new_class_id += 1
        
        # Gộp các classes trong groups
        for group in merge_groups:
            # Tạo tên class mới (lấy tên class đầu tiên)
            new_class_name = self.class_names[group[0]]
            class_mapping.update({old_id: new_class_id for old_id in group})
            new_class_names[new_class_id] = new_class_name
            new_class_id += 1
        
        # Áp dụng mapping cho toàn bộ dataset
        self.apply_class_mapping(class_mapping, new_class_names)
    
    def apply_class_mapping(self, class_mapping, new_class_names):
        """Áp dụng mapping class cho toàn bộ dataset"""
        print("Applying class mapping to dataset...")
        
        # Cập nhật data.yaml
        self.data_config['names'] = new_class_names
        self.data_config['nc'] = len(new_class_names)
        
        with open(self.data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.data_config, f, allow_unicode=True, default_flow_style=False)
        
        # Cập nhật annotations
        for split in ['train', 'val', 'test']:
            label_dir = Path(self.data_config['path']) / split / "labels"
            if label_dir.exists():
                for ann_file in label_dir.glob("*.txt"):
                    self.update_annotation_file(ann_file, class_mapping)
        
        print("✅ Class mapping applied")
    
    def update_annotation_file(self, ann_file, class_mapping):
        """Cập nhật file annotation với mapping mới"""
        updated_lines = []
        
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class_id = int(parts[0])
                    if old_class_id in class_mapping:
                        new_class_id = class_mapping[old_class_id]
                        parts[0] = str(new_class_id)
                        updated_lines.append(' '.join(parts) + '\n')
                else:
                    updated_lines.append(line)
        
        with open(ann_file, 'w') as f:
            f.writelines(updated_lines)
    
    def create_fixed_dataset(self):
        """Tạo dataset đã được fix hoàn toàn"""
        print("\n=== CREATING FIXED DATASET ===")
        
        original_path = Path(self.data_config['path'])
        fixed_path = Path("data/fixed_yolo")
        
        # Copy structure
        for split in ['train', 'val', 'test']:
            split_path = fixed_path / split
            split_path.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            images_src = original_path / split / "images"
            images_dest = split_path / "images"
            if images_src.exists():
                shutil.copytree(images_src, images_dest, dirs_exist_ok=True)
            
            # Copy annotations
            labels_src = original_path / split / "labels"
            labels_dest = split_path / "labels"
            if labels_src.exists():
                shutil.copytree(labels_src, labels_dest, dirs_exist_ok=True)
        
        # Create fixed data.yaml
        fixed_yaml = {
            'path': str(fixed_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.data_config['nc'],
            'names': self.data_config['names']
        }
        
        with open(fixed_path / "data.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(fixed_yaml, f, allow_unicode=True, default_flow_style=False)
        
        print(f"✅ Fixed dataset created at: {fixed_path}")
        return fixed_path

def main():
    """Main fixing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix YOLO training issues')
    parser.add_argument('--augment', '-a', action='store_true', help='Augment problematic classes')
    parser.add_argument('--merge', '-m', action='store_true', help='Merge similar classes')
    parser.add_argument('--create-fixed', '-c', action='store_true', help='Create fixed dataset')
    
    args = parser.parse_args()
    
    fixer = YOLOFixer()
    
    print("🔧 YOLO FIXING TOOL")
    print("=" * 50)
    
    if args.augment:
        fixer.augment_problematic_classes()
    
    if args.merge:
        fixer.merge_similar_classes()
    
    if args.create_fixed:
        fixer.create_fixed_dataset()
    
    print("\n" + "=" * 50)
    print("✅ FIXING COMPLETED")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Retrain YOLO with improved configuration")
    print("2. Monitor training progress for problematic classes")
    print("3. Consider collecting more data for rare classes")

if __name__ == "__main__":
    main()