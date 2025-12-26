
from ultralytics import YOLO
import yaml
from pathlib import Path

class YOLOTrainer:
    def __init__(self, config_path="config/yolo_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def train(self):
        # Load model
        model_name = self.config['model']['name']
        model = YOLO(f"{model_name}.pt" if self.config['model']['pretrained'] else model_name)
        
        # Training parameters
        train_params = self.config['training']
        paths = self.config['paths']
        
        # Train
        results = model.train(
            data=str(Path(paths['dataset']) / "data.yaml"),
            epochs=train_params['epochs'],
            imgsz=train_params['img_size'],
            batch=train_params['batch_size'],
            device=train_params['device'],
            workers=train_params['workers'],
            patience=train_params['patience'],
            save=True,
            project=paths['output'],
            name='train',
            exist_ok=True,
            
            # Optimizer
            optimizer=train_params['optimizer']['name'],
            lr0=train_params['optimizer']['lr'],
            weight_decay=train_params['optimizer']['weight_decay'],
            
            # Augmentation
            hsv_h=train_params['augmentation']['hsv_h'],
            hsv_s=train_params['augmentation']['hsv_s'],
            hsv_v=train_params['augmentation']['hsv_v'],
            degrees=train_params['augmentation']['degrees'],
            translate=train_params['augmentation']['translate'],
            scale=train_params['augmentation']['scale'],
            shear=train_params['augmentation']['shear'],
            perspective=train_params['augmentation']['perspective'],
            flipud=train_params['augmentation']['flipud'],
            fliplr=train_params['augmentation']['fliplr'],
            mosaic=train_params['augmentation']['mosaic'],
        )
        
        return results
    
    def evaluate(self, model_path):
        """Evaluate model"""
        model = YOLO(model_path)
        
        paths = self.config['paths']
        metrics = model.val(
            data=str(Path(paths['dataset']) / "data.yaml"),
            split='test'
        )
        
        return metrics

def main():
    """Main entry point for YOLO training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for traffic sign detection')
    parser.add_argument('--config', '-c', default='config/yolo_config.yaml',
                       help='Path to YOLO configuration file')
    parser.add_argument('--eval', '-e', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--model-path', '-m', default=None,
                       help='Path to model for evaluation (if different from config)')
    
    args = parser.parse_args()
    
    try:
        trainer = YOLOTrainer(args.config)
        print("Starting YOLOv8 training...")
        
        # Train model
        results = trainer.train()
        print("Training completed!")
        
        # Evaluate if requested
        if args.eval:
            model_path = args.model_path or trainer.config['paths']['output'] + '/train/weights/best.pt'
            print(f"Evaluating model: {model_path}")
            metrics = trainer.evaluate(model_path)
            print("Evaluation completed!")
            print(f"mAP: {metrics.box.map}")
            print(f"mAP50: {metrics.box.map50}")
            print(f"mAP75: {metrics.box.map75}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()