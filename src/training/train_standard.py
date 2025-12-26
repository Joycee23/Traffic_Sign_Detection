"""
Standard Training Script for Traffic Sign Classification
Step 5: Train with standardized dataset (100% CHUẨN)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import yaml
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')
from src.cnn_classifier import CNNClassifier

class StandardTrainer:
    def __init__(self, data_yaml_path="data/temp_standard/data.yaml"):
        """Initialize trainer with standardized dataset"""
        self.data_yaml_path = data_yaml_path
        
        # Load data configuration
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.num_classes = len(self.data_config['names'])
        self.class_names = self.data_config['names']
        
        print(f"Training with {self.num_classes} standardized classes")
        self.classifier = CNNClassifier(self.num_classes)
    
    def create_data_generators(self, batch_size=32, img_size=224):
        """Create data generators with augmentation"""
        train_dir = os.path.join(self.data_config['path'], self.data_config['train'])
        val_dir = os.path.join(self.data_config['path'], self.data_config['val'])
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        print(f"Training directory: {train_dir}")
        print(f"Validation directory: {val_dir}")
        
        train_generator = train_datagen.flow_from_directory(
            os.path.dirname(train_dir),  # Parent directory of images
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.dirname(val_dir),  # Parent directory of images  
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def get_callbacks(self, output_dir="models/standard_cnn"):
        """Create training callbacks"""
        os.makedirs(output_dir, exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                f"{output_dir}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=f"{output_dir}/logs",
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(self, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the model with standardized dataset"""
        print("=== STANDARD TRAINING STARTED ===")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(batch_size)
        
        # Build model
        print("Building model architecture...")
        self.classifier.build_model(learning_rate=learning_rate)
        self.classifier.model.summary()
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        print("Starting training process...")
        history = self.classifier.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def save_training_report(self, history, output_dir="models/standard_cnn"):
        """Save training report and metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final model
        final_model_path = f"{output_dir}/final_model.h5"
        self.classifier.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save training history
        history_path = f"{output_dir}/training_history.npy"
        np.save(history_path, history.history)
        print(f"Training history saved to: {history_path}")
        
        # Create training report
        report = {
            'training_date': datetime.now().isoformat(),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_accuracy': max(history.history['val_accuracy']),
            'best_accuracy': max(history.history['accuracy']),
            'total_epochs': len(history.history['accuracy'])
        }
        
        report_path = f"{output_dir}/training_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f)
        
        print(f"Training report saved to: {report_path}")
        return report
    
    def plot_training_history(self, history, output_dir="models/standard_cnn"):
        """Plot training history graphs"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Accuracy plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plot_path = f"{output_dir}/training_history.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Training plots saved to: {plot_path}")
        return plot_path

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN with standardized dataset')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-yaml', '-d', default='data/temp_standard/data.yaml', help='Path to data.yaml')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = StandardTrainer(args.data_yaml)
        
        # Start training
        history = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Save results
        report = trainer.save_training_report(history)
        trainer.plot_training_history(history)
        
        print("\n=== TRAINING SUMMARY ===")
        print(f"Best Validation Accuracy: {report['best_val_accuracy']:.4f}")
        print(f"Final Validation Accuracy: {report['final_val_accuracy']:.4f}")
        print(f"Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()