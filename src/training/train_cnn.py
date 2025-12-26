import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import yaml
import sys
import gc
import os

sys.path.append('.')
from src.cnn_classifier import CNNClassifier


# ================= GPU MEMORY SAFETY =================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class CNNTrainer:
    def __init__(self, data_yaml_path="data/processed/data.yaml"):
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        self.num_classes = len(data_config['names'])
        self.classifier = CNNClassifier(self.num_classes)

    def create_data_generators(
        self,
        train_dir,
        val_dir,
        batch_size=8,       
        img_size=224
    ):
        

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )

        return train_generator, val_generator

    def get_callbacks(self, model_save_path="models/cnn"):
        """Create callbacks"""

        callbacks = [
            ModelCheckpoint(
                f"{model_save_path}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=f"{model_save_path}/logs",
                histogram_freq=0   
            )
        ]

        return callbacks

    def train(self, train_dir, val_dir, epochs=50, batch_size=8):
        """Train CNN model"""

        print("Clearing session & memory...")
        tf.keras.backend.clear_session()
        gc.collect()

        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators(
            train_dir,
            val_dir,
            batch_size=batch_size
        )

        print("Building model...")
        self.classifier.build_model()
        self.classifier.model.summary()

        print("Starting training...")
        callbacks = self.get_callbacks()

        history = self.classifier.train(
            train_gen,
            val_gen,
            epochs=epochs,
            callbacks=callbacks
        )

        print("Training completed!")
        return history


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train CNN model for traffic sign classification'
    )
    parser.add_argument(
        '--data-yaml', '-d',
        default='data/processed/data.yaml'
    )
    parser.add_argument(
        '--train-dir', '-t',
        default='data/cnn_processed/train'
    )
    parser.add_argument(
        '--val-dir', '-v',
        default='data/cnn_processed/val'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8   # ↓ an toàn bộ nhớ
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='models/cnn'
    )

    args = parser.parse_args()

    try:
        trainer = CNNTrainer(args.data_yaml)

        print(f"Training directory: {args.train_dir}")
        print(f"Validation directory: {args.val_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")

        history = trainer.train(
            args.train_dir,
            args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        os.makedirs(args.output_dir, exist_ok=True)
        final_model_path = os.path.join(args.output_dir, 'final_model.h5')
        trainer.classifier.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        from src.utils.visualization import Visualizer
        visualizer = Visualizer()
        history_plot_path = os.path.join(
            args.output_dir,
            'training_history.png'
        )
        visualizer.plot_training_history(
            history,
            save_path=history_plot_path
        )
        print(f"Training history plot saved to: {history_plot_path}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
