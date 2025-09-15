import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
import shutil
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ======== STEP 1: MODEL CLASS =========
class FastRheoNetCNN:
    def __init__(self, input_shape=(128, 128, 1), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_fast_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # Block 1
        x = layers.Conv2D(32, (5, 5), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.1)(x)

        # Block 2
        x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)

        # Block 3
        x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)

        # Block 4
        x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Pooling & Dense Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(self.num_classes, activation='sigmoid')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name='Fast-RheoNet-CNN')
        return self.model

    def compile_model(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        loss = BinaryFocalCrossentropy(alpha=0.7, gamma=2.0)

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
        return self.model


# ======== STEP 2: DATA PREPROCESSOR (Balanced) =========
class FastDataPreprocessor:
    def __init__(self, base_path, img_size=(128, 128), batch_size=64):
        self.base_path = base_path
        self.img_size = img_size
        self.batch_size = batch_size

    def balance_train_data(self):
        """
        Oversample minority class in the train folder to match majority class count.
        """
        train_path = os.path.join(self.base_path, 'train')
        class_folders = [os.path.join(train_path, c) for c in os.listdir(train_path)]
        counts = {cls: len(os.listdir(cls)) for cls in class_folders}

        print(f"[INFO] Class counts before balancing: {counts}")
        max_count = max(counts.values())

        for cls_path, count in counts.items():
            if count < max_count:
                files = os.listdir(cls_path)
                diff = max_count - count
                for i in range(diff):
                    f = random.choice(files)
                    src = os.path.join(cls_path, f)
                    new_name = f"copy_{i}_{f}"
                    dst = os.path.join(cls_path, new_name)
                    shutil.copy(src, dst)

        counts_after = {cls: len(os.listdir(cls)) for cls in class_folders}
        print(f"[INFO] Class counts after balancing: {counts_after}")

    def create_fast_generators(self):
        # First balance the train data
        self.balance_train_data()

        # Augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            validation_split=0.2
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Train generator
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.base_path, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            subset='training',
            seed=42,
            shuffle=True
        )

        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.base_path, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            subset='validation',
            seed=42,
            shuffle=True
        )

        # Test generator
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.base_path, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False
        )

        return train_generator, validation_generator, test_generator


# ======== STEP 3: TRAINER =========
class FastTrainer:
    def __init__(self, model):
        self.model = model
        self.history = None

    def get_fast_callbacks(self):
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
            ModelCheckpoint('fast_rheonet_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        return callbacks

    def train_fast(self, train_gen, val_gen, epochs=30):
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

        callbacks = self.get_fast_callbacks()

        print(f"Training steps per epoch: {len(train_gen)}")
        print(f"Validation steps: {len(val_gen)}")

        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        return self.history

    def plot_training_history(self):
        if not self.history:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history.history['accuracy'], 'b-', label='Training')
        axes[0].plot(self.history.history['val_accuracy'], 'r-', label='Validation')
        axes[0].set_title('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history.history['loss'], 'b-', label='Training')
        axes[1].plot(self.history.history['val_loss'], 'r-', label='Validation')
        axes[1].set_title('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ======== STEP 4: EVALUATOR =========
class FastEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_fast(self, test_generator):
        predictions = self.model.predict(test_generator)
        predicted_classes = (predictions > 0.5).astype(int)
        true_labels = test_generator.classes

        accuracy = np.mean(predicted_classes.flatten() == true_labels)
        report = classification_report(
            true_labels, predicted_classes,
            target_names=['Normal', 'RA'],
            output_dict=True, zero_division=0
        )

        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)

        results = {
            'accuracy': accuracy,
            'precision': report['RA']['precision'],
            'recall': report['RA']['recall'],
            'f1_score': report['RA']['f1-score'],
            'auc_roc': roc_auc,
            'predictions': predictions,
            'true_labels': true_labels
        }
        return results

    def plot_results(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        cm = confusion_matrix(results['true_labels'], (results['predictions'] > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'RA'], yticklabels=['Normal', 'RA'], ax=ax1)
        ax1.set_title('Confusion Matrix')

        fpr, tpr, _ = roc_curve(results['true_labels'], results['predictions'])
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {results["auc_roc"]:.3f}')
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def print_results(self, results):
        print("="*60)
        print("FAST RheoNet-CNN RESULTS")
        print("="*60)
        print(f"Accuracy:  {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"Precision: {results['precision']:.3f} ({results['precision']*100:.1f}%)")
        print(f"Recall:    {results['recall']:.3f} ({results['recall']*100:.1f}%)")
        print(f"F1-Score:  {results['f1_score']:.3f} ({results['f1_score']*100:.1f}%)")
        print(f"AUC-ROC:   {results['auc_roc']:.3f} ({results['auc_roc']*100:.1f}%)")
        print("="*60)


# ======== STEP 5: MAIN =========
def fast_main():
    BASE_PATH = r"D:\poojitha"
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 64
    EPOCHS = 25

    print("\n1. Building Fast RheoNet-CNN...")
    fast_model = FastRheoNetCNN(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    model = fast_model.build_fast_model()
    model = fast_model.compile_model(learning_rate=0.002)

    print(f"Model parameters: {model.count_params():,}")

    print("\n2. Setting up balanced data preprocessing...")
    preprocessor = FastDataPreprocessor(BASE_PATH, IMG_SIZE, BATCH_SIZE)
    train_gen, val_gen, test_gen = preprocessor.create_fast_generators()

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")

    print(f"\n3. Fast training for {EPOCHS} epochs...")
    trainer = FastTrainer(model)
    history = trainer.train_fast(train_gen, val_gen, epochs=EPOCHS)

    print("\n4. Fast evaluation...")
    evaluator = FastEvaluator(model)
    results = evaluator.evaluate_fast(test_gen)
    evaluator.print_results(results)
    evaluator.plot_results(results)
    trainer.plot_training_history()

    model.save('fast_rheonet_final_balanced.h5')
    print("\nModel saved as 'fast_rheonet_final_balanced.h5'")
    return model, results


if __name__ == "__main__":
    fast_model, fast_results = fast_main()
