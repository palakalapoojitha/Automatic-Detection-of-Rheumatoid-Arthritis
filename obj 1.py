import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
tf.random.set_seed(42)

class RheoNetCNN:
    """
    RheoNet-CNN: An Enhanced Deep Learning Framework for Automatic
    Detection of Rheumatoid Arthritis Using Hand X-ray Images
    """
    
    def __init__(self, input_shape=(224, 224, 1), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the RheoNet-CNN architecture with residual connections,
        batch normalization, and dropout layers
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # First Convolutional Block
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        bn1 = layers.BatchNormalization()(conv1)
        dropout1 = layers.Dropout(0.3)(bn1)
        pool1 = layers.MaxPooling2D((2, 2))(dropout1)
        
        # Second Convolutional Block
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        bn2 = layers.BatchNormalization()(conv2)
        dropout2 = layers.Dropout(0.3)(bn2)
        pool2 = layers.MaxPooling2D((2, 2))(dropout2)
        
        # Third Convolutional Block
        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        bn3 = layers.BatchNormalization()(conv3)
        dropout3 = layers.Dropout(0.3)(bn3)
        pool3 = layers.MaxPooling2D((2, 2))(dropout3)
        
        # Residual Connection (skip connection from first block to third block)
        # Resize pool1 to match pool3 dimensions
        pool1_resized = layers.Conv2D(128, (1, 1), padding='same')(pool1)
        pool1_resized = layers.MaxPooling2D((4, 4))(pool1_resized)
        
        # Add residual connection
        residual = layers.Add()([pool3, pool1_resized])
        
        # Flatten and Dense layers
        flatten = layers.Flatten()(residual)
        
        # Fully Connected Layer 1
        dense1 = layers.Dense(256, activation='relu')(flatten)
        dropout4 = layers.Dropout(0.3)(dense1)
        
        # Fully Connected Layer 2
        dense2 = layers.Dense(128, activation='relu')(dropout4)
        dropout5 = layers.Dropout(0.3)(dense2)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='sigmoid')(dropout5)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='RheoNet-CNN')
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return self.model
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")

class DataPreprocessor:
    """Data preprocessing and augmentation class"""
    
    def __init__(self, base_path, img_size=(224, 224), batch_size=32):
        self.base_path = base_path
        self.img_size = img_size
        self.batch_size = batch_size
        
    def create_data_generators(self):
        """Create training and validation data generators with augmentation"""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest',
            validation_split=0.15  # 15% for validation
        )
        
        # Test data preprocessing (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.base_path, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            subset='training',
            seed=42
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.base_path, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            subset='validation',
            seed=42
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

class ModelTrainer:
    """Model training and evaluation class"""
    
    def __init__(self, model):
        self.model = model
        self.history = None
        
    def get_callbacks(self):
        """Define training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_rheonet_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train_model(self, train_gen, val_gen, epochs=50):
        """Train the model"""
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall plot
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

class ModelEvaluator:
    """Model evaluation and analysis class"""
    
    def __init__(self, model):
        self.model = model
        
    def evaluate_model(self, test_generator):
        """Comprehensive model evaluation"""
        # Get predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = (predictions > 0.5).astype(int)
        
        # Get true labels
        true_labels = test_generator.classes
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes.flatten() == true_labels)
        
        # Classification report
        report = classification_report(
            true_labels, 
            predicted_classes,
            target_names=['Normal', 'RA'],
            output_dict=True
        )
        
        # ROC curve
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        results = {
            'accuracy': accuracy,
            'precision': report['RA']['precision'],
            'recall': report['RA']['recall'],
            'f1_score': report['RA']['f1-score'],
            'auc_roc': roc_auc,
            'classification_report': report,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_labels': true_labels,
            'fpr': fpr,
            'tpr': tpr
        }
        
        return results
    
    def plot_confusion_matrix(self, true_labels, predicted_classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predicted_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'RA'],
                   yticklabels=['Normal', 'RA'])
        plt.title('Confusion Matrix - RheoNet-CNN')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, fpr, tpr, auc_score):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'RheoNet-CNN (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - RheoNet-CNN')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def print_detailed_results(self, results):
        """Print detailed evaluation results"""
        print("="*60)
        print("RheoNet-CNN MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"Precision: {results['precision']:.3f} ({results['precision']*100:.1f}%)")
        print(f"Recall:    {results['recall']:.3f} ({results['recall']*100:.1f}%)")
        print(f"F1-Score:  {results['f1_score']:.3f} ({results['f1_score']*100:.1f}%)")
        print(f"AUC-ROC:   {results['auc_roc']:.3f} ({results['auc_roc']*100:.1f}%)")
        print("="*60)

def perform_statistical_analysis(rheonet_results, baseline_results_dict):
    """Perform statistical significance tests"""
    print("\nSTATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*50)
    
    rheonet_scores = [
        rheonet_results['accuracy'],
        rheonet_results['precision'],
        rheonet_results['recall'],
        rheonet_results['f1_score'],
        rheonet_results['auc_roc']
    ]
    
    for model_name, baseline_results in baseline_results_dict.items():
        baseline_scores = [
            baseline_results['accuracy'],
            baseline_results['precision'],
            baseline_results['recall'],
            baseline_results['f1_score'],
            baseline_results['auc_roc']
        ]
        
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(rheonet_scores, baseline_scores)
        print(f"\nRheoNet-CNN vs {model_name}:")
        print(f"Wilcoxon signed-rank test p-value: {p_value:.2e}")
        
        if p_value < 0.05:
            print("✓ Statistically significant improvement")
        else:
            print("✗ Not statistically significant")

def main():
    """Main execution function"""
    
    # Configuration
    BASE_PATH = r"D:\poojitha"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    
    print("RheoNet-CNN: Enhanced Deep Learning Framework for RA Detection")
    print("="*65)
    
    # Initialize components
    print("\n1. Initializing RheoNet-CNN model...")
    rheonet = RheoNetCNN(input_shape=(224, 224, 1))
    model = rheonet.build_model()
    model = rheonet.compile_model(learning_rate=0.001)
    
    print("Model architecture:")
    rheonet.get_model_summary()
    
    # Data preprocessing
    print("\n2. Setting up data preprocessing...")
    preprocessor = DataPreprocessor(BASE_PATH, IMG_SIZE, BATCH_SIZE)
    train_gen, val_gen, test_gen = preprocessor.create_data_generators()
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    
    # Model training
    print(f"\n3. Training RheoNet-CNN for {EPOCHS} epochs...")
    trainer = ModelTrainer(model)
    history = trainer.train_model(train_gen, val_gen, epochs=EPOCHS)
    
    # Plot training history
    print("\n4. Plotting training history...")
    trainer.plot_training_history()
    
    # Model evaluation
    print("\n5. Evaluating model performance...")
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate_model(test_gen)
    
    # Display results
    evaluator.print_detailed_results(results)
    
    # Plot visualizations
    print("\n6. Generating evaluation plots...")
    evaluator.plot_confusion_matrix(results['true_labels'], 
                                   results['predicted_classes'])
    evaluator.plot_roc_curve(results['fpr'], results['tpr'], 
                            results['auc_roc'])
    
    # Save model
    print("\n7. Saving trained model...")
    model.save('rheonet_cnn_final.h5')
    print("Model saved as 'rheonet_cnn_final.h5'")
    
    print("\n" + "="*65)
    print("RheoNet-CNN training and evaluation completed successfully!")
    print("="*65)
    
    return model, results

if __name__ == "__main__":
    # Run the main function
    trained_model, evaluation_results = main()
    
   
