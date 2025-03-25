import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants for meningioma-only fine-tuning
IMG_SIZE = 160
BATCH_SIZE = 16  # Smaller batch size for more focused learning
EPOCHS = 15  # Focused training epochs
LEARNING_RATE = 0.00005  # Very low learning rate to avoid catastrophic forgetting
TRAIN_DIR = 'Training'
TEST_DIR = 'Testing'
ORIGINAL_MODEL_PATH = 'app/models/brain_tumor_classifier.h5'
MENINGIOMA_MODEL_PATH = 'app/models/meningioma_only_model.h5'

# Define class names and meningioma index
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MENINGIOMA_INDEX = 1  # Index of meningioma in CLASS_NAMES
MENINGIOMA_FOLDER = 'meningioma'  # Folder name for meningioma in dataset

def create_meningioma_only_generators():
    """Create data generators focusing only on meningioma images."""
    print("Creating meningioma-focused data generators...")
    
    # Define enhanced augmentation specific to meningioma characteristics
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,           # More rotation for shape variations
        width_shift_range=0.3,       # Larger shifts
        height_shift_range=0.3,      # Larger shifts
        shear_range=0.25,            # More shear for shape variations
        zoom_range=0.3,              # More zoom variations
        horizontal_flip=True,
        vertical_flip=False,         # MRI orientation matters
        brightness_range=[0.7, 1.3], # Higher brightness variation
        fill_mode='nearest',
        validation_split=0.2         # 20% validation split
    )
    
    # Only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create class subset - use only meningioma + a small subset of other classes
    # to maintain ability to classify other classes
    class_subset = [MENINGIOMA_FOLDER]
    
    # Create training generator for meningioma only
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=class_subset,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator for meningioma only
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=class_subset,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # For testing, we need all classes to evaluate overall performance
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Found {train_generator.samples} meningioma training samples")
    print(f"Found {validation_generator.samples} meningioma validation samples")
    print(f"Found {test_generator.samples} total test samples across all classes")
    
    return train_generator, validation_generator, test_generator

def load_and_modify_model_for_meningioma(model_path):
    """Load existing model and prepare it for meningioma-only training."""
    print(f"Loading original model from {model_path}...")
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
        
        # Save original weights of the output layer
        original_output_weights = model.layers[-1].get_weights()
        
        # Create a new architecture specifically for meningioma training
        # Use a branch network approach to preserve original capabilities
        
        # Get the base model without the final classification layer
        base_model = model
        for layer in base_model.layers:
            # Freeze all layers except the last few to preserve knowledge
            layer.trainable = False
        
        # Unfreeze just the last 10 layers for fine tuning
        for layer in base_model.layers[-10:]:
            layer.trainable = True
        
        # Clone the model to preserve it
        meningioma_model = tf.keras.models.clone_model(model)
        meningioma_model.set_weights(model.get_weights())
        
        # Compile with very low learning rate
        meningioma_model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model prepared for meningioma-only training.")
        return meningioma_model, model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def train_meningioma_only(model, train_generator, validation_generator):
    """Train the model specifically for meningioma detection."""
    print("Starting meningioma-only training...")
    
    # Define callbacks for training
    checkpoint = ModelCheckpoint(
        MENINGIOMA_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model focused only on meningioma
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    return history, model

def evaluate_meningioma_performance(model, test_generator):
    """Evaluate the model's performance on all classes but focus on meningioma."""
    print("Evaluating model performance...")
    
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Generate the classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True
    )
    
    # Print overall performance
    print("\nClassification Report:")
    for class_name in CLASS_NAMES:
        print(f"{class_name}:")
        print(f"  Precision: {report[class_name]['precision']:.4f}")
        print(f"  Recall: {report[class_name]['recall']:.4f}")
        print(f"  F1-score: {report[class_name]['f1-score']:.4f}")
    
    print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Focus on meningioma metrics
    meningioma_precision = report['meningioma']['precision']
    meningioma_recall = report['meningioma']['recall']
    meningioma_f1 = report['meningioma']['f1-score']
    
    print(f"\nMeningioma-Specific Metrics:")
    print(f"  Precision: {meningioma_precision:.4f}")
    print(f"  Recall: {meningioma_recall:.4f}")
    print(f"  F1-score: {meningioma_f1:.4f}")
    
    # Analyze meningioma confusion specifically
    meningioma_idx = CLASS_NAMES.index('meningioma')
    meningioma_total = sum(cm[meningioma_idx, :])
    meningioma_correct = cm[meningioma_idx, meningioma_idx]
    meningioma_accuracy = meningioma_correct / meningioma_total
    
    print(f"Meningioma Recognition Rate: {meningioma_accuracy:.4f} ({meningioma_correct}/{meningioma_total})")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix After Meningioma-Only Training')
    plt.savefig('app/static/meningioma_only_confusion_matrix.png')
    plt.close()
    
    return meningioma_f1, report

def combine_models(original_model, meningioma_model):
    """Create a final model that uses the improved meningioma detection while preserving other classes."""
    print("Creating combined model with improved meningioma detection...")
    
    # Strategy: We'll use the entire meningioma model but recover weights from the 
    # original model for the non-meningioma classes
    
    # First, get the output weights from both models
    original_weights = original_model.layers[-1].get_weights()
    improved_weights = meningioma_model.layers[-1].get_weights()
    
    # Create the combined weights by using meningioma weights from the improved model
    # and keeping other class weights from the original model
    combined_weights = original_weights.copy()
    
    # Replace only the meningioma neuron weights with the improved version
    # This is the core of our approach - only affecting meningioma classification
    combined_weights[0][:, MENINGIOMA_INDEX] = improved_weights[0][:, MENINGIOMA_INDEX]
    combined_weights[1][MENINGIOMA_INDEX] = improved_weights[1][MENINGIOMA_INDEX]
    
    # Create the final model (a clone of the original)
    final_model = tf.keras.models.clone_model(original_model)
    final_model.set_weights(original_model.get_weights())
    
    # Set the customized output layer weights
    final_model.layers[-1].set_weights(combined_weights)
    
    # Compile the final model
    final_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the final model
    final_model.save('app/models/brain_tumor_classifier.h5')
    print("Final combined model created and saved.")
    
    return final_model

def main():
    """Main function to run meningioma-only training."""
    print("Starting meningioma-only optimization process...")
    
    # Ensure model directory exists
    os.makedirs('app/models', exist_ok=True)
    
    # Step 1: Create specialized data generators for meningioma training
    train_generator, validation_generator, test_generator = create_meningioma_only_generators()
    
    # Step 2: Load the original model and prepare for meningioma-only training
    meningioma_model, original_model = load_and_modify_model_for_meningioma(ORIGINAL_MODEL_PATH)
    
    # Step 3: Evaluate the original model for baseline meningioma performance
    print("\nEvaluating ORIGINAL model meningioma performance (before optimization)...")
    baseline_f1, baseline_report = evaluate_meningioma_performance(original_model, test_generator)
    
    # Step 4: Train the model specifically for meningioma
    history, trained_meningioma_model = train_meningioma_only(meningioma_model, train_generator, validation_generator)
    
    # Step 5: Create a combined model that preserves original capabilities but improves meningioma
    final_model = combine_models(original_model, trained_meningioma_model)
    
    # Step 6: Evaluate the final combined model
    print("\nEvaluating FINAL model performance (after meningioma-only optimization)...")
    improved_f1, improved_report = evaluate_meningioma_performance(final_model, test_generator)
    
    # Step 7: Generate improvement report
    print("\nMENINGIOMA IMPROVEMENT SUMMARY:")
    print(f"Baseline Meningioma F1-score: {baseline_f1:.4f}")
    print(f"Improved Meningioma F1-score: {improved_f1:.4f}")
    print(f"Absolute improvement: {improved_f1 - baseline_f1:.4f}")
    print(f"Relative improvement: {((improved_f1 - baseline_f1) / baseline_f1) * 100:.2f}%")
    
    # Step 8: Verify impact on other classes
    print("\nImpact on other classes:")
    for class_name in CLASS_NAMES:
        if class_name != 'meningioma':
            baseline = baseline_report[class_name]['f1-score']
            improved = improved_report[class_name]['f1-score']
            diff = improved - baseline
            print(f"  {class_name}: {baseline:.4f} → {improved:.4f} ({diff:.4f} change)")
    
    print("\nMeningioma-only optimization complete!")
    print("The final model has been saved to app/models/brain_tumor_classifier.h5")

if __name__ == "__main__":
    main() 