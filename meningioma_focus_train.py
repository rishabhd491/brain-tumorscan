import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants
IMG_SIZE = 160  # Keep the same image size as original model
BATCH_SIZE = 16  # Smaller batch size for more gradient updates
EPOCHS = 10     # Short fine-tuning
LEARNING_RATE = 0.0001  # Low learning rate for fine-tuning
TRAIN_DIR = 'Training'
TEST_DIR = 'Testing'
ORIGINAL_MODEL_PATH = 'app/models/brain_tumor_classifier.h5'
IMPROVED_MODEL_PATH = 'app/models/meningioma_improved_model.h5'

# Define class names and meningioma index
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MENINGIOMA_INDEX = 1  # Index of meningioma in CLASS_NAMES

def create_meningioma_focused_generators():
    """Create data generators with special focus on meningioma class."""
    # Data augmentation specifically designed for meningioma characteristics
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,         # More rotation for meningioma variants
        width_shift_range=0.25,    # Increased for more position variants
        height_shift_range=0.25,   # Increased for more position variants
        shear_range=0.2,           # Increased for shape variation
        zoom_range=0.25,           # Increased zoom variations
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],  # More brightness variation
        fill_mode='nearest',
        validation_split=0.2        # 20% validation
    )
    
    # Only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create training generator with more meningioma augmentation
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Create test generator
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Print class distribution
    print("Class distribution in training set:")
    for class_name, count in zip(train_generator.class_indices.keys(), 
                               np.bincount(train_generator.classes)):
        print(f"  {class_name}: {count} images")
    
    # Calculate class weights with heavy emphasis on meningioma
    class_weights = {i: 1.0 for i in range(len(CLASS_NAMES))}
    class_weights[MENINGIOMA_INDEX] = 3.0  # Give 3x weight to meningioma
    print("Class weights:", class_weights)
    
    return train_generator, validation_generator, test_generator, class_weights

def load_and_prepare_for_meningioma_tuning(model_path):
    """Load existing model and prepare it for meningioma-focused fine-tuning."""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Make only the last 30 layers trainable for fine-tuning
    for layer in model.layers[:-30]:
        layer.trainable = False
    for layer in model.layers[-30:]:
        layer.trainable = True
    
    # Recompile with very low learning rate to avoid destroying existing features
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model prepared for meningioma-focused fine-tuning.")
    return model

def meningioma_focused_fine_tuning(model, train_generator, validation_generator, class_weights):
    """Fine-tune the model with special focus on meningioma class."""
    # Define callbacks for fine-tuning
    checkpoint = ModelCheckpoint(
        IMPROVED_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Fine-tune the model with heavy class weighting for meningioma
    print("Starting meningioma-focused fine-tuning...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    return history, model

def evaluate_meningioma_performance(model, test_generator):
    """Evaluate the model's performance with focus on meningioma class."""
    print("Evaluating model performance...")
    
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Generate classification report
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
    
    # Highlight meningioma performance
    meningioma_precision = report['meningioma']['precision']
    meningioma_recall = report['meningioma']['recall']
    meningioma_f1 = report['meningioma']['f1-score']
    
    print(f"\nMeningioma-Specific Performance:")
    print(f"  Precision: {meningioma_precision:.4f}")
    print(f"  Recall: {meningioma_recall:.4f}")
    print(f"  F1-score: {meningioma_f1:.4f}")
    
    # Save the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (After Meningioma Optimization)')
    plt.savefig('app/static/meningioma_improved_confusion_matrix.png')
    plt.close()
    
    # Calculate meningioma-specific confusion metrics
    meningioma_idx = CLASS_NAMES.index('meningioma')
    meningioma_total = sum(cm[meningioma_idx, :])
    meningioma_correct = cm[meningioma_idx, meningioma_idx]
    meningioma_accuracy = meningioma_correct / meningioma_total
    
    print(f"Meningioma Recognition Rate: {meningioma_accuracy:.4f} ({meningioma_correct}/{meningioma_total})")
    
    # Count misclassifications by class
    misclassifications = {}
    for i, class_name in enumerate(CLASS_NAMES):
        if i != meningioma_idx:
            # How many meningiomas were incorrectly classified as this class
            misclassifications[class_name] = cm[meningioma_idx, i]
    
    print("Meningioma misclassified as:")
    for class_name, count in misclassifications.items():
        print(f"  {class_name}: {count} instances ({count/meningioma_total:.2%})")
    
    return meningioma_f1, report

def analyze_meningioma_errors(model, test_generator):
    """Analyze which specific meningioma cases are misclassified."""
    print("\nAnalyzing meningioma misclassifications...")
    
    # Get file paths
    file_paths = test_generator.filepaths
    
    # Get true labels and class indices
    y_true = test_generator.classes
    class_indices = test_generator.class_indices
    indices_to_classes = {v: k for k, v in class_indices.items()}
    
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Find meningioma cases
    meningioma_idx = class_indices['meningioma']
    meningioma_cases = np.where(y_true == meningioma_idx)[0]
    
    # Find misclassified meningioma cases
    misclassified_meningioma = meningioma_cases[y_pred[meningioma_cases] != meningioma_idx]
    
    print(f"Total meningioma test cases: {len(meningioma_cases)}")
    print(f"Misclassified meningioma cases: {len(misclassified_meningioma)} ({len(misclassified_meningioma)/len(meningioma_cases):.2%})")
    
    # Group misclassifications by predicted class
    misclassifications_by_class = {}
    for idx in misclassified_meningioma:
        pred_class = indices_to_classes[y_pred[idx]]
        if pred_class not in misclassifications_by_class:
            misclassifications_by_class[pred_class] = []
        misclassifications_by_class[pred_class].append((idx, file_paths[idx], predictions[idx]))
    
    # Print summary of misclassifications
    for pred_class, cases in misclassifications_by_class.items():
        print(f"\nMeningioma cases misclassified as {pred_class}: {len(cases)}")
        
        # Print top 3 worst misclassifications for each wrong class
        cases.sort(key=lambda x: x[2][meningioma_idx])  # Sort by meningioma confidence (ascending)
        for i, (idx, path, pred) in enumerate(cases[:3]):
            print(f"  {i+1}. File: {os.path.basename(path)}")
            print(f"     Confidence scores: meningioma={pred[meningioma_idx]:.4f}, {pred_class}={pred[y_pred[idx]]:.4f}")

def main():
    """Main function to run targeted meningioma optimization."""
    print("Starting targeted improvement for meningioma classification...")
    
    # Ensure model directory exists
    os.makedirs('app/models', exist_ok=True)
    
    # Create data generators with meningioma focus
    train_generator, validation_generator, test_generator, class_weights = create_meningioma_focused_generators()
    
    # Load and prepare existing model
    model = load_and_prepare_for_meningioma_tuning(ORIGINAL_MODEL_PATH)
    
    # Evaluate before fine-tuning to get baseline
    print("\nBASELINE MENINGIOMA PERFORMANCE (BEFORE OPTIMIZATION):")
    baseline_f1, baseline_report = evaluate_meningioma_performance(model, test_generator)
    
    # Fine-tune model with meningioma focus
    history, improved_model = meningioma_focused_fine_tuning(model, train_generator, validation_generator, class_weights)
    
    # Evaluate after fine-tuning
    print("\nIMPROVED MENINGIOMA PERFORMANCE (AFTER OPTIMIZATION):")
    improved_f1, improved_report = evaluate_meningioma_performance(improved_model, test_generator)
    
    # Analyze error cases
    analyze_meningioma_errors(improved_model, test_generator)
    
    # Print improvement summary
    print("\nMENINGIOMA OPTIMIZATION SUMMARY:")
    print(f"Baseline F1-score: {baseline_f1:.4f}")
    print(f"Improved F1-score: {improved_f1:.4f}")
    print(f"Absolute improvement: {improved_f1 - baseline_f1:.4f}")
    print(f"Relative improvement: {((improved_f1 - baseline_f1) / baseline_f1) * 100:.2f}%")
    
    # Report on other classes to ensure we didn't hurt their performance
    print("\nImpact on other classes:")
    for class_name in CLASS_NAMES:
        if class_name != 'meningioma':
            baseline = baseline_report[class_name]['f1-score']
            improved = improved_report[class_name]['f1-score']
            diff = improved - baseline
            print(f"  {class_name}: {baseline:.4f} → {improved:.4f} ({diff:.4f} change)")
    
    print(f"\nImproved model saved to {IMPROVED_MODEL_PATH}")
    print("Meningioma optimization complete!")

if __name__ == "__main__":
    main() 