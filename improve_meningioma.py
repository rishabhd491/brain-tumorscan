import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants for meningioma improvement
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5  # Very low learning rate for fine-tuning
TRAIN_DIR = 'Training'
TEST_DIR = 'Testing'
ORIGINAL_MODEL_PATH = 'app/models/brain_tumor_classifier.h5'
IMPROVED_MODEL_PATH = 'app/models/improved_brain_tumor_classifier.h5'

# Define class names and meningioma index
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MENINGIOMA_INDEX = 1  # Index of meningioma in CLASS_NAMES

# Create a custom loss function that penalizes meningioma errors more heavily
def weighted_categorical_crossentropy():
    """
    A weighted version of keras.losses.categorical_crossentropy that 
    gives a higher weight to meningioma classification errors.
    """
    def loss(y_true, y_pred):
        # Define class weights - 2.0 for meningioma, 1.0 for others
        weights = tf.constant([1.0, 2.0, 1.0, 1.0], dtype=tf.float32)
        
        # Clip prediction values to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Calculate normal categorical crossentropy
        loss = y_true * tf.math.log(y_pred)
        loss = -tf.reduce_sum(loss, axis=-1)
        
        # Apply class weights based on the true class
        class_weights = tf.reduce_sum(y_true * weights, axis=-1)
        weighted_loss = loss * class_weights
        
        return tf.reduce_mean(weighted_loss)
    return loss

def create_data_generators():
    """Create data generators with special augmentation for meningioma."""
    print("Creating data generators with targeted augmentation...")
    
    # Define augmentation that will be applied to all classes
    base_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Define more aggressive augmentation specifically for meningioma
    meningioma_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,          # More rotation
        width_shift_range=0.3,      # More width shift
        height_shift_range=0.3,     # More height shift
        shear_range=0.25,           # More shear 
        zoom_range=0.3,             # More zoom
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],# Brightness variation
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Standard test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create the training generator
    print("Creating training generator...")
    train_generator = base_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator
    validation_generator = base_datagen.flow_from_directory(
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
    
    # Check class distribution
    print("\nChecking class distribution in training set:")
    unique, counts = np.unique(train_generator.classes, return_counts=True)
    class_distribution = dict(zip([CLASS_NAMES[i] for i in unique], counts))
    print(class_distribution)
    
    # Calculate class weights
    print("\nCalculating balanced class weights...")
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    
    class_weights = {i: weight for i, weight in enumerate(weights)}
    
    # Increase weight for meningioma class even more
    class_weights[MENINGIOMA_INDEX] *= 1.5  # Give 50% extra weight to meningioma
    print("Class weights:", class_weights)
    
    return train_generator, validation_generator, test_generator, class_weights

def load_and_prepare_model(model_path):
    """Load the trained model and prepare for meningioma-focused fine-tuning."""
    print(f"Loading model from {model_path}...")
    
    try:
        # Load the model
        model = load_model(model_path)
        print("Model loaded successfully.")
        
        # Check model architecture
        print("Model summary:")
        model.summary()
        
        # Freeze all layers except the last few
        print("\nFreezing all layers except the last 50...")
        for layer in model.layers[:-50]:
            layer.trainable = False
        
        # Make last 50 layers trainable
        for layer in model.layers[-50:]:
            layer.trainable = True
        
        # Count trainable and non-trainable parameters
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        print(f"Trainable parameters: {trainable_count:,}")
        print(f"Non-trainable parameters: {non_trainable_count:,}")
        
        # Compile the model with our custom weighted loss function for meningioma
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=weighted_categorical_crossentropy(),  # Custom loss function
            metrics=['accuracy']
        )
        
        print("Model compiled with weighted loss function that penalizes meningioma errors more.")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def fine_tune_model(model, train_generator, validation_generator, class_weights):
    """Fine-tune the model with focus on improving meningioma classification."""
    print("\nFine-tuning model with focus on meningioma...")
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        IMPROVED_MODEL_PATH,
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
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights  # Apply computed class weights
    )
    
    return history, model

def evaluate_model(model, test_generator):
    """Evaluate the model with focus on meningioma performance."""
    print("\nEvaluating model performance...")
    
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
    
    # Focus on meningioma performance
    meningioma_precision = report['meningioma']['precision']
    meningioma_recall = report['meningioma']['recall']
    meningioma_f1 = report['meningioma']['f1-score']
    
    print(f"\nMeningioma-Specific Performance:")
    print(f"  Precision: {meningioma_precision:.4f}")
    print(f"  Recall: {meningioma_recall:.4f}")
    print(f"  F1-score: {meningioma_f1:.4f}")
    
    # Calculate meningioma-specific metrics
    meningioma_idx = CLASS_NAMES.index('meningioma')
    meningioma_total = sum(cm[meningioma_idx, :])
    meningioma_correct = cm[meningioma_idx, meningioma_idx]
    meningioma_accuracy = meningioma_correct / meningioma_total
    
    print(f"Meningioma Recognition Rate: {meningioma_accuracy:.4f} ({meningioma_correct}/{meningioma_total})")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix After Meningioma-Focused Improvement')
    plt.savefig('app/static/improved_confusion_matrix.png')
    plt.close()
    
    return report, cm

def compare_performances(original_model, improved_model, test_generator):
    """Compare performance before and after improvement."""
    print("\nComparing original vs. improved model performance...")
    
    # Get predictions from original model
    original_predictions = original_model.predict(test_generator)
    original_y_pred = np.argmax(original_predictions, axis=1)
    
    # Get predictions from improved model
    improved_predictions = improved_model.predict(test_generator)
    improved_y_pred = np.argmax(improved_predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Generate reports
    original_report = classification_report(
        y_true, 
        original_y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True
    )
    
    improved_report = classification_report(
        y_true, 
        improved_y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True
    )
    
    # Focus on meningioma
    meningioma_idx = CLASS_NAMES.index('meningioma')
    
    # Print comparison
    print("\nMeningioma Performance Comparison:")
    print(f"                Original    Improved    Change")
    print(f"Precision:      {original_report['meningioma']['precision']:.4f}      {improved_report['meningioma']['precision']:.4f}      {improved_report['meningioma']['precision'] - original_report['meningioma']['precision']:.4f}")
    print(f"Recall:         {original_report['meningioma']['recall']:.4f}      {improved_report['meningioma']['recall']:.4f}      {improved_report['meningioma']['recall'] - original_report['meningioma']['recall']:.4f}")
    print(f"F1-score:       {original_report['meningioma']['f1-score']:.4f}      {improved_report['meningioma']['f1-score']:.4f}      {improved_report['meningioma']['f1-score'] - original_report['meningioma']['f1-score']:.4f}")
    
    # Check impact on other classes
    print("\nImpact on Other Classes (F1-score):")
    for class_name in CLASS_NAMES:
        if class_name != 'meningioma':
            original = original_report[class_name]['f1-score']
            improved = improved_report[class_name]['f1-score']
            difference = improved - original
            print(f"{class_name}: {original:.4f} → {improved:.4f} ({difference:.4f})")
    
    return original_report, improved_report

def main():
    """Main function to improve meningioma classification."""
    print("Starting specialized meningioma improvement process...")
    
    # Create output directories if they don't exist
    os.makedirs('app/models', exist_ok=True)
    os.makedirs('app/static', exist_ok=True)
    
    # Create data generators with focus on meningioma
    train_generator, validation_generator, test_generator, class_weights = create_data_generators()
    
    # Load the original model
    original_model = load_model(ORIGINAL_MODEL_PATH)
    
    # Evaluate original model
    print("\nEvaluating original model performance...")
    original_predictions = original_model.predict(test_generator)
    original_y_pred = np.argmax(original_predictions, axis=1)
    y_true = test_generator.classes
    original_report = classification_report(
        y_true, 
        original_y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True
    )
    print(f"Original meningioma F1-score: {original_report['meningioma']['f1-score']:.4f}")
    
    # Load and prepare model for fine-tuning
    model = load_and_prepare_model(ORIGINAL_MODEL_PATH)
    
    # Fine-tune the model
    history, improved_model = fine_tune_model(model, train_generator, validation_generator, class_weights)
    
    # Evaluate improved model
    improved_report, cm = evaluate_model(improved_model, test_generator)
    
    # Compare performances
    compare_performances(original_model, improved_model, test_generator)
    
    # Save improved model (was already saved by callback during training)
    print(f"\nFinal improved model saved to {IMPROVED_MODEL_PATH}")
    
    # Create backup of original model
    original_backup_path = 'app/models/original_brain_tumor_classifier_backup.h5'
    original_model.save(original_backup_path)
    print(f"Original model backed up to {original_backup_path}")
    
    # Optionally save the improved model as the main model
    improved_model.save('app/models/brain_tumor_classifier.h5')
    print("Improved model saved as the main model.")
    
    print("\nMeningioma improvement process complete!")
    
    # Print instructions
    print("\nTo use the improved model in your app, ensure 'app/models/brain_tumor_classifier.h5' is being loaded.")
    print("You can compare the original vs. improved confusion matrices in the static folder.")

if __name__ == "__main__":
    main() 