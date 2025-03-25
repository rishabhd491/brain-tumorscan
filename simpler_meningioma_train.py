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

print("TensorFlow version:", tf.__version__)

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.00005  # Very low learning rate
TRAIN_DIR = 'Training'
TEST_DIR = 'Testing'
MODEL_PATH = 'app/models/brain_tumor_classifier.h5'

# Define class names and meningioma index
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MENINGIOMA_INDEX = 1  # Index of meningioma in CLASS_NAMES

def create_data_generators():
    """Create data generators with heavy augmentation for meningioma class."""
    print("Creating data generators...")
    
    # Create a custom generator that applies more augmentation to meningioma
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,          # Increased rotation
        width_shift_range=0.3,      # Increased shift
        height_shift_range=0.3,     # Increased shift
        shear_range=0.2,
        zoom_range=0.3,             # Increased zoom
        horizontal_flip=True,
        brightness_range=[0.7, 1.3], 
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create training generator
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
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Print class distribution
    print("\nClass distribution in training set:")
    unique, counts = np.unique(train_generator.classes, return_counts=True)
    class_distribution = dict(zip([CLASS_NAMES[i] for i in unique], counts))
    print(class_distribution)
    
    # Create class weights with extra emphasis on meningioma
    class_weights = {0: 1.0, 1: 3.0, 2: 1.0, 3: 1.0}  # 3x weight for meningioma
    print("Class weights:", class_weights)
    
    return train_generator, validation_generator, test_generator, class_weights

def load_and_prepare_model():
    """Load and prepare the model for fine-tuning."""
    print(f"Loading model from {MODEL_PATH}...")
    
    try:
        # Load model with compile=False to avoid custom loss issues
        model = load_model(MODEL_PATH, compile=False)
        
        # Freeze all layers except the last 50
        for layer in model.layers[:-50]:
            layer.trainable = False
        
        # Make last 50 layers trainable
        for layer in model.layers[-50:]:
            layer.trainable = True
        
        # Print trainable layers
        trainable_count = sum(layer.trainable for layer in model.layers)
        print(f"Total layers: {len(model.layers)}")
        print(f"Trainable layers: {trainable_count}")
        print(f"Frozen layers: {len(model.layers) - trainable_count}")
        
        # Compile with standard loss but very low learning rate
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def train_model(model, train_generator, validation_generator, class_weights):
    """Train the model with focus on meningioma class."""
    print("Starting model training...")
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH + '.new',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
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
    
    # Train the model with class weights
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    return history, model

def evaluate_model(model, test_generator):
    """Evaluate the model with detailed metrics for meningioma."""
    print("\nEvaluating model...")
    
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
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Improved Meningioma Detection')
    plt.tight_layout()
    plt.savefig('app/static/meningioma_improved_cm.png')
    plt.close()
    
    return report, cm

def plot_training_history(history):
    """Plot the training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('app/static/meningioma_training_history.png')
    plt.close()

def save_final_model(model):
    """Save the final model."""
    # Create a backup of the original model
    if os.path.exists(MODEL_PATH):
        backup_path = MODEL_PATH + '.backup'
        if not os.path.exists(backup_path):
            os.rename(MODEL_PATH, backup_path)
            print(f"Original model backed up to {backup_path}")
    
    # Save the new model
    if os.path.exists(MODEL_PATH + '.new'):
        os.rename(MODEL_PATH + '.new', MODEL_PATH)
        print(f"New model saved to {MODEL_PATH}")
    else:
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

def main():
    """Main function to improve meningioma detection."""
    print("Starting improved meningioma training...")
    
    # Create necessary directories
    os.makedirs('app/models', exist_ok=True)
    os.makedirs('app/static', exist_ok=True)
    
    # Create data generators with special focus on meningioma
    train_generator, validation_generator, test_generator, class_weights = create_data_generators()
    
    # Load and prepare the model
    model = load_and_prepare_model()
    
    # Train the model
    history, trained_model = train_model(model, train_generator, validation_generator, class_weights)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    report, cm = evaluate_model(trained_model, test_generator)
    
    # Save the final model
    save_final_model(trained_model)
    
    print("\nMeningioma training improvement complete!")
    print("The model has been updated with better meningioma detection.")
    print("You can view the performance metrics in the plots saved in app/static/")

if __name__ == "__main__":
    main() 