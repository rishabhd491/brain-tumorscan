import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants for meningioma-focused training
IMG_SIZE = 160  # Keep the same image size for consistency
BATCH_SIZE = 32  # Smaller batch size for better generalization
EPOCHS = 25  # Increased epochs for better learning
LEARNING_RATE = 0.0005  # Adjusted for better convergence
TRAIN_DIR = 'Training'
TEST_DIR = 'Testing'
MODEL_SAVE_PATH = 'app/models/brain_tumor_classifier.h5'

# Define class names
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Focus specifically on improving meningioma results
MENINGIOMA_INDEX = 1  # Index of meningioma in CLASS_NAMES

# Use all available training data
USE_SUBSET = False  # Use all data

def create_data_generators():
    """Create and configure data generators for training and testing."""
    # Custom augmentation focused on meningioma characteristics
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,       # Increased
        height_shift_range=0.2,      # Increased
        shear_range=0.15,
        zoom_range=0.2,              # Increased zoom range
        horizontal_flip=True,
        vertical_flip=False,         # MRI orientation matters
        brightness_range=[0.8, 1.2], 
        fill_mode='nearest',
        validation_split=0.2         # Use 20% of training data for validation
    )
    
    # Only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator with validation split
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
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
    print("Class distribution in training set:")
    for class_name, count in zip(train_generator.class_indices.keys(), 
                                np.bincount(train_generator.classes)):
        print(f"  {class_name}: {count} images")
    
    # Compute class weights to give more importance to meningioma
    class_weights = compute_class_weights(train_generator.classes)
    print("Class weights:", class_weights)
    
    if USE_SUBSET:
        MAX_SAMPLES_PER_CLASS = 500
        print(f"Using subset of data: {MAX_SAMPLES_PER_CLASS} samples per class")
        train_generator.samples = min(train_generator.samples, MAX_SAMPLES_PER_CLASS * len(CLASS_NAMES))
        validation_generator.samples = min(validation_generator.samples, int(MAX_SAMPLES_PER_CLASS * len(CLASS_NAMES) * 0.2))
        test_generator.samples = min(test_generator.samples, MAX_SAMPLES_PER_CLASS * len(CLASS_NAMES) // 2)
    else:
        print(f"Using all available training data: {train_generator.samples} training samples, {validation_generator.samples} validation samples")
    
    return train_generator, validation_generator, test_generator, class_weights

def compute_class_weights(y_train):
    """Compute class weights to address class imbalance."""
    # Compute balanced class weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Convert to dictionary
    class_weights = {i: weight for i, weight in enumerate(weights)}
    
    # Give extra weight to meningioma class to improve its performance
    class_weights[MENINGIOMA_INDEX] *= 1.5  # 50% more weight
    
    return class_weights

def build_model(num_classes):
    """Build and compile the MobileNetV2 model with improved architecture for meningioma detection."""
    # Load MobileNetV2 as base model with pre-trained ImageNet weights
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Initially freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add custom classification head with more capacity and regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Combine base model and new layers into final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, train_generator, validation_generator, class_weights):
    """Train the model with callbacks for better performance."""
    # Define callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,  # Increased
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
        class_weight=class_weights  # Apply class weights
    )
    
    return history

def fine_tune_model(model, base_model, train_generator, validation_generator, class_weights):
    """Fine-tune the model by unfreezing more layers of the base model."""
    # Unfreeze more layers of the base model for better fine-tuning
    # Include more layers to allow the model to better recognize meningioma features
    for layer in base_model.layers[-50:]:  # Increased to unfreeze more layers
        layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks for fine-tuning
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,  # Increased patience
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Fine-tune the model with more epochs
    history_fine = model.fit(
        train_generator,
        epochs=15,  # Increased for better fine-tuning
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights  # Apply class weights during fine-tuning
    )
    
    return history_fine

def evaluate_model(model, test_generator):
    """Evaluate the model on the test dataset."""
    # Get the predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get the true labels
    y_true = test_generator.classes
    
    # Generate the classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True
    )
    
    print("Classification Report:")
    for class_name in CLASS_NAMES:
        print(f"{class_name}:")
        print(f"  Precision: {report[class_name]['precision']:.4f}")
        print(f"  Recall: {report[class_name]['recall']:.4f}")
        print(f"  F1-score: {report[class_name]['f1-score']:.4f}")
    
    print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
    
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('app/static/confusion_matrix.png')
    plt.close()
    
    # Calculate and print meningioma-specific metrics
    meningioma_idx = CLASS_NAMES.index('meningioma')
    meningioma_precision = report['meningioma']['precision']
    meningioma_recall = report['meningioma']['recall']
    meningioma_f1 = report['meningioma']['f1-score']
    
    print(f"\nMeningioma Specific Metrics:")
    print(f"  Precision: {meningioma_precision:.4f}")
    print(f"  Recall: {meningioma_recall:.4f}")
    print(f"  F1-score: {meningioma_f1:.4f}")
    
    # Calculate and return the test accuracy
    test_loss, test_acc = model.evaluate(test_generator)
    return test_acc

def plot_training_history(history, history_fine=None):
    """Plot the training and validation accuracy/loss."""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    
    # If fine-tuning history is provided, plot it too
    if history_fine is not None:
        # Get the last epoch index from the initial training
        last_epoch = len(history.history['accuracy'])
        
        # Plot fine-tuning accuracy
        epochs_fine = range(last_epoch, last_epoch + len(history_fine.history['accuracy']))
        ax1.plot(epochs_fine, history_fine.history['accuracy'], 'g-', label='Fine-tuning Training Accuracy')
        ax1.plot(epochs_fine, history_fine.history['val_accuracy'], 'g--', label='Fine-tuning Validation Accuracy')
    
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot the loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    
    # If fine-tuning history is provided, plot it too
    if history_fine is not None:
        # Plot fine-tuning loss
        ax2.plot(epochs_fine, history_fine.history['loss'], 'g-', label='Fine-tuning Training Loss')
        ax2.plot(epochs_fine, history_fine.history['val_loss'], 'g--', label='Fine-tuning Validation Loss')
    
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('app/static/training_history.png')
    plt.close()

def main():
    """Main function to train and evaluate the model with focus on meningioma classification."""
    print("Starting specialized training for improved meningioma classification...")
    
    # Create data generators
    print("Creating data generators...")
    train_generator, validation_generator, test_generator, class_weights = create_data_generators()
    
    # Get the number of classes
    num_classes = len(train_generator.class_indices)
    
    # Build the model
    print("Building model...")
    model, base_model = build_model(num_classes)
    
    # Train the model (initial training)
    print("Training model...")
    history = train_model(model, train_generator, validation_generator, class_weights)
    
    # Fine-tune the model
    print("Fine-tuning model...")
    history_fine = fine_tune_model(model, base_model, train_generator, validation_generator, class_weights)
    
    # Plot the training history
    print("Plotting training history...")
    plot_training_history(history, history_fine)
    
    # Evaluate the model
    print("Evaluating model...")
    test_accuracy = evaluate_model(model, test_generator)
    print(f"Final test accuracy: {test_accuracy:.4f}")
    
    # Save model summary to a file
    with open('app/models/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("Training complete!")

if __name__ == "__main__":
    # Make sure the model directory exists
    os.makedirs('app/models', exist_ok=True)
    main() 