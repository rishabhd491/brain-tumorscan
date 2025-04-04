import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, Attention
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import shutil 
import cv2
from tqdm import tqdm

print("TensorFlow version:", tf.__version__)

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants for hyper-specialized meningioma training
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 5e-5  # Very conservative learning rate
TRAIN_DIR = 'Training'
TEST_DIR = 'Testing'
MENINGIOMA_AUGMENTED_DIR = 'Training_meningioma_augmented'
ORIGINAL_MODEL_PATH = 'app/models/brain_tumor_classifier.h5'
MENINGIOMA_SPECIALIST_MODEL_PATH = 'app/models/meningioma_specialist_model.h5'

# Define class names and meningioma index
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MENINGIOMA_INDEX = 1  # Index of meningioma in CLASS_NAMES
MENINGIOMA_CLASS = 'meningioma'  # Name of meningioma class folder

# Create a more aggressive custom loss function for meningioma
def meningioma_focal_loss(gamma=2.0, alpha=4.0):
    """
    Focal loss with extra emphasis on meningioma class.
    - gamma: focusing parameter that reduces loss for well-classified examples
    - alpha: weighting factor for meningioma class
    """
    def loss(y_true, y_pred):
        # Get meningioma mask (which samples are meningioma)
        meningioma_mask = y_true[:, MENINGIOMA_INDEX]
        
        # Clip prediction values to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Apply focal weighting
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Apply class weighting with extra weight for meningioma
        weight_vector = tf.ones_like(y_true)
        # Set meningioma weight to alpha
        weight_vector = tf.tensor_scatter_nd_update(
            weight_vector, 
            tf.where(tf.equal(y_true[:, MENINGIOMA_INDEX:MENINGIOMA_INDEX+1], 1.0)), 
            tf.ones_like(y_true[:, 0:1]) * alpha
        )
        
        # Combine weights
        final_weight = focal_weight * weight_vector
        weighted_loss = cross_entropy * final_weight
        
        return tf.reduce_mean(weighted_loss)
    
    return loss

def create_meningioma_augmentation_directory():
    """Create a directory with extra augmented meningioma samples."""
    print("Creating specialized meningioma augmentation dataset...")
    
    # Define source and destination paths
    src_path = os.path.join(TRAIN_DIR, MENINGIOMA_CLASS)
    aug_path = os.path.join(MENINGIOMA_AUGMENTED_DIR, MENINGIOMA_CLASS)
    
    # Create destination directory if it doesn't exist
    os.makedirs(aug_path, exist_ok=True)
    
    # First, copy original images
    if not os.listdir(aug_path):
        for filename in os.listdir(src_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(src_path, filename)
                dst_file = os.path.join(aug_path, filename)
                shutil.copy2(src_file, dst_file)
    
    # Define a more aggressive augmentation for meningioma
    augmentation = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.4,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='reflect'  # Better border handling
    )
    
    # Generate multiple augmented versions of each meningioma image
    print("Generating augmented meningioma images...")
    img_files = [f for f in os.listdir(src_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Limit the number of augmentations per image to control the dataset size
    augmentations_per_image = 5
    target_aug_count = len(img_files) * augmentations_per_image
    
    # Only generate new augmentations if needed
    existing_files = len([f for f in os.listdir(aug_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing_files >= len(img_files) + target_aug_count:
        print(f"Using existing augmented dataset with {existing_files} files")
        return
    
    for filename in tqdm(img_files):
        # Load and preprocess image
        img_path = os.path.join(src_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Generate augmentations
        aug_iter = augmentation.flow(img, batch_size=1)
        
        for i in range(augmentations_per_image):
            aug_img = next(aug_iter)[0]
            aug_img = (aug_img * 255).astype(np.uint8)
            
            # Save augmented image
            aug_filename = f"aug_{i}_{filename}"
            aug_path_file = os.path.join(aug_path, aug_filename)
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(aug_path_file, aug_img_bgr)
    
    print(f"Created augmented meningioma dataset with {len(os.listdir(aug_path))} images")

def copy_other_classes():
    """Copy other classes to the augmented directory."""
    for class_name in CLASS_NAMES:
        if class_name != MENINGIOMA_CLASS:
            src_path = os.path.join(TRAIN_DIR, class_name)
            dst_path = os.path.join(MENINGIOMA_AUGMENTED_DIR, class_name)
            
            if not os.path.exists(dst_path):
                os.makedirs(dst_path, exist_ok=True)
                
                for filename in os.listdir(src_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        src_file = os.path.join(src_path, filename)
                        dst_file = os.path.join(dst_path, filename)
                        shutil.copy2(src_file, dst_file)
    
    print("Copied other classes to augmented directory")

def create_data_generators():
    """Create data generators with normal augmentation for training."""
    print("Creating data generators...")
    
    # Create the augmented dataset structure
    os.makedirs(MENINGIOMA_AUGMENTED_DIR, exist_ok=True)
    create_meningioma_augmentation_directory()
    copy_other_classes()
    
    # Define augmentation for training (lighter, since we already did heavy augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% validation
    )
    
    # Only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Use the augmented directory for training
    train_generator = train_datagen.flow_from_directory(
        MENINGIOMA_AUGMENTED_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        MENINGIOMA_AUGMENTED_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Test generator uses the original test directory
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
    
    # Calculate balanced class weights
    print("\nCalculating class weights...")
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    
    class_weights = {i: weight for i, weight in enumerate(weights)}
    
    # Despite the augmentation, still increase weight for meningioma class
    class_weights[MENINGIOMA_INDEX] *= 1.2
    print("Class weights:", class_weights)
    
    return train_generator, validation_generator, test_generator, class_weights

def get_attention_meningioma_model():
    """Build a specialized model with attention mechanisms for meningioma."""
    print("Building specialized meningioma attention model...")
    
    # Load MobileNetV2 as base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze most base model layers
    for layer in base_model.layers[:-80]:  # Unfreeze more layers
        layer.trainable = False
    
    # Extract features from the base model
    x = base_model.output
    
    # Add attention mechanism to focus on relevant tumor regions
    attention_layer = GlobalAveragePooling2D()(x)
    attention_layer = Dense(512, activation='relu')(attention_layer)
    attention_layer = Dense(base_model.output.shape[-1], activation='sigmoid')(attention_layer)
    attention_layer = tf.reshape(attention_layer, [-1, 1, 1, base_model.output.shape[-1]])
    
    # Apply attention weights to the base model output
    x = tf.multiply(x, attention_layer)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # First dense block with stronger regularization
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Higher dropout
    
    # Second dense block
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Final classification layer with L2 regularization
    predictions = Dense(len(CLASS_NAMES), activation='softmax', 
                       kernel_regularizer=l2(0.001))(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model with our specialized loss function
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=meningioma_focal_loss(gamma=2.0, alpha=4.0),  # Specialized loss
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, validation_generator, class_weights):
    """Train the specialized meningioma model."""
    print("Training meningioma specialist model...")
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        MENINGIOMA_SPECIALIST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,  # More patience
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
    
    # Add cosine annealing learning rate scheduler for better convergence
    cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * len(train_generator),
        alpha=1e-7
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train with class weights
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    return history, model

def evaluate_model(model, test_generator):
    """Evaluate the model's performance with focus on meningioma."""
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
    
    # Analyze meningioma errors
    meningioma_idx = CLASS_NAMES.index('meningioma')
    meningioma_samples = np.where(y_true == meningioma_idx)[0]
    meningioma_correct = np.where((y_true == meningioma_idx) & (y_pred == meningioma_idx))[0]
    meningioma_incorrect = np.where((y_true == meningioma_idx) & (y_pred != meningioma_idx))[0]
    
    print(f"Meningioma samples: {len(meningioma_samples)}")
    print(f"Correctly classified: {len(meningioma_correct)} ({len(meningioma_correct)/len(meningioma_samples):.2%})")
    print(f"Incorrectly classified: {len(meningioma_incorrect)} ({len(meningioma_incorrect)/len(meningioma_samples):.2%})")
    
    # Count what meningioma is confused with
    if len(meningioma_incorrect) > 0:
        incorrect_predictions = y_pred[meningioma_incorrect]
        unique_incorrect, counts_incorrect = np.unique(incorrect_predictions, return_counts=True)
        print("Meningioma incorrectly classified as:")
        for idx, count in zip(unique_incorrect, counts_incorrect):
            print(f"  {CLASS_NAMES[idx]}: {count} samples ({count/len(meningioma_incorrect):.2%})")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Meningioma Specialist Model')
    plt.savefig('app/static/meningioma_specialist_confusion_matrix.png')
    plt.close()
    
    return report, cm

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('app/static/meningioma_specialist_training_history.png')
    plt.close()

def compare_with_original_model(test_generator):
    """Compare performance with the original model."""
    print("\nComparing with original model...")
    
    # Load original model
    try:
        # Try to load with custom loss function if available
        original_model = load_model(ORIGINAL_MODEL_PATH, compile=False)
        original_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    except:
        print("Could not load original model for comparison.")
        return
    
    # Get predictions from original model
    original_predictions = original_model.predict(test_generator)
    original_y_pred = np.argmax(original_predictions, axis=1)
    
    # Load the specialist model
    specialist_model = load_model(MENINGIOMA_SPECIALIST_MODEL_PATH, compile=False)
    specialist_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Get predictions from specialist model
    specialist_predictions = specialist_model.predict(test_generator)
    specialist_y_pred = np.argmax(specialist_predictions, axis=1)
    
    # True labels
    y_true = test_generator.classes
    
    # Generate reports
    original_report = classification_report(
        y_true, 
        original_y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True
    )
    
    specialist_report = classification_report(
        y_true, 
        specialist_y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True
    )
    
    # Compare meningioma performance
    print("\nMENINGIOMA PERFORMANCE COMPARISON:")
    print("Metric        Original    Specialist    Improvement")
    print("-" * 55)
    
    # Precision
    original_precision = original_report['meningioma']['precision']
    specialist_precision = specialist_report['meningioma']['precision']
    precision_improvement = specialist_precision - original_precision
    print(f"Precision     {original_precision:.4f}      {specialist_precision:.4f}        {precision_improvement:+.4f} ({precision_improvement/original_precision*100:+.2f}%)")
    
    # Recall
    original_recall = original_report['meningioma']['recall']
    specialist_recall = specialist_report['meningioma']['recall']
    recall_improvement = specialist_recall - original_recall
    print(f"Recall        {original_recall:.4f}      {specialist_recall:.4f}        {recall_improvement:+.4f} ({recall_improvement/original_recall*100:+.2f}%)")
    
    # F1-score
    original_f1 = original_report['meningioma']['f1-score']
    specialist_f1 = specialist_report['meningioma']['f1-score']
    f1_improvement = specialist_f1 - original_f1
    print(f"F1-score      {original_f1:.4f}      {specialist_f1:.4f}        {f1_improvement:+.4f} ({f1_improvement/original_f1*100:+.2f}%)")
    
    # Check impact on other classes
    print("\nIMPACT ON OTHER CLASSES (F1-score):")
    for class_name in CLASS_NAMES:
        if class_name != 'meningioma':
            original = original_report[class_name]['f1-score']
            specialist = specialist_report[class_name]['f1-score']
            diff = specialist - original
            diff_percent = diff / original * 100 if original > 0 else 0
            print(f"{class_name}: {original:.4f} → {specialist:.4f} ({diff:+.4f}, {diff_percent:+.2f}%)")
    
    # Overall accuracy
    original_accuracy = original_report['accuracy']
    specialist_accuracy = specialist_report['accuracy']
    accuracy_improvement = specialist_accuracy - original_accuracy
    print(f"\nOverall Accuracy: {original_accuracy:.4f} → {specialist_accuracy:.4f} ({accuracy_improvement:+.4f}, {accuracy_improvement/original_accuracy*100:+.2f}%)")
    
    # Create a backup of the original model before replacing
    backup_dir = 'app/models/backups'
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, 'brain_tumor_classifier_backup.h5')
    shutil.copy2(ORIGINAL_MODEL_PATH, backup_path)
    print(f"Original model backed up to {backup_path}")

def save_model_for_production():
    """Save the specialist model as the main model."""
    # Copy the specialist model to the main model path
    shutil.copy2(MENINGIOMA_SPECIALIST_MODEL_PATH, ORIGINAL_MODEL_PATH)
    print(f"Specialist model saved as the main model at {ORIGINAL_MODEL_PATH}")
    
    # Create a custom predict.py script that includes the loss function
    with open('app/models/meningioma_loss.py', 'w') as f:
        f.write("""
import tensorflow as tf

def meningioma_focal_loss(gamma=2.0, alpha=4.0):
    \"\"\"
    Focal loss with extra emphasis on meningioma class.
    - gamma: focusing parameter that reduces loss for well-classified examples
    - alpha: weighting factor for meningioma class
    \"\"\"
    MENINGIOMA_INDEX = 1  # Index of meningioma in CLASS_NAMES
    
    def loss(y_true, y_pred):
        # Get meningioma mask (which samples are meningioma)
        meningioma_mask = y_true[:, MENINGIOMA_INDEX]
        
        # Clip prediction values to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Apply focal weighting
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Apply class weighting with extra weight for meningioma
        weight_vector = tf.ones_like(y_true)
        # Set meningioma weight to alpha
        weight_vector = tf.tensor_scatter_nd_update(
            weight_vector, 
            tf.where(tf.equal(y_true[:, MENINGIOMA_INDEX:MENINGIOMA_INDEX+1], 1.0)), 
            tf.ones_like(y_true[:, 0:1]) * alpha
        )
        
        # Combine weights
        final_weight = focal_weight * weight_vector
        weighted_loss = cross_entropy * final_weight
        
        return tf.reduce_mean(weighted_loss)
    
    return loss
""")

def main():
    """Main function to run specialized meningioma training."""
    print("Starting specialized meningioma training process...")
    
    # Create output directories
    os.makedirs('app/models', exist_ok=True)
    os.makedirs('app/static', exist_ok=True)
    
    # Create data generators with augmented meningioma samples
    train_generator, validation_generator, test_generator, class_weights = create_data_generators()
    
    # Create the specialized model
    model = get_attention_meningioma_model()
    
    # Train the model
    history, trained_model = train_model(model, train_generator, validation_generator, class_weights)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    report, cm = evaluate_model(trained_model, test_generator)
    
    # Compare with original model
    compare_with_original_model(test_generator)
    
    # Save the model for production
    save_model_for_production()
    
    print("\nSpecialized meningioma training complete!")
    print(f"The improved model has been saved to {ORIGINAL_MODEL_PATH}")
    print("A backup of the original model has been created in app/models/backups/")

if __name__ == "__main__":
    main() 