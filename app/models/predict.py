import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Class names in the order they were used during training
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image size used for model training - updated to match training parameters
IMG_SIZE = 160  # Changed from 224 to match the training model size

# Define the custom loss function that was used during training
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

def load_brain_tumor_model():
    """Load the trained brain tumor classification model."""
    # First try the local path
    model_path = os.path.join(os.path.dirname(__file__), 'brain_tumor_classifier.h5')
    
    # If not found, try alternate paths
    if not os.path.exists(model_path):
        # Try absolute path
        alt_path = os.path.join(os.getcwd(), 'app', 'models', 'brain_tumor_classifier.h5')
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"Using alternate model path: {model_path}")
        # Try other variants if needed
        elif os.path.exists('app/models/brain_tumor_classifier.h5'):
            model_path = 'app/models/brain_tumor_classifier.h5'
            print(f"Using direct path: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or alternate locations. Please train the model first.")
    
    try:
        # First try to load with custom objects for the custom loss function
        model = load_model(model_path, custom_objects={
            'loss': weighted_categorical_crossentropy(),
            'weighted_categorical_crossentropy': weighted_categorical_crossentropy
        })
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading with custom loss: {e}")
        # Fallback: try to load with compile=False and then recompile
        try:
            model = load_model(model_path, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Model loaded with fallback method")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    return model

def preprocess_image(img_path):
    """Preprocess the input image for prediction."""
    # Read and resize the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the same size used during training
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def predict_tumor_type(model, img_path):
    """Predict tumor type from the given image path."""
    try:
        # Preprocess the image
        img_array = preprocess_image(img_path)
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities
        }
    
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # This is just for testing the module
    model = load_brain_tumor_model()
    print("Model loaded successfully.")
    
    # Sample prediction (replace with actual image path for testing)
    test_img_path = "path/to/test/image.jpg"
    if os.path.exists(test_img_path):
        result = predict_tumor_type(model, test_img_path)
        print(f"Prediction result: {result}") 