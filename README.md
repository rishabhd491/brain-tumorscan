# Brain Tumor MRI Classification Web Application

This web application uses a Convolutional Neural Network (CNN) based on MobileNet architecture to classify brain tumor MRI scans into four categories:
- Meningioma
- Glioma
- Pituitary tumor
- No tumor

## Features
- Upload and analyze MRI scan images
- Real-time prediction using a pre-trained CNN model
- Display classification results with confidence scores
- Responsive web interface

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Train the model (this may take some time):
   ```
   python train_model.py
   ```
6. Run the application:
   ```
   python app.py
   ```
7. Open your web browser and navigate to http://localhost:5000

## Usage

### Training the Model

The model needs to be trained before it can be used for predictions. The training process includes:

1. Loading and preprocessing the brain MRI dataset
2. Creating a MobileNetV2-based model with custom classification head
3. Training the model using transfer learning
4. Fine-tuning the model by unfreezing some layers
5. Evaluating the model performance on the test dataset

To start the training process, run:
```
python train_model.py
```

The trained model will be saved to `app/models/brain_tumor_classifier.h5`.

### Using the Web Application

1. Start the web server:
   ```
   python app.py
   ```

2. Open your web browser and go to http://localhost:5000

3. Upload a brain MRI scan image (JPG or PNG format)

4. Click "Analyze MRI Scan" to process the image

5. View the classification results, including:
   - Predicted tumor type (or no tumor)
   - Confidence score for the prediction
   - Probability distribution for all classes
   - Brief description of the detected tumor type

## Model Information

The classification model uses MobileNetV2, a lightweight CNN architecture optimized for mobile and embedded vision applications. The model is trained on a dataset of brain MRI scans containing examples of meningioma, glioma, pituitary tumors, and scans with no tumors.

### Dataset

The model is trained on the following dataset:
- Training dataset: 5,712 MRI images
- Testing dataset: 1,311 MRI images

### Model Architecture

- Base model: MobileNetV2 (pre-trained on ImageNet)
- Custom classification head with Global Average Pooling, Dense layers, and Dropout
- Training approach: Transfer learning with fine-tuning

## Directory Structure

- `app.py`: Main Flask application
- `train_model.py`: Script to train the CNN model
- `app/models/`: Directory for model files
  - `brain_tumor_classifier.h5`: Trained model file
  - `predict.py`: Utility for loading model and making predictions
- `app/templates/`: HTML templates for the web interface
- `app/static/`: CSS, JavaScript, and uploaded images
- `Training/`: Training dataset
- `Testing/`: Testing dataset

## Disclaimer

This application is for educational and demonstration purposes only. It is not intended to be used for medical diagnosis. The predictions made by this system should not be used as a substitute for professional medical advice, diagnosis, or treatment. 