#!/bin/bash

# Script to improve meningioma detection in the brain tumor classifier model

# Display header
echo "============================================================="
echo "    Brain Tumor Classifier - Meningioma Improvement Script    "
echo "============================================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if training script exists
if [ ! -f "simpler_meningioma_train.py" ]; then
    echo "Error: simpler_meningioma_train.py not found in current directory"
    exit 1
fi

# Create necessary directories
echo "Setting up directories..."
mkdir -p app/models
mkdir -p app/static

# Check if Training and Testing directories exist
if [ ! -d "Training" ] || [ ! -d "Testing" ]; then
    echo "Warning: Training or Testing directories not found!"
    echo "Please ensure your data is organized as follows:"
    echo "  - Training/"
    echo "    - glioma/"
    echo "    - meningioma/"
    echo "    - notumor/"
    echo "    - pituitary/"
    echo "  - Testing/"
    echo "    - (same structure as Training)"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if model exists
if [ ! -f "app/models/brain_tumor_classifier.h5" ]; then
    echo "Warning: Model file not found at app/models/brain_tumor_classifier.h5"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install requirements if needed
echo "Checking requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Run the training script
echo ""
echo "Starting meningioma improvement training..."
echo ""
python simpler_meningioma_train.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================="
    echo "    Training completed successfully!                         "
    echo "============================================================="
    echo ""
    echo "Performance metrics have been saved to:"
    echo "  - app/static/meningioma_training_history.png"
    echo "  - app/static/meningioma_improved_cm.png"
    echo ""
    echo "The improved model has been saved to:"
    echo "  - app/models/brain_tumor_classifier.h5"
    echo ""
    echo "Your original model has been backed up to:"
    echo "  - app/models/brain_tumor_classifier.h5.backup"
else
    echo ""
    echo "============================================================="
    echo "    Training encountered an error                           "
    echo "============================================================="
    echo ""
    echo "Please check the error messages above for details."
    echo "For troubleshooting, refer to MENINGIOMA_TRAINING_GUIDE.md"
fi 