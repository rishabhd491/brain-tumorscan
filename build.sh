#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

echo "Starting build script for Brain Tumor Detection App"

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Create necessary directories
echo "Creating application directories..."
mkdir -p app/static/uploads
mkdir -p app/database

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify model files
echo "Verifying model files..."
if [ -f "app/models/brain_tumor_classifier.h5" ]; then
    echo "Model file found!"
else
    echo "Warning: Model file not found. The application may not function properly."
    # You might want to add a download step here if you're hosting your model somewhere
    # Example: wget -O app/models/brain_tumor_classifier.h5 https://your-storage-url/brain_tumor_classifier.h5
fi

# Set permissions
echo "Setting file permissions..."
chmod -R 755 app/static/uploads
chmod -R 755 app/database

echo "Build process completed!" 