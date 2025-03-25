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
mkdir -p app/models
mkdir -p app/utils
mkdir -p app/templates

# Create empty __init__.py files to make packages
echo "Creating Python package structures..."
touch app/__init__.py
touch app/models/__init__.py
touch app/utils/__init__.py

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify model files
echo "Verifying model files..."
if [ -f "app/models/brain_tumor_classifier.h5" ]; then
    echo "Model file found!"
else
    echo "Warning: Model file not found. Creating a placeholder file..."
    # Create an empty file so the app can start without errors
    touch app/models/brain_tumor_classifier.h5
    echo "Placeholder model file created. The application will start, but prediction functionality will be disabled."
fi

# Set permissions
echo "Setting file permissions..."
chmod -R 755 app/static/uploads
chmod -R 755 app/database
chmod -R 755 app/models

echo "Build process completed!" 