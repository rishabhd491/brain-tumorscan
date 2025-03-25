#!/bin/bash
set -e

echo "Running startup initialization for Brain Tumor Detection App"

# Create required directories
mkdir -p app/static/uploads
mkdir -p app/database

# Set proper permissions
chmod -R 755 app/static/uploads
chmod -R 755 app/database

# Start the application with gunicorn
echo "Starting application with gunicorn..."
exec gunicorn app:app 