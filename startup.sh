#!/bin/bash
set -e

echo "Running startup initialization for Brain Tumor Detection App"

# Create required directories
mkdir -p app/static/uploads
mkdir -p app/database
mkdir -p app/models
mkdir -p app/utils
mkdir -p app/templates

# Create necessary package files if they don't exist
if [ ! -f "app/__init__.py" ]; then
    echo "# Package initialization" > app/__init__.py
fi

if [ ! -f "app/models/__init__.py" ]; then
    echo "# Models package" > app/models/__init__.py
fi

if [ ! -f "app/utils/__init__.py" ]; then
    echo "# Utils package" > app/utils/__init__.py
fi

# Set proper permissions
chmod -R 755 app/static/uploads
chmod -R 755 app/database
chmod -R 755 app/models
chmod -R 755 app/utils

# Print debug info
echo "Current directory: $(pwd)"
ls -la
echo "App directory:"
ls -la app/
echo "Models directory:"
ls -la app/models/

# Start the application with gunicorn
echo "Starting application with gunicorn..."
export PYTHONPATH=$(pwd):$PYTHONPATH
exec gunicorn server:app 