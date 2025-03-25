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

# Create required files if they don't exist
if [ ! -f "app/models/tumor_info.py" ]; then
    echo "Creating tumor_info.py placeholder..."
    cat > app/models/tumor_info.py << 'EOF'
# Placeholder tumor info module
tumor_types = {
    'glioma': {
        'name': 'Glioma',
        'description': 'A tumor that starts in the glial cells of the brain or spine.',
        'symptoms': ['Headaches', 'Seizures', 'Personality changes', 'Nausea', 'Vision problems'],
        'treatment': 'Treatment typically includes surgery, radiation therapy, and chemotherapy.',
        'prognosis': 'Prognosis varies depending on the grade and location of the tumor.'
    },
    'meningioma': {
        'name': 'Meningioma',
        'description': 'A tumor that forms on membranes that cover the brain and spinal cord just inside the skull.',
        'symptoms': ['Headaches', 'Hearing loss', 'Vision problems', 'Memory loss', 'Seizures'],
        'treatment': 'Treatment often includes surgery and sometimes radiation therapy.',
        'prognosis': 'Most meningiomas are benign and grow slowly, with a favorable prognosis.'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'A tumor that forms in the pituitary gland at the base of the brain.',
        'symptoms': ['Headaches', 'Vision problems', 'Hormonal imbalances', 'Fatigue', 'Mood changes'],
        'treatment': 'Treatment may include surgery, radiation therapy, or medication to control hormone production.',
        'prognosis': 'Most pituitary tumors are benign and have a good prognosis with proper treatment.'
    },
    'notumor': {
        'name': 'No Tumor',
        'description': 'No evidence of tumor in the brain tissue.',
        'symptoms': ['N/A - No tumor detected'],
        'treatment': 'No treatment needed for brain tumor. Any symptoms may be due to other conditions.',
        'prognosis': 'Excellent - no brain tumor present.'
    }
}

def get_tumor_info(tumor_type):
    """Get information about a specific tumor type."""
    tumor_type = tumor_type.lower().replace(' ', '')
    return tumor_types.get(tumor_type)

def get_all_tumor_types():
    """Get a list of all available tumor types."""
    return list(tumor_types.keys())
EOF
fi

# Create report generator placeholder if it doesn't exist
if [ ! -f "app/utils/report_generator.py" ]; then
    echo "Creating report_generator.py placeholder..."
    cat > app/utils/report_generator.py << 'EOF'
# Placeholder report generator module
def generate_scan_report(patient, scan, tumor_info=None):
    """
    Generate a PDF report for a scan.
    This is a placeholder function that returns None.
    To enable actual PDF generation, install reportlab and implement the function.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import io
        
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        
        # Just return empty bytes as a placeholder
        return buffer.getvalue()
    except ImportError:
        print("ReportLab not installed. PDF report generation is not available.")
        return None
    except Exception as e:
        print(f"Error generating report: {e}")
        return None
EOF
fi

# Set permissions
echo "Setting file permissions..."
chmod -R 755 app/static/uploads
chmod -R 755 app/database
chmod -R 755 app/models
chmod -R 755 app/utils

echo "Build process completed!" 