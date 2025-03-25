import os
import sys
import uuid
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, send_from_directory, send_file, current_app
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the path so Python can find the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add app directory to path for proper imports
app_dir = os.path.join(current_dir, 'app')
if os.path.exists(app_dir) and app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Add models directory to path
models_dir = os.path.join(current_dir, 'app', 'models')
if os.path.exists(models_dir) and models_dir not in sys.path:
    sys.path.insert(0, models_dir)

# Add utils directory to path
utils_dir = os.path.join(current_dir, 'app', 'utils')
if os.path.exists(utils_dir) and utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

# Print paths for debugging (only when deploying)
if 'RENDER' in os.environ:
    print("Python path:", sys.path)
    print("Current directory:", current_dir)
    print("App directory:", app_dir)
    print("Models directory:", models_dir)
    print("Utils directory:", utils_dir)

# Suppress TensorFlow logging except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Import prediction module
try:
    # First attempt - normal import
    from app.models.predict import load_brain_tumor_model, predict_tumor_type
    model_loaded = True
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Second attempt - direct import
    try:
        from predict import load_brain_tumor_model, predict_tumor_type
        model_loaded = True
    except Exception as e:
        logger.error(f"Second attempt failed: {e}")
        # Third attempt - add path and import
        try:
            sys.path.append(os.path.join(current_dir, 'app', 'models'))
            from predict import load_brain_tumor_model, predict_tumor_type
            model_loaded = True
        except Exception as e:
            logger.error(f"Third attempt failed: {e}")
            model_loaded = False

# Import patient management module
try:
    # First attempt
    from app.models.patient import (
        init_db, add_patient, add_scan, get_patient, get_patient_scans,
        get_all_patients, search_patients, get_scan, update_scan
    )
    patient_module_loaded = True
except Exception as e:
    logger.error(f"Error loading patient module: {e}")
    # Second attempt
    try:
        from patient import (
            init_db, add_patient, add_scan, get_patient, get_patient_scans,
            get_all_patients, search_patients, get_scan, update_scan
        )
        patient_module_loaded = True
    except Exception as e:
        logger.error(f"Second attempt failed for patient module: {e}")
        # Third attempt
        try:
            sys.path.append(os.path.join(current_dir, 'app', 'models'))
            from patient import (
                init_db, add_patient, add_scan, get_patient, get_patient_scans,
                get_all_patients, search_patients, get_scan, update_scan
            )
            patient_module_loaded = True
        except Exception as e:
            logger.error(f"Third attempt failed for patient module: {e}")
            patient_module_loaded = False

# Import tumor information module
try:
    from app.models.tumor_info import get_tumor_info, get_all_tumor_types
    tumor_info_loaded = True
except Exception as e:
    logger.error(f"Error loading tumor info: {e}")
    try:
        sys.path.append(os.path.join(current_dir, 'app', 'models'))
        from tumor_info import get_tumor_info, get_all_tumor_types
        tumor_info_loaded = True
    except Exception as e:
        logger.error(f"Second attempt failed for tumor info: {e}")
        tumor_info_loaded = False

# Import report generator
try:
    from app.utils.report_generator import generate_scan_report
    report_generator_loaded = True
except Exception as e:
    logger.error(f"Error loading report generator: {e}")
    try:
        sys.path.append(os.path.join(current_dir, 'app', 'utils'))
        from report_generator import generate_scan_report
        report_generator_loaded = True
    except Exception as e:
        logger.error(f"Second attempt failed for report generator: {e}")
        report_generator_loaded = False

# Initialize Flask app
app = Flask(__name__, 
            static_folder='app/static',
            template_folder='app/templates')

# Configure app
app.config['SECRET_KEY'] = os.urandom(24)

# Set upload folder - use Render disk mountpoint if available
if 'RENDER' in os.environ:
    app.config['UPLOAD_FOLDER'] = '/opt/render/project/src/app/static/uploads'
else:
    app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['PATIENTS_PER_PAGE'] = 10  # Pagination setting

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('app', 'database'), exist_ok=True)

# For Render deployment: If we're on Render, ensure paths are correct
if 'RENDER' in os.environ:
    logger.info("Running on Render. Setting up environment...")
    # Log the current working directory for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    # Force create the upload and database directories with absolute paths
    render_upload_path = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])
    render_db_path = os.path.join(os.getcwd(), 'app', 'database')
    logger.info(f"Creating upload directory at: {render_upload_path}")
    logger.info(f"Creating database directory at: {render_db_path}")
    os.makedirs(render_upload_path, exist_ok=True)
    os.makedirs(render_db_path, exist_ok=True)
    
    # Update static URL path for Render
    app.static_url_path = '/static'
    app.static_folder = os.path.join(os.getcwd(), 'app/static')
    logger.info(f"Static folder set to: {app.static_folder}")

# Add path_exists function to Jinja environment
@app.context_processor
def utility_functions():
    def path_exists(path):
        return os.path.exists(path)
    return dict(path_exists=path_exists)

# Initialize database if patient module is loaded
if patient_module_loaded:
    init_db()

# Initialize the patient list cache at startup
patients_list = []

def refresh_patients_list():
    """Refresh the in-memory patients list from the database"""
    global patients_list
    logger.info("Refreshing patients list from database")
    try:
        # Get all patients from the database
        all_patients = get_all_patients()
        
        # Create a new list to store enhanced patient info
        enhanced_patients = []
        
        # Enhance each patient with additional information
        for patient in all_patients:
            # Get scans for this patient
            patient_scans = get_patient_scans(patient['id'])
            scan_count = len(patient_scans)
            
            # Add scan count to patient info
            patient['scan_count'] = scan_count
            
            # Add most recent scan if available
            if scan_count > 0:
                # Sort scans by date (most recent first)
                sorted_scans = sorted(patient_scans, key=lambda x: x.get('scan_date', ''), reverse=True)
                patient['latest_scan'] = sorted_scans[0]
            else:
                patient['latest_scan'] = None
                
            enhanced_patients.append(patient)
            
        # Update the global list
        patients_list = enhanced_patients
        logger.info(f"Patients list refreshed successfully with {len(patients_list)} patients")
        return True
    except Exception as e:
        logger.error(f"Error refreshing patients list: {str(e)}")
        return False

# Call refresh at startup
refresh_patients_list()

# Load the model if available
model = None
if model_loaded:
    try:
        model = load_brain_tumor_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't crash the app if model loading fails - we'll handle this gracefully in the UI
        flash("Warning: Brain tumor model could not be loaded. Prediction functionality will be disabled.")

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html', model_loaded=model is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction (legacy route)."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # Make prediction if model is loaded
        if model is not None:
            try:
                result = predict_tumor_type(model, file_path)
                return render_template('result.html', 
                                      result=result, 
                                      image_path=f"uploads/{unique_filename}")
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                flash(f"Error during prediction: {str(e)}")
                return redirect(url_for('index'))
        else:
            flash("Model not loaded. Please train the model first.")
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload PNG or JPG images.')
        return redirect(url_for('index'))

# Patient Management Routes

@app.route('/patients')
def patients():
    """Display patient list with pagination and search."""
    if not patient_module_loaded:
        flash("Patient management functionality is not available.")
        return redirect(url_for('index'))

    # Get search query, if any
    search_query = request.args.get('search', None)
    
    # Get page number
    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        page = 1
    
    # Get patients from in-memory list or database
    if search_query:
        # If search query, search directly in the database for accuracy
        all_patients = search_patients(search_query)
    else:
        # Use the in-memory list for faster access
        all_patients = patients_list
    
    # Pagination
    per_page = app.config['PATIENTS_PER_PAGE']
    total_pages = (len(all_patients) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    patients_page = all_patients[start_idx:end_idx]
    
    # Create pagination info
    pagination = {
        'page': page,
        'pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages
    }
    
    return render_template('patients.html', patients=patients_page, pagination=pagination)

@app.route('/patients/register', methods=['GET', 'POST'])
def register_patient():
    """Register a new patient."""
    if not patient_module_loaded:
        flash("Patient management functionality is not available.")
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Extract form data
        name = request.form.get('name')
        try:
            age = int(request.form.get('age'))
        except (ValueError, TypeError):
            flash("Please enter a valid age.")
            return redirect(url_for('register_patient'))
        
        gender = request.form.get('gender')
        contact = request.form.get('contact')
        email = request.form.get('email')
        address = request.form.get('address')
        medical_history = request.form.get('medical_history')
        
        # Validate required fields
        if not (name and age and gender):
            flash("Please fill all required fields.")
            return redirect(url_for('register_patient'))
        
        # Add patient to database
        patient_id = add_patient(name, age, gender, contact, email, address, medical_history)
        
        if patient_id:
            # Refresh patients list after adding a new patient
            refresh_patients_list()
            
            flash(f"Patient {name} registered successfully.")
            return redirect(url_for('patient_detail', patient_id=patient_id))
        else:
            flash("Failed to register patient. Please try again.")
            return redirect(url_for('register_patient'))
    
    # GET request
    return render_template('patient_form.html')

@app.route('/patients/<int:patient_id>')
def patient_detail(patient_id):
    """Display patient details and scan history."""
    if not patient_module_loaded:
        flash("Patient management functionality is not available.")
        return redirect(url_for('index'))
    
    # Try to find patient in the in-memory list first for faster access
    patient = next((p for p in patients_list if p['id'] == patient_id), None)
    
    # If not found in list or list is empty, get from database
    if not patient:
        patient = get_patient(patient_id)
    
    if not patient:
        flash("Patient not found.")
        return redirect(url_for('patients'))
    
    # Get scans - try to use scans from in-memory patient data first
    if 'scans' in patient and patient['scans']:
        scans = patient['scans']
    else:
        # Otherwise fetch from database
        scans = get_patient_scans(patient_id)
    
    return render_template('patient_detail.html', patient=patient, scans=scans)

@app.route('/patients/<int:patient_id>/edit', methods=['GET', 'POST'])
def edit_patient(patient_id):
    """Edit patient information."""
    if not patient_module_loaded:
        flash("Patient management functionality is not available.")
        return redirect(url_for('index'))
    
    patient = get_patient(patient_id)
    
    if not patient:
        flash("Patient not found.")
        return redirect(url_for('patients'))
    
    if request.method == 'POST':
        # Extract and update patient info
        # (Implementation would go here)
        flash("Patient information updated.")
        return redirect(url_for('patient_detail', patient_id=patient_id))
    
    # GET request
    return render_template('patient_form.html', patient=patient)

@app.route('/patients/<int:patient_id>/scan', methods=['GET', 'POST'])
def scan_patient(patient_id):
    """Upload and analyze scan for a specific patient."""
    if not patient_module_loaded or not model_loaded:
        flash("Required functionality is not available.")
        return redirect(url_for('index'))
    
    patient = get_patient(patient_id)
    
    if not patient:
        flash("Patient not found.")
        return redirect(url_for('patients'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        scan_notes = request.form.get('scan_notes', '')
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            rel_file_path = os.path.join('uploads', unique_filename)
            abs_file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(abs_file_path)
            
            # Make prediction
            try:
                result = predict_tumor_type(model, abs_file_path)
                logger.info(f"Prediction result: {result}")
                
                # Check if there was an error
                if 'error' in result:
                    flash(f"Error during analysis: {result['error']}")
                    return redirect(url_for('scan_patient', patient_id=patient_id))
                
                # Handle different result key formats
                tumor_type = None
                if 'predicted_class' in result:
                    tumor_type = result['predicted_class']
                elif 'class' in result:
                    tumor_type = result['class']
                else:
                    flash("Error: Prediction result has unexpected format")
                    return redirect(url_for('scan_patient', patient_id=patient_id))
                
                # Add scan to database using the correct keys from result
                scan_id = add_scan(
                    patient_id=patient_id,
                    image_path=rel_file_path,
                    tumor_type=tumor_type,
                    confidence=result.get('confidence', 0),
                    doctor_notes=scan_notes
                )
                
                if scan_id:
                    # Refresh patients list after adding a new scan
                    refresh_patients_list()
                    
                    return redirect(url_for('scan_detail', scan_id=scan_id))
                else:
                    flash("Failed to save scan data.")
                    return redirect(url_for('patient_detail', patient_id=patient_id))
                
            except Exception as e:
                logger.error(f"Scan processing error: {e}")
                flash(f"Error during scan analysis: {str(e)}")
                return redirect(url_for('scan_patient', patient_id=patient_id))
        else:
            flash('File type not allowed. Please upload PNG or JPG images.')
            return redirect(url_for('scan_patient', patient_id=patient_id))
    
    # GET request
    previous_scans = get_patient_scans(patient_id)
    return render_template('scan_form.html', patient=patient, previous_scans=previous_scans)

@app.route('/scans/<int:scan_id>')
def scan_detail(scan_id):
    """Display scan details with tumor information."""
    if not patient_module_loaded:
        flash("Required functionality is not available.")
        return redirect(url_for('index'))
    
    scan = get_scan(scan_id)
    
    if not scan:
        flash("Scan not found.")
        return redirect(url_for('patients'))
    
    patient = get_patient(scan['patient_id'])
    
    if not patient:
        flash("Patient not found.")
        return redirect(url_for('patients'))
    
    # Get tumor information if available
    tumor_info = None
    if tumor_info_loaded and scan.get('tumor_type'):
        # Normalize tumor type to handle different formats
        tumor_type = scan['tumor_type'].lower().replace(' ', '_')
        tumor_info = get_tumor_info(tumor_type)
        logger.info(f"Tumor info lookup for type '{tumor_type}': {'Found' if tumor_info else 'Not found'}")
        
        # If no tumor info found with the exact match, try a backup lookup strategy
        if not tumor_info:
            # Try to map to one of our known tumor types
            if 'glioma' in tumor_type:
                tumor_info = get_tumor_info('glioma')
            elif 'mening' in tumor_type:
                tumor_info = get_tumor_info('meningioma')
            elif 'pituit' in tumor_type:
                tumor_info = get_tumor_info('pituitary')
            elif 'no' in tumor_type or 'not' in tumor_type or 'healthy' in tumor_type:
                tumor_info = get_tumor_info('notumor')
            
            if tumor_info:
                logger.info(f"Found tumor info using backup lookup strategy for '{tumor_type}'")
    
    return render_template('scan_detail.html', scan=scan, patient=patient, tumor_info=tumor_info)

@app.route('/scans/<int:scan_id>/notes', methods=['POST'])
def update_scan_notes(scan_id):
    """Update doctor's notes for a scan."""
    if not patient_module_loaded:
        flash("Required functionality is not available.")
        return redirect(url_for('index'))
    
    scan = get_scan(scan_id)
    
    if not scan:
        flash("Scan not found.")
        return redirect(url_for('patients'))
    
    doctor_notes = request.form.get('doctor_notes', '')
    
    if update_scan(scan_id, doctor_notes=doctor_notes):
        flash("Scan notes updated successfully.")
    else:
        flash("Failed to update scan notes.")
    
    return redirect(url_for('scan_detail', scan_id=scan_id))

@app.route('/scans/<int:scan_id>/report')
def generate_report(scan_id):
    """Generate and download a PDF report for a scan."""
    if not patient_module_loaded or not report_generator_loaded:
        flash("Report generation is not available.")
        return redirect(url_for('patients'))
    
    # Get scan and patient data
    scan = get_scan(scan_id)
    
    if not scan:
        flash("Scan not found.")
        return redirect(url_for('patients'))
    
    patient = get_patient(scan['patient_id'])
    
    if not patient:
        flash("Patient not found.")
        return redirect(url_for('patients'))
    
    # Get tumor information if available
    tumor_info = None
    if tumor_info_loaded and scan['tumor_type']:
        tumor_info = get_tumor_info(scan['tumor_type'])
    
    # Generate the PDF report
    pdf_data = generate_scan_report(patient, scan, tumor_info)
    
    if not pdf_data:
        flash("Failed to generate report.")
        return redirect(url_for('scan_detail', scan_id=scan_id))
    
    # Return the PDF as a downloadable file
    filename = f"scan_report_{patient['name'].replace(' ', '_')}_{scan_id}.pdf"
    return send_file(
        io.BytesIO(pdf_data),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )

@app.route('/scans/<int:scan_id>/print_report')
def print_report(scan_id):
    """Generate and display a printable HTML report for a scan."""
    if not patient_module_loaded or not report_generator_loaded:
        flash("Report generation is not available.")
        return redirect(url_for('patients'))
    
    # Get scan and patient data
    scan = get_scan(scan_id)
    
    if not scan:
        flash("Scan not found.")
        return redirect(url_for('patients'))
    
    patient = get_patient(scan['patient_id'])
    
    if not patient:
        flash("Patient not found.")
        return redirect(url_for('patients'))
    
    # Get tumor information if available
    tumor_info = None
    if tumor_info_loaded and scan['tumor_type']:
        tumor_info = get_tumor_info(scan['tumor_type'])
    
    # Render the printable report template
    return render_template('print_report.html', 
                          patient=patient,
                          scan=scan,
                          tumor_info=tumor_info,
                          now=datetime.now())

# Other routes

@app.route('/about')
def about():
    """Render the about page with model information."""
    model_summary_path = 'app/models/model_summary.txt'
    model_summary = None
    
    if os.path.exists(model_summary_path):
        with open(model_summary_path, 'r') as f:
            model_summary = f.read()
    
    # Add tumor type information if available
    tumor_types = None
    if tumor_info_loaded:
        tumor_types = []
        for tumor_type in get_all_tumor_types():
            info = get_tumor_info(tumor_type)
            if info:
                tumor_types.append(info)
    
    return render_template('about.html', model_summary=model_summary, tumor_types=tumor_types)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    """Handle server errors."""
    logger.error(f"Server error: {e}")
    return render_template('error.html', error=str(e)), 500

@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_from_directory(os.path.join(app.root_path, 'app/static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/api/patients', methods=['GET'])
def api_get_patients():
    """API endpoint to get all patients as JSON."""
    if not patient_module_loaded:
        return jsonify({"error": "Patient management functionality is not available"}), 503
    
    # Use the in-memory list for faster access
    return jsonify({"patients": patients_list})

@app.route('/api/patients/<int:patient_id>', methods=['GET'])
def api_get_patient(patient_id):
    """API endpoint to get a specific patient's data as JSON."""
    if not patient_module_loaded:
        return jsonify({"error": "Patient management functionality is not available"}), 503
    
    # Try to find patient in the in-memory list first
    patient = next((p for p in patients_list if p['id'] == patient_id), None)
    
    # If not found in list or list is empty, get from database
    if not patient:
        patient = get_patient(patient_id)
    
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    return jsonify({"patient": patient})

@app.route('/patients_list')
def patients_list_route():
    """Display the in-memory patients list"""
    return render_template('patients_list.html', patients=patients_list)

@app.route('/refresh_patients')
def refresh_patients_route():
    """Refresh the patients list and redirect back to patients list page"""
    refresh_patients_list()
    flash('Patients list has been refreshed from the database')
    return redirect(url_for('patients_list_route'))

# Add an error handler for common errors
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle exceptions gracefully."""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return render_template('error.html', 
                          error=str(e), 
                          title="An Error Occurred", 
                          message="The application encountered an unexpected error."), 500

if __name__ == '__main__':
    if not model_loaded:
        logger.warning("Model not loaded. Please run train_model.py first.")
    
    if not patient_module_loaded:
        logger.warning("Patient management module not loaded.")
    
    if not tumor_info_loaded:
        logger.warning("Tumor information module not loaded.")
    
    # Run the app on port 5001 to avoid conflict with AirPlay on macOS
    app.run(debug=True, host='0.0.0.0', port=5005)