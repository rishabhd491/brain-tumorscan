import os
from flask import render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

app = SQLAlchemy()

@app.route('/')
def index():
    model_loaded = os.path.exists(os.path.join(app.config['MODEL_PATH'], 'tumor_model.h5'))
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/patients')
def patients():
    # Fetch all patients from the database
    patients = Patient.query.all()
    return render_template('patients.html', patients=patients)

@app.route('/patients/register', methods=['GET', 'POST'])
def register_patient():
    if request.method == 'POST':
        # Get form data
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        dob = request.form.get('dob')
        gender = request.form.get('gender')
        contact_number = request.form.get('contact_number')
        email = request.form.get('email')
        address = request.form.get('address')
        medical_history = request.form.get('medical_history')
        
        # Validate required fields
        if not first_name or not last_name or not dob:
            flash('First name, last name, and date of birth are required')
            return redirect(url_for('register_patient'))
        
        # Create new patient record
        new_patient = Patient(
            first_name=first_name,
            last_name=last_name,
            dob=dob,
            gender=gender,
            contact_number=contact_number,
            email=email,
            address=address,
            medical_history=medical_history,
            registered_date=datetime.now(),
            is_visible=True
        )
        
        # Save to database
        db.session.add(new_patient)
        db.session.commit()
        
        flash(f'Patient {first_name} {last_name} registered successfully!')
        return redirect(url_for('patient_detail', patient_id=new_patient.id))
    
    return render_template('register_patient.html')

@app.route('/patients/<int:patient_id>')
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    scans = Scan.query.filter_by(patient_id=patient_id).all()
    return render_template('patient_detail.html', patient=patient, scans=scans)

@app.route('/patients/<int:patient_id>/scan', methods=['GET', 'POST'])
def patient_scan(patient_id):
    # Check if the model is loaded
    model_loaded = os.path.exists(os.path.join(app.config['MODEL_PATH'], 'tumor_model.h5'))
    if not model_loaded:
        flash('Model not loaded. Please train the model first.')
        return redirect(url_for('patient_detail', patient_id=patient_id))
    
    patient = Patient.query.get_or_404(patient_id)
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Process the image and get prediction
            prediction, confidence = predict_image(file)
            
            # Save original file with timestamp to avoid overwriting
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            original_filename = f"{timestamp}_{filename}"
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(original_path)
            
            # Create a new scan record
            new_scan = Scan(
                patient_id=patient_id,
                scan_date=datetime.now(),
                original_image=original_filename,
                prediction=prediction,
                confidence=confidence,
                notes=request.form.get('notes', '')
            )
            
            db.session.add(new_scan)
            db.session.commit()
            
            return redirect(url_for('scan_result', scan_id=new_scan.id))
    
    return render_template('patient_scan.html', patient=patient)

@app.route('/patients/<int:patient_id>/toggle-visibility', methods=['POST'])
def toggle_patient_visibility(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient.is_visible = not patient.is_visible
    db.session.commit()
    return jsonify({'message': f'Patient visibility toggled to {patient.is_visible}', 'visible': patient.is_visible})

@app.route('/scans/<int:scan_id>')
def scan_result(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    patient = Patient.query.get_or_404(scan.patient_id)
    return render_template('scan_result.html', scan=scan, patient=patient)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the model is loaded
    model_loaded = os.path.exists(os.path.join(app.config['MODEL_PATH'], 'tumor_model.h5'))
    if not model_loaded:
        flash('Model not loaded. Please train the model first.')
        return redirect(url_for('index'))
    
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Process the image and get prediction
        prediction, confidence = predict_image(file)
        
        # Save the file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(file_path)
        
        # Create a new anonymous scan record
        new_scan = Scan(
            scan_date=datetime.now(),
            original_image=saved_filename,
            prediction=prediction,
            confidence=confidence,
            notes="Quick scan (no patient)"
        )
        
        db.session.add(new_scan)
        db.session.commit()
        
        return redirect(url_for('scan_result', scan_id=new_scan.id))
    
    else:
        flash('File type not allowed')
        return redirect(url_for('index')) 