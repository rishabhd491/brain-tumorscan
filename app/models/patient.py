from datetime import datetime
import sqlite3
import os
import json
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join('app', 'database', 'patients.db')

# If running on Render, use absolute path
if 'RENDER' in os.environ:
    DB_PATH = os.path.join(os.getcwd(), 'app', 'database', 'patients.db')
    logger.info(f"Running on Render. Using database path: {DB_PATH}")

# Ensure database directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    """Initialize the database with required tables."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create patients table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            contact TEXT,
            email TEXT,
            address TEXT,
            medical_history TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create scans table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT NOT NULL,
            tumor_type TEXT,
            confidence REAL,
            doctor_notes TEXT,
            additional_info TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False
    finally:
        if conn:
            conn.close()

def add_patient(name, age, gender, contact=None, email=None, address=None, medical_history=None):
    """Add a new patient to the database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO patients (name, age, gender, contact, email, address, medical_history)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, age, gender, contact, email, address, medical_history))
        
        patient_id = cursor.lastrowid
        conn.commit()
        
        logger.info(f"Added patient {name} with ID {patient_id}")
        return patient_id
    except Exception as e:
        logger.error(f"Error adding patient: {e}")
        return None
    finally:
        if conn:
            conn.close()

def add_scan(patient_id, image_path, tumor_type=None, confidence=None, doctor_notes=None, additional_info=None):
    """Add a new scan for a patient."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert additional_info to JSON if it's a dictionary
        if isinstance(additional_info, dict):
            additional_info = json.dumps(additional_info)
        
        cursor.execute('''
        INSERT INTO scans (patient_id, image_path, tumor_type, confidence, doctor_notes, additional_info)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (patient_id, image_path, tumor_type, confidence, doctor_notes, additional_info))
        
        scan_id = cursor.lastrowid
        conn.commit()
        
        logger.info(f"Added scan for patient ID {patient_id}, scan ID {scan_id}")
        return scan_id
    except Exception as e:
        logger.error(f"Error adding scan: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_patient(patient_id):
    """Get patient details by ID."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        patient = cursor.fetchone()
        
        if patient:
            return dict(patient)
        return None
    except Exception as e:
        logger.error(f"Error retrieving patient {patient_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_patient_scans(patient_id):
    """Get all scans for a patient."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM scans WHERE patient_id = ? ORDER BY scan_date DESC', (patient_id,))
        scans = cursor.fetchall()
        
        return [dict(scan) for scan in scans]
    except Exception as e:
        logger.error(f"Error retrieving scans for patient {patient_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_all_patients():
    """Get all patients."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
        patients = cursor.fetchall()
        
        return [dict(patient) for patient in patients]
    except Exception as e:
        logger.error(f"Error retrieving all patients: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_scan(scan_id):
    """Get scan details by ID."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM scans WHERE id = ?', (scan_id,))
        scan = cursor.fetchone()
        
        if scan:
            return dict(scan)
        return None
    except Exception as e:
        logger.error(f"Error retrieving scan {scan_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def search_patients(query):
    """Search patients by name or other details."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        search_param = f"%{query}%"
        cursor.execute('''
        SELECT * FROM patients 
        WHERE name LIKE ? OR contact LIKE ? OR email LIKE ? OR address LIKE ?
        ORDER BY created_at DESC
        ''', (search_param, search_param, search_param, search_param))
        
        patients = cursor.fetchall()
        return [dict(patient) for patient in patients]
    except Exception as e:
        logger.error(f"Error searching patients: {e}")
        return []
    finally:
        if conn:
            conn.close()

def update_scan(scan_id, tumor_type=None, confidence=None, doctor_notes=None, additional_info=None):
    """Update scan details."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert additional_info to JSON if it's a dictionary
        if isinstance(additional_info, dict):
            additional_info = json.dumps(additional_info)
        
        cursor.execute('''
        UPDATE scans 
        SET tumor_type = COALESCE(?, tumor_type),
            confidence = COALESCE(?, confidence),
            doctor_notes = COALESCE(?, doctor_notes),
            additional_info = COALESCE(?, additional_info)
        WHERE id = ?
        ''', (tumor_type, confidence, doctor_notes, additional_info, scan_id))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating scan {scan_id}: {e}")
        return False
    finally:
        if conn:
            conn.close() 