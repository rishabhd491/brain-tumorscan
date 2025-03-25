import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
import logging

logger = logging.getLogger(__name__)

def generate_scan_report(patient, scan, tumor_info=None):
    """
    Generate a PDF report for a scan
    
    Args:
        patient (dict): Patient information
        scan (dict): Scan information
        tumor_info (dict, optional): Tumor type information
        
    Returns:
        bytes: PDF file as bytes
    """
    try:
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
            title=f"Brain MRI Scan Report - {patient['name']}"
        )
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        title_style.alignment = 1  # Center alignment
        
        section_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Custom styles
        bold_style = ParagraphStyle(
            'Bold',
            parent=styles['Normal'],
            fontName='Helvetica-Bold'
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Title
        elements.append(Paragraph("Brain MRI Scan Report", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Current date
        current_date = datetime.now().strftime("%B %d, %Y")
        elements.append(Paragraph(f"Report Date: {current_date}", normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Patient Information
        elements.append(Paragraph("Patient Information", section_style))
        
        patient_data = [
            ["Name:", patient.get('name', 'N/A')],
            ["Patient ID:", str(patient.get('id', 'N/A'))],
            ["Age:", f"{patient.get('age', 'N/A')} years"],
            ["Gender:", patient.get('gender', 'N/A')],
            ["Contact:", patient.get('contact', 'N/A')],
            ["Medical History:", patient.get('medical_history', 'None provided')]
        ]
        
        patient_table = Table(patient_data, colWidths=[1.5*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        
        elements.append(patient_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Scan Information
        elements.append(Paragraph("Scan Information", section_style))
        
        # Format scan date
        scan_date = scan.get('scan_date', 'N/A')
        if isinstance(scan_date, datetime):
            scan_date = scan_date.strftime("%B %d, %Y")
        
        scan_data = [
            ["Scan ID:", str(scan.get('id', 'N/A'))],
            ["Scan Date:", scan_date],
            ["Predicted Tumor Type:", scan.get('tumor_type', 'N/A').replace('_', ' ').title()],
            ["Confidence:", f"{scan.get('confidence', 0) * 100:.1f}%"]
        ]
        
        scan_table = Table(scan_data, colWidths=[1.5*inch, 4*inch])
        scan_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        
        elements.append(scan_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add scan image if available
        from flask import current_app
        if scan.get('image_path'):
            elements.append(Paragraph("MRI Scan Image", section_style))
            
            # Get the absolute path to the image
            image_path = os.path.join(
                current_app.root_path, 
                'static', 
                scan['image_path'].replace('uploads/', '')
            )
            
            # Check if file exists
            if os.path.exists(image_path):
                # Add image, scaled to fit
                img = Image(image_path)
                img.drawHeight = 3*inch
                img.drawWidth = 4*inch
                elements.append(img)
                elements.append(Spacer(1, 0.25*inch))
            else:
                elements.append(Paragraph("Image file not found.", normal_style))
                elements.append(Spacer(1, 0.25*inch))
        
        # Tumor Information
        if tumor_info:
            elements.append(Paragraph("Tumor Information", section_style))
            
            tumor_data = [
                ["Type:", tumor_info.get('name', 'N/A')],
                ["Description:", tumor_info.get('description', 'N/A')],
                ["Symptoms:", tumor_info.get('symptoms', 'N/A')],
                ["Treatments:", tumor_info.get('treatments', 'N/A')],
                ["Prognosis:", tumor_info.get('prognosis', 'N/A')]
            ]
            
            tumor_table = Table(tumor_data, colWidths=[1.5*inch, 4*inch])
            tumor_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            
            elements.append(tumor_table)
            elements.append(Spacer(1, 0.25*inch))
        
        # Doctor's Notes
        elements.append(Paragraph("Doctor's Notes", section_style))
        doctor_notes = scan.get('doctor_notes', 'No notes provided.')
        elements.append(Paragraph(doctor_notes, normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Disclaimer
        elements.append(Paragraph("Disclaimer", section_style))
        disclaimer_text = """This report is generated based on a machine learning model's analysis of the MRI scan.
        The prediction should be verified by a qualified medical professional. This tool is intended to assist
        healthcare providers and should not be used as the sole basis for diagnosis or treatment decisions."""
        elements.append(Paragraph(disclaimer_text, normal_style))
        
        # Build the PDF
        doc.build(elements)
        
        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None 