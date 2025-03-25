# Brain Tumor Detection System

A Flask web application for brain tumor detection and classification using machine learning.

## Overview

This application uses a trained convolutional neural network to detect and classify brain tumors from MRI scans into four categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

## Features

- Upload and analyze MRI scan images
- View detection results with confidence scores
- Patient data management
- Detailed scan reports
- User-friendly interface

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`

## Deployment

This application is configured for deployment on Render.

## Technologies Used

- Python
- Flask
- TensorFlow
- Keras
- HTML/CSS/JavaScript
- SQLite

## License

MIT License
