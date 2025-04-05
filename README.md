# Brain Tumor Detection System

A comprehensive Flask web application for brain tumor detection and classification using advanced machine learning techniques.

## Overview

This application leverages state-of-the-art convolutional neural networks to detect and classify brain tumors from MRI scans. The system provides accurate classification into four categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

## Key Features

### Image Analysis
- Upload and analyze MRI scan images in various formats
- Real-time tumor detection and classification
- Confidence score visualization
- Detailed scan analysis reports

### Patient Management
- Secure patient data storage
- Historical scan tracking
- Comprehensive patient profiles
- Report generation and export

### Advanced Features
- Multiple training models for specialized tumor detection
- Model improvement scripts for continuous learning
- High-performance image processing
- Scalable architecture for deployment

## Technical Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, scikit-learn
- **Image Processing**: OpenCV, Pillow
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite
- **Deployment**: Render, Gunicorn

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment support

### Setup Steps

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd brain-tumorscan-main
   ```

2. Create and activate virtual environment:
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the application at `http://localhost:5000`

## Deployment

The application is configured for deployment on Render. To deploy:

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following build command:
   ```bash
   ./build.sh
   ```
4. Set the start command to:
   ```bash
   ./startup.sh
   ```

## Model Training

The project includes several specialized training scripts:
- `meningioma_focus_train.py`: Focused training for meningioma detection
- `meningioma_specialist_train.py`: Advanced meningioma classification
- `train_model.py`: General tumor classification training
- `improve_meningioma.py`: Model improvement and fine-tuning

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- TensorFlow and Keras teams for the ML framework
- Flask community for the web framework
- Medical imaging research community
