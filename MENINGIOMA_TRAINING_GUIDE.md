# Meningioma Detection Improvement Guide

This guide explains how to use the `simpler_meningioma_train.py` script to improve the model's ability to correctly identify meningioma brain tumors.

## Background

The model has shown good overall performance but some weakness in detecting meningioma tumors. This script is designed to address this specific issue by:

1. Applying extra data augmentation
2. Using class weighting (3x weight for meningioma)
3. Fine-tuning only the last 50 layers of the model
4. Using a very low learning rate (0.00005)
5. Implementing early stopping and learning rate reduction

## Requirements

Make sure you have all required packages installed:

```bash
pip install -r requirements.txt
```

## Directory Structure

Ensure your project has the following structure:
```
brain tumor project/
├── Training/              # Training data directory with class subdirectories
│   ├── glioma/           # Contains glioma images
│   ├── meningioma/       # Contains meningioma images 
│   ├── notumor/          # Contains non-tumor images
│   └── pituitary/        # Contains pituitary tumor images
├── Testing/               # Testing data with same structure as Training
├── app/
│   ├── models/           # Where the model will be saved
│   └── static/           # Where performance plots will be saved
├── requirements.txt
└── simpler_meningioma_train.py
```

## Running the Training Script

To run the training script:

```bash
python simpler_meningioma_train.py
```

## What Happens During Training

1. The script loads the existing model from `app/models/brain_tumor_classifier.h5`
2. It creates a backup of your original model as `brain_tumor_classifier.h5.backup`
3. It applies data augmentation with emphasis on meningioma class
4. It trains for up to 15 epochs (may stop earlier due to early stopping)
5. It saves training history plots to `app/static/meningioma_training_history.png`
6. It generates a confusion matrix at `app/static/meningioma_improved_cm.png`
7. The improved model is saved to the original path `app/models/brain_tumor_classifier.h5`

## Evaluating Results

After training, check the console output for:
- Meningioma-specific performance metrics (precision, recall, F1-score)
- Overall model accuracy
- Meningioma recognition rate

Also examine the generated confusion matrix to see how meningioma classification has improved.

## Reverting to Original Model

If you need to revert to the original model, simply:

```bash
# Navigate to your app/models directory
cd app/models

# Remove the current model 
rm brain_tumor_classifier.h5

# Rename the backup to the original name
mv brain_tumor_classifier.h5.backup brain_tumor_classifier.h5
```

## Troubleshooting

If you encounter errors:

1. **Memory issues**: Reduce `BATCH_SIZE` in the script
2. **Loading model errors**: Ensure the model path is correct
3. **Directory errors**: Make sure your Training and Testing directories exist with the right structure 