# Hepatic Steatosis Diagnosis from Ultrasound Images

## Team Members
- **Felipe Campolina** (762732)
- **Leandro Guido** (777801)  
- **Marcelo Augusto** (775119)

## Project Overview

This project implements a comprehensive computer-aided diagnosis system for **hepatic steatosis** (fatty liver disease) detection using ultrasound B-mode images. The system combines traditional machine learning approaches with deep learning techniques to provide accurate classification between healthy and steatotic liver tissue.

### Clinical Context

Hepatic steatosis is characterized by excessive fat accumulation in liver cells, commonly associated with obesity, diabetes, and metabolic syndrome. Early detection through ultrasound imaging is crucial for preventing progression to more severe liver diseases.

## Technical Architecture

### 🔬 Feature Extraction Methods

#### 1. Texture Analysis with Gray-Level Co-occurrence Matrix (GLCM)
- **Multi-distance analysis**: 1, 2, 4, and 8-pixel distances
- **Radial approach**: 16 angular directions averaged for rotation invariance
- **Features extracted**:
  - Entropy (texture randomness)
  - Homogeneity (local texture uniformity)

#### 2. Statistical Feature Matrix (SFM) 
- **Coarseness**: Measures texture granularity
- **Contrast**: Local intensity variations
- **Periodicity**: Repetitive texture patterns
- **Roughness**: Surface irregularity metrics

### 🤖 Machine Learning Models

#### Support Vector Machine (SVM)
- **Configurable kernels**: Linear, RBF, Polynomial, Sigmoid
- **Hyperparameter optimization**: C, gamma, degree, coef0
- **Class balancing**: Weighted classes for imbalanced datasets
- **Feature set**: 12-dimensional texture descriptors

#### MobileNet Deep Learning
- **Transfer learning**: Pre-trained ImageNet weights
- **Fine-tuning capability**: Configurable layer unfreezing
- **Architecture**: MobileNet backbone + custom classification head
- **Input**: 224×224 RGB images
- **Optimization**: Adam/SGD with configurable learning rates

### 📊 Validation Strategy

#### Leave-One-Patient-Out Cross-Validation (LOPOCV)
- **Patient-level splitting**: Ensures no data leakage between train/test
- **Comprehensive evaluation**: 55 patients, 10 ROIs per patient
- **Performance metrics**:
  - Accuracy, Sensitivity, Specificity
  - Precision, F1-Score
  - Confusion matrices
  - Learning curves (for deep learning)

### 🖥️ User Interface Features

#### Interactive GUI Application
- **Image visualization**: Grayscale ultrasound display
- **ROI selection**: Manual region-of-interest annotation
- **Real-time analysis**: Texture descriptor calculation
- **Model comparison**: Side-by-side SVM vs MobileNet results
- **Parameter tuning**: GUI-based hyperparameter adjustment

#### Analysis Tools
- **Histogram analysis**: Intensity distribution visualization
- **GLCM visualization**: Co-occurrence matrix display
- **Feature inspection**: Real-time texture descriptor values
- **Model inference**: Saved model deployment for new images

## Dataset Structure

### Input Data
- **Format**: MATLAB `.mat` file containing ultrasound B-mode images
- **Organization**: 55 patients × 10 images per patient = 550 total images
- **Resolution**: Variable ultrasound image dimensions
- **Preprocessing**: 28×28 pixel ROI extraction for SVM, 224×224 for MobileNet

### Generated Features
- **CSV export**: Comprehensive feature dataset (`data.csv`)
- **Feature vector**: 18 attributes including coordinates, texture descriptors
- **Labels**: Binary classification (Healthy/Steatotic)

## Installation & Setup

### Prerequisites
```bash
# Verify Python installation
python --version  # Requires Python 3.7+
```

### Dependencies Installation
```bash
# Core scientific computing
pip install numpy scipy matplotlib pandas

# Computer vision and image processing
pip install opencv-python-headless pillow scikit-image

# Machine learning frameworks
pip install scikit-learn

# Deep learning (TensorFlow/Keras)
pip install tensorflow

# Feature extraction
pip install pyfeats

# GUI framework
pip install tkinter  # Usually included with Python

# Data visualization
pip install seaborn

# Model persistence
pip install joblib
```

### Alternative Installation
```bash
# Install all dependencies at once
pip install numpy opencv-python-headless matplotlib pillow scipy scikit-image pyfeats pandas scikit-learn tensorflow seaborn joblib
```

## Usage Guide

### 1. Application Launch
```bash
python main.py
```

### 2. Core Workflows

#### Data Preparation
1. **Load Dataset**: Import MATLAB `.mat` file containing ultrasound images
2. **ROI Generation**: Extract regions of interest from liver areas
3. **Feature Extraction**: Calculate GLCM and SFM texture descriptors
4. **Export Data**: Generate `data.csv` for machine learning training

#### Model Training & Evaluation
1. **SVM Classification**: Traditional machine learning approach
2. **MobileNet Training**: Deep learning with transfer learning
3. **Cross-Validation**: LOPOCV evaluation with comprehensive metrics
4. **Model Comparison**: Side-by-side performance analysis

#### Inference & Deployment
1. **Load Trained Models**: Import saved SVM (`svm_model.sav`) or MobileNet (`mobilenet_model.h5`)
2. **Image Classification**: Classify new ultrasound images
3. **Result Interpretation**: Binary prediction with confidence metrics

### 3. Advanced Features

#### Parameter Optimization
- **SVM Tuning**: Kernel selection, regularization, gamma adjustment
- **MobileNet Configuration**: Learning rates, dropout, fine-tuning layers
- **Training Control**: Epochs, batch size, early stopping

#### Visualization & Analysis
- **Performance Metrics**: Confusion matrices, ROC curves
- **Learning Curves**: Training/validation loss progression
- **Feature Analysis**: Texture descriptor distributions

## Project Structure

```
trabalhoPAI/
├── main.py                          # Main application entry point
├── liver.ipynb                      # Jupyter notebook for data exploration
├── data.csv                         # Extracted feature dataset
├── dataset_liver_bmodes_*.mat       # Raw ultrasound image data
├── mobilenet_model.h5               # Saved deep learning model
├── svm_model.sav                    # Saved SVM model
├── ROIS/                            # Extracted regions of interest
├── matrizes_confusao/               # Confusion matrix visualizations
├── fotos/                           # Sample images
└── fotos_teste/                     # Test images for inference
```

## Performance Metrics

The system evaluates both models using standard medical imaging metrics:

- **Accuracy**: Overall classification correctness
- **Sensitivity (Recall)**: True positive rate for steatosis detection
- **Specificity**: True negative rate for healthy tissue identification
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall

## Technical Specifications

### System Requirements
- **Operating System**: Windows, macOS, Linux
- **Python Version**: 3.7 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for deep learning)
- **Storage**: 2GB free space for models and data

### Performance Characteristics
- **SVM Training**: Fast, seconds to minutes
- **MobileNet Training**: Moderate, minutes to hours depending on parameters
- **Inference Speed**: Real-time classification (<1 second per image)
- **Model Size**: SVM (~KB), MobileNet (~17MB)

## Research Applications

This system is designed for:
- **Clinical Decision Support**: Assisting radiologists in ultrasound interpretation
- **Screening Programs**: Automated hepatic steatosis detection
- **Research Studies**: Quantitative analysis of liver tissue characteristics
- **Educational Tools**: Medical imaging and machine learning education

## Future Enhancements

- **Multi-class Classification**: Severity grading of steatosis
- **Real-time Integration**: DICOM compatibility for clinical systems
- **Ensemble Methods**: Combining SVM and deep learning predictions
- **Explainable AI**: Visualization of decision-making features

---

**Note**: This system is intended for research and educational purposes. Clinical deployment requires appropriate validation and regulatory approval.
