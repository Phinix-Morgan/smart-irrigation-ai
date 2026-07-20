# 🌱 IrriSmart AI

> AI-powered Smart Irrigation Prediction System using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-orange)
![License](https://img.shields.io/badge/License-MIT-success)

---

## Overview

IrriSmart AI is an end-to-end Machine Learning application that predicts irrigation requirements using environmental, soil, and field parameters.

Instead of relying on a single ML model, the system combines:

- LightGBM Classification
- Random Forest Regression
- K-Means Clustering

to provide intelligent irrigation recommendations, water usage estimation, and field clustering through an interactive Flask dashboard.

---

## Features

- Smart Irrigation Need Prediction
- Water Requirement Estimation
- Field-Level Prediction
- Interactive Dashboard
- Model Analysis
- Feature Importance Visualization
- PCA Cluster Visualization
- Confusion Matrix
- Cross Validation Metrics
- Regression Analysis
- Dataset Analytics
- Dark Mode
- Responsive UI

---

## Machine Learning Pipeline

Dataset

↓

Data Cleaning

↓

Feature Engineering

↓

Label Encoding

↓

Train/Test Split

↓

LightGBM Classifier

↓

Random Forest Regressor

↓

K-Means Clustering

↓

Evaluation

↓

Model Serialization (Joblib)

↓

Flask Inference

---

## Models Used

### LightGBM
- Irrigation Classification
- Low / Medium / High

### Random Forest
- Water Requirement Prediction (mm)

### K-Means
- Field Segmentation

---

## Tech Stack

### Backend

- Flask
- Python

### Machine Learning

- LightGBM
- Random Forest
- Scikit-Learn
- PCA
- K-Means

### Data

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Chart.js

### Frontend

- HTML
- CSS
- JavaScript
- Jinja2

---

## Folder Structure

```text
IrriSmart-AI
│
├── ml/
│   ├── model.py
│   └── saved_model/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│
├── app.py
├── train_and_save.py
├── requirements.txt
└── README.md
```

---

## Dashboard

The application includes a complete analytics dashboard showing:

- Classification Accuracy
- Cross Validation
- Regression Metrics
- Cluster Distribution
- Feature Importance
- Confusion Matrix
- PCA Analysis
- Dataset Statistics

---

## Prediction Modes

### Quick Prediction

Fast irrigation prediction.

### Field Prediction

Complete farm analysis including:

- Water Requirement
- Irrigation Need
- Risk Level
- Weekly Schedule
- Cluster Information
- Cost Estimation
- AI Recommendation

---

## Installation

```bash
git clone https://github.com/yourusername/IrriSmart-AI.git

cd IrriSmart-AI

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python app.py
```

---

## Screenshots

```
Home Page

Dashboard

Quick Prediction

Field Prediction

Analysis

About
```

(Add screenshots here)

---

## Future Improvements

- Docker Support
- Cloud Deployment
- Live Weather API
- Satellite Imagery
- IoT Sensor Integration
- Mobile Application
- Deep Learning Models

---

## What I Learned

- Production ML deployment
- Model serialization
- Flask architecture
- LightGBM
- Ensemble learning
- ML visualization
- Feature engineering
- Dashboard design
- End-to-end ML workflow

---

## License

MIT License