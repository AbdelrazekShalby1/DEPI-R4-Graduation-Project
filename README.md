# AI Medical System

## Overview
An intelligent healthcare platform that integrates Machine Learning, Computer Vision, and Conversational AI technologies to support medical diagnosis and patient interaction.

The system combines:
- 5 Machine Learning models for disease prediction
- 3 Computer Vision models for medical image analysis
- An AI-powered chatbot for healthcare assistance

---

## Features

### Machine Learning Models
- Kidney Disease Prediction
- Liver Disease Prediction
- Alzheimer's Detection
- Epileptic Seizure Prediction
- Lung Disease Prediction

### Computer Vision Models
- Brain Tumor Classification
- X-Ray Analysis
- Medical Image Processing

### Chatbot
- Symptom-based interaction
- Healthcare guidance
- Real-time patient support

---

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- Scikit-Learn
- OpenCV
- HTML
- CSS
- JavaScript

---

## Project Structure

```text
DEPI-R4-Graduation-Project
│
├── DEPI/
├── ml-models/
├── Brain_tumor classification/
├── x-ray/
├── chatbot-main/
├── tests/
└── README.md
```

The integrated Flask application, HTML templates, and application dependencies
are in `DEPI/`. Model folders and `chatbot-main/` live directly in the repository
root. The application resolves these paths relative to its source file, so no
additional `Models/` parent directory is required.

---

## Installation

```powershell
git clone https://github.com/AbdelrazekShalby1/DEPI-R4-Graduation-Project.git
cd DEPI-R4-Graduation-Project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r DEPI/requirements.txt
python DEPI/app.py
```

Keep the pinned `xgboost==2.1.4` and `scikit-learn==1.6.1` versions from the
requirements file: these serialized models are version sensitive. Loading the
kidney model with XGBoost 3.3 can fail with `input stream corrupted`.

Open http://127.0.0.1:8000 in your browser. The application can also be started
with `python app.py` after changing into `DEPI/`.

For the Gemini chatbot, create `chatbot-main/.env` and set
`GEMINI_KEY_1=your_api_key_here`, then restart the server. Do not commit API keys.
Without provider credentials, live Gemini responses are not available.

### Model availability

The five ML model/scaler pairs, brain MRI classifier, and chest X-ray classifier
are included. The separate brain **detection** endpoint requires an additional
YOLO artifact in `brain_tomor detection/` at the repository root; this artifact
is not included in the current repository. Brain classification and brain
detection are different endpoints.

### Regression tests

From the repository root, with the application dependencies installed:

```powershell
python -m unittest discover -s tests -v
```

These tests verify artifact paths and exercise the real kidney prediction
endpoint using the application's default input values. They do not validate
clinical accuracy or call Gemini.

---

## Objectives

- Improve diagnostic accuracy
- Support medical decision-making
- Enhance healthcare accessibility
- Provide intelligent patient interaction

---

## Contributors

DEPI R4 Graduation Project Team
