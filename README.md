# PneumoScan

AI-powered pneumonia detection from chest X-rays. 7 deep learning models. Weighted ensemble inference. Built end-to-end in Python.

**Live Demo:** https://yusra-shah--pneumo-scan-serve.modal.run

---

## Overview

A doctor logs in, registers a patient, and uploads a chest X-ray. PneumoScan runs the image through a 7-model ensemble and returns a diagnosis. Every model votes independently. Results include confidence score, severity, subtype, and a Grad-CAM heatmap showing which lung regions influenced the prediction. A bilingual PDF report (English + Urdu) is generated and saved per patient in MongoDB.

All results come from real PyTorch inference on trained weights.

---

## Screenshots

### Login
![Login](screenshots/1.png)

### Dashboard
![Dashboard](screenshots/Dashboard.png)

### Scan Upload
![Scan](screenshots/Scan.png)

### Real-time Analysis
![Analyzing](screenshots/Analyzing.png)

### Grad-CAM Heatmap
![Grad-CAM](screenshots/Grad-Cam.png)

### Normal Result
![Normal](screenshots/Normal.png)

### Pneumonia Detected — Severe, Bacterial, 95.55%
![Pneumonia](screenshots/Penumonia.png)

### Patient Records
![Patient Records](screenshots/Patient_records.png)

### Patient Profile with Scan History
![Patient](screenshots/Patient.png)

### AI Models Panel
![Models](screenshots/models.png)

### Generated PDF Report
![Report](screenshots/report.png)

---

## Models

Trained on 15,000 chest X-rays. Weighted soft-vote ensemble. Weights stored in float16 for deployment.

| Model | Role | Val Acc | Test Acc | Weight |
|---|---|---|---|---|
| DenseNet-121 | Anchor — CheXNet architecture | 98.87% | 98.67% | 0.25 |
| InceptionV3 | Multi-scale feature detection | 98.80% | 98.73% | 0.10 |
| AttentionCNN | Grad-CAM explainability | 98.67% | 98.13% | 0.00 |
| EfficientNet-B4 | Efficiency and accuracy balance | 98.40% | 98.20% | 0.20 |
| ViT-B/16 | Global pattern detection | 98.13% | 97.73% | 0.17 |
| ResNet-50 | General-purpose feature extractor | 98.47% | 97.67% | 0.18 |
| MobileNetV3 | Speed model, runs first | 97.13% | 97.80% | 0.10 |

Ensemble accuracy: 98.4%

---

## Features

- Doctor authentication with bcrypt hashed passwords and MongoDB session management
- Patient registration, search, and full medical history tracking
- Real-time ensemble inference with per-model vote breakdown
- Grad-CAM heatmap overlay on the original X-ray
- Severity grading: None / Mild / Moderate / Severe
- Subtype classification: Bacterial or Viral
- Bilingual PDF report generation in English and Urdu via ReportLab
- MongoDB Atlas with transactions, nested documents, and audit logging

---

## Stack

| Layer | Technology |
|---|---|
| GUI | PySide6 6.6.1 |
| Deep learning | PyTorch 2.1.0, timm 0.9.7 |
| Explainability | grad-cam 1.4.8, OpenCV |
| Database | MongoDB Atlas |
| Reports | ReportLab 4.3.0 |
| Deployment | Modal serverless |
| Auth | bcrypt |

---

## Dataset

15,000 balanced chest X-ray images. 7,500 Normal, 7,500 Pneumonia.

Sources: Kaggle Kermany dataset, NIH ChestX-ray14, Albumentations augmentation pipeline.

Split: 80% train / 10% val / 10% test, stratified.

---

## Run Locally

```bash
git clone https://github.com/Yusra-Shah/Pneumo-Scan.git
cd Pneumo-Scan
python -m venv nemo_env
nemo_env\Scripts\activate
pip install -r requirements.txt
python main.py
```

Requires Python 3.12 and a MongoDB Atlas URI in `config.yaml`.

---

## Project Structure

```
Pneumo/
  core/
    inference/engine.py       Ensemble inference engine
    report_generator.py       Bilingual PDF generation
    gradcam.py                Grad-CAM heatmap logic
  gui/
    main_window.py
    dashboard.py
    scan_panel.py
    patients_panel.py
    models_panel.py
    styles.py
  database/                   MongoDB Atlas integration
  weights/lung/               Trained .pth files (float16)
  outputs/
    reports/                  Generated PDFs
    heatmaps/                 Saved Grad-CAM overlays
  training/                   Training scripts and augmentation pipeline
  modal_app.py                Modal deployment
  main.py                     Entry point
```

---

*Built by Yusra Shah and Sinya Kumari — Sukkur IBA University, 2026*
