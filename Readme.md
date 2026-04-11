# NEMO Scan
### Neural Engine for Medical Observation

AI-powered chest X-ray diagnostic system. Detects pneumonia using an ensemble of 7 deep learning models with Grad-CAM explainability, severity scoring, bilingual PDF reports, and full patient history tracking.

---

## Setup (Windows)

### Prerequisites
- Python 3.12.x installed (python.org)
- VS Code installed with Claude Code extension
- MongoDB Atlas free account (mongodb.com/atlas)
- Kaggle account (kaggle.com) for dataset download

### Step 1 — Clone and enter the project
```
git clone <your-repo-url>
cd NEMO_Scan
```

### Step 2 — Create virtual environment
```
python -m venv nemo_env
nemo_env\Scripts\activate
```

### Step 3 — Install dependencies
```
pip install -r requirements.txt
```

### Step 4 — Verify installation
```
python verify_env.py
```

### Step 5 — Create folder structure
```
python create_folders.py
```

### Step 6 — Configure database
Open `config.yaml` and replace the MongoDB URI with your Atlas connection string.

### Step 7 — Run the application
```
python main.py
```

---

## Project Structure

```
NEMO_Scan/
    core/
        models/lung/         7 model definition files
        models/cornea/       Future — placeholder
        models/bone/         Future — placeholder
        inference/           Inference engine + Grad-CAM + report generator
    training/                Training scripts for all 7 models
    domains/
        lung/data/           Train / val / test image folders
        lung/checkpoints/    Saved model weights during training
        cornea/              Future domain scaffold
        bone/                Future domain scaffold
    gui/                     PySide6 interface (all screens)
    utils/                   Preprocessing, metrics, logging
    weights/lung/            Final .pth weight files
    outputs/
        reports/             Generated PDF reports
        heatmaps/            Grad-CAM overlay images
    config.yaml              All global settings
    requirements.txt         Python dependencies
    verify_env.py            Import checker
    create_folders.py        One-time folder creator
    main.py                  Application entry point
```

---

## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Deep Learning | PyTorch | 2.6.0 |
| Model Hub | timm | 1.0.15 |
| GUI | PySide6 | 6.8.3 |
| Database | MongoDB Atlas | via pymongo 4.10.1 |
| Explainability | grad-cam | 1.5.4 |
| Image Processing | OpenCV | 4.11.0 |
| Augmentation | Albumentations | 2.0.8 |
| Reports | ReportLab | 4.3.0 |
| Auth | bcrypt | 4.3.0 |

---

## The 7 Models

| Model | Role | Ensemble Weight |
|-------|------|----------------|
| MobileNetV3 | Speed — instant result | 0.10 |
| ResNet-50 | Core backbone | 0.18 |
| DenseNet-121 | Anchor (CheXNet) | 0.25 |
| EfficientNet-B4 | Efficiency balance | 0.20 |
| ViT-B/16 | Global attention | 0.17 |
| InceptionV3 | Multi-scale features | 0.10 |
| Attention CNN | Grad-CAM heatmap only | — |

---

## Database (MongoDB Atlas)

This project doubles as a demonstration for Advanced Database Management Systems. The MongoDB schema covers:
- Doctor authentication with bcrypt-hashed passwords
- Nested scan documents with full model vote breakdowns
- Patient history with chronological scan tracking
- Multi-document transactions for atomic scan record writes
- Aggregation queries for dashboard statistics
- Audit logging for all write operations

Collections: `doctors`, `patients`, `scans`, `audit_log`

---

*Academic use only. NEMO Scan v1.0 Prototype — Lung Module.*