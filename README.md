# Bristol Regional Food Network AI Service

This repository contains the AI service for the Bristol Regional Food Network project. It combines three pieces of work into one deployable Python service:

- a product recommendation subsystem for quick reorders and next-order suggestions
- a fruit and vegetable quality inspection model backed by a trained PyTorch checkpoint
- a Flask API that exposes both subsystems to the wider DESD platform

This is designed to be run as a local service. Client applications call the API over HTTP instead of importing model code directly.

## Features

### Recommendation Service

- Generates quick-reorder recommendations from customer purchase history.
- Generates next-order recommendations using collaborative filtering and a frequency-recency baseline.
- Applies producer-diversity re-ranking so recommendations do not over-concentrate on one producer.
- Returns explainable recommendation payloads with scores, confidence values, reason text, and reason codes.
- Records recommendation outcomes so future retraining can use accepted items and user overrides.

### Quality Inspection

- Loads the included `best_quality_model.pt` checkpoint by default.
- Classifies uploaded produce images as healthy or rotten.
- Returns confidence, condition, quality breakdown, legacy grade, recommended action, and explanation text.
- Tracks inspections by producer and product type for admin monitoring.

### API And Monitoring

- Provides health, model registry, recommendation, inspection, and admin endpoints.
- Supports uploading and activating new model artifacts at runtime.
- Writes local interaction logs for monitoring and future retraining.
- Keeps generated runtime artifacts out of git.

## Repository Structure

```text
.
├── best_quality_model.pt          # Default trained quality checkpoint
├── data/                          # Runtime data directory, kept with .gitkeep
├── logs/                          # Runtime log directory, kept with .gitkeep
├── requirements.txt               # Python dependencies
├── src/
│   ├── baseline.py                # Frequency-recency recommender
│   ├── charts.py                  # Reporting/chart helpers
│   ├── data_generator.py          # Synthetic purchase data generation
│   ├── demo.py                    # End-to-end recommender demo
│   ├── evaluation.py              # Recommendation evaluation metrics
│   ├── fairness.py                # Producer-diversity re-ranking
│   ├── feedback.py                # Recommendation feedback logging
│   ├── mf_model.py                # Matrix-factorisation recommender
│   ├── recommender.py             # Service-facing recommender wrapper
│   ├── quality/                   # Quality model training and inference
│   └── service/                   # Flask API, model registry, runtime storage
└── tests/                         # Pytest test suite
```

## Requirements

- Python 3.10-3.12 recommended
- pip
- The packages listed in `requirements.txt`

The main dependencies are Flask, PyTorch, torchvision, Pillow, pandas, NumPy, scikit-learn, joblib, matplotlib, and pytest. PyTorch wheels may not be available for every Python and platform combination, such as Python 3.13 on Windows ARM.

## Setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Run The Tests

Run the full test suite:

```powershell
pytest -v
```

Run only the service API tests:

```powershell
pytest tests/test_service_api.py -v
```

## Run The API

Start the Flask service:

```powershell
python -m src.service.api
```

The API starts on:

```text
http://localhost:5050
```

Check the service is running:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5050/health"
```

## API Endpoints

### Health

```http
GET /health
```

Returns service status and the active recommender and quality models.

### Model Registry

```http
GET /models
POST /models/upload
POST /models/activate
```

Supported runtime types:

- `recommendation_service` for recommender `.joblib` artifacts
- `quality_checkpoint` for quality model `.pt` checkpoints

Example quality model upload:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5050/models/upload" `
  -Form @{
    model_type = "quality"
    runtime = "quality_checkpoint"
    name = "quality-model"
    version = "1.0"
    activate = "true"
    metadata = '{"class_names":["fresh","rotten"]}'
    file = Get-Item "C:\path\to\model.pt"
  }
```

### Recommendations

```http
POST /recommendations/quick-reorder
POST /recommendations/next-order
POST /recommendations/outcome
```

Example next-order request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5050/recommendations/next-order" `
  -ContentType "application/json" `
  -Body '{"customer_id":"C0001","k":5,"fairness":true}'
```

Example outcome logging request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5050/recommendations/outcome" `
  -ContentType "application/json" `
  -Body '{"event_id":"rec-123","accepted":["P0001"],"added_outside_recommendation":["P0009"]}'
```

### Quality Inspection

```http
POST /quality/inspect
```

Example request with an uploaded image:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5050/quality/inspect" `
  -Form @{
    producer_id = "producer-1"
    product_type = "Tomato"
    quantity = "8"
    image = Get-Item "C:\path\to\tomato.jpg"
  }
```

Typical response fields include:

- predicted class label
- healthy or rotten condition
- confidence
- color, size, and ripeness scores
- A/B/C grade
- suggested action
- explanation text

### Admin Monitoring

```http
GET /admin/overview
GET /admin/interactions
```

Example requests:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5050/admin/overview"
Invoke-RestMethod -Uri "http://127.0.0.1:5050/admin/interactions?event_type=quality_inspection&limit=10"
```

## Recommender Demo

Run the end-to-end recommender demo:

```powershell
python -m src.demo
```

The demo generates synthetic order data, trains recommendation models, evaluates them, demonstrates fairness re-ranking, logs feedback, and writes a deployable recommender artifact.

Generated demo artifacts are ignored by git.

## Quality Model Training

Build train and validation CSV files from image folders:

```powershell
python -m src.quality.build_dataset_csv `
  --image_root "C:\path\to\Fruit And Vegetable Diseases Dataset" `
  --output_dir data
```

Train a binary quality classifier:

```powershell
python -m src.quality.train `
  --train_csv data\quality_train.csv `
  --val_csv data\quality_val.csv `
  --image_root "C:\path\to\Fruit And Vegetable Diseases Dataset" `
  --epochs 20 `
  --batch_size 32 `
  --reg_weight 0 `
  --imagenet_normalize `
  --save_dir models\quality
```

## Runtime Files

The service creates local runtime files when it starts and handles requests:

- `data/model_registry.json`
- `data/service_models/`
- `logs/service_interactions.jsonl`
- `logs/feedback.jsonl`
- generated CSV, joblib, and chart artifacts from demos or training runs

These files are intentionally ignored by git. The tracked `.gitkeep` files preserve the required `data/` and `logs/` directories.
