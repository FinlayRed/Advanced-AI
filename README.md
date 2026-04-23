# Bristol Regional Food Network AI Service

This repository contains the AI service and integration layer I implemented for
the Bristol Regional Food Network project. My work focused on turning the model
code into something the wider DESD platform could actually call, monitor, and
swap at runtime.

The main things I implemented were:

* a Flask API around the recommendation and quality subsystems
* a model registry with upload and activation workflows
* explainable outputs for recommendation and quality responses
* interaction logging for monitoring and future retraining
* admin-facing summary endpoints for visibility into AI activity
* tests for the service layer

## What I implemented

The service layer lives in `src/service/`.

Main files:

* `src/service/api.py` - HTTP API endpoints
* `src/service/storage.py` - model registry and interaction log storage
* `src/service/quality_runtime.py` - quality model runtime wrapper
* `src/recommender.py` - recommendation explanations and service-facing wrapper
* `src/quality/infer.py` - quality inference output normalisation
* `tests/test_service_api.py` - service API tests

The goal was to create a stable service boundary between this repo and the DESD
web app. The DESD project calls this service over HTTP rather than importing
machine learning code directly.

## What the service does

The API exposes four main areas.

### 1. Health and runtime status

This is used to confirm the service is alive and to show which models are
currently active.

* `GET /health`

### 2. Model management

This is the deployment workflow I added so a trained model can be exported
offline, uploaded into the service, and activated without changing DESD.

* `GET /models`
* `POST /models/upload`
* `POST /models/activate`

Supported runtime types:

* `recommendation_service` for recommender `.joblib` artefacts
* `quality_checkpoint` for trained quality `.pt` checkpoints
* `quality_heuristic` for the bootstrap fallback runtime

### 3. Recommendation endpoints

These are used by the DESD recommendation demo page and by direct API testing.

* `POST /recommendations/quick-reorder`
* `POST /recommendations/next-order`
* `POST /recommendations/outcome`

These responses include:

* ranked `product_id`s
* score and confidence
* producer/category context
* explanation text
* reason codes
* event IDs for outcome logging

### 4. Quality and admin endpoints

These are used by the DESD quality inspection UI and the manager monitoring
page.

* `POST /quality/inspect`
* `GET /admin/overview`
* `GET /admin/interactions`

Quality responses include:

* predicted class label
* confidence
* color / size / ripeness breakdown
* final `A/B/C` grade
* action suggestion
* explanation text

## API structure

### `GET /health`

Typical response:

```json
{
  "status": "ok",
  "active_models": {
    "recommender": {
      "model_id": "...",
      "model_type": "recommender",
      "name": "bootstrap-recommender",
      "version": "1.0",
      "runtime": "recommendation_service",
      "is_active": true
    },
    "quality": {
      "model_id": "...",
      "model_type": "quality",
      "name": "yas-quality-model",
      "version": "1.0",
      "runtime": "quality_checkpoint",
      "is_active": true
    }
  }
}
```

### `GET /models`

Returns all registered models, including active and inactive versions.

### `POST /models/upload`

Form fields:

* `model_type`
* `runtime`
* `name`
* `version`
* `activate`
* `metadata`
* `file`

### `POST /models/activate`

JSON body:

```json
{
  "model_id": "abc123"
}
```

### `POST /recommendations/next-order`

JSON body:

```json
{
  "customer_id": "C0001",
  "k": 5,
  "fairness": true
}
```

Typical response:

```json
{
  "event_id": "rec-123",
  "model": {
    "name": "bootstrap-recommender",
    "version": "1.0"
  },
  "recommendations": [
    {
      "product_id": "P0001",
      "rank": 1,
      "category": "Tomato",
      "producer_id": "PR001",
      "score": 0.92,
      "confidence": 0.71,
      "reason_text": "Recommended because similar customers repeatedly bought this item.",
      "reason_codes": ["next_order", "nmf", "direct_rank"]
    }
  ]
}
```

### `POST /recommendations/quick-reorder`

JSON body:

```json
{
  "customer_id": "C0001",
  "k": 3
}
```

### `POST /recommendations/outcome`

JSON body:

```json
{
  "event_id": "rec-123",
  "accepted": ["P0001", "P0002"],
  "added_outside_recommendation": ["P0009"]
}
```

This is what feeds monitoring and future retraining signals.

### `POST /quality/inspect`

Form fields:

* `producer_id`
* `product_type`
* `quantity`
* `image` or `image_path`

Typical response:

```json
{
  "model": {
    "name": "yas-quality-model",
    "version": "1.0"
  },
  "inspection": {
    "predicted_class_label": "fresh",
    "confidence": 0.84,
    "quality": {
      "color_score": 72.0,
      "size_score": 86.0,
      "ripeness_score": 83.0
    },
    "grade": "B",
    "action": "sell_normal",
    "reason_text": "Grade B because color below 75. Suggested action: sell normal."
  }
}
```

### `GET /admin/overview`

Returns aggregate monitoring data such as:

* active models
* recommendation event counts
* override rate
* inspection counts
* grade distribution
* recent examples

### `GET /admin/interactions`

Optional query params:

* `event_type`
* `producer_id`
* `limit`

This returns recent raw interaction log entries.

## How to use the API

Start the service:

```bash
python -m src.service.api
```

### Quick smoke test

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5050/health"
Invoke-RestMethod -Uri "http://127.0.0.1:5050/models"
```

### Recommendation request

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5050/recommendations/next-order" `
  -ContentType "application/json" `
  -Body '{"customer_id":"C0001","k":5,"fairness":true}'
```

### Recommendation outcome logging

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5050/recommendations/outcome" `
  -ContentType "application/json" `
  -Body '{"event_id":"rec-123","accepted":["P0001"],"added_outside_recommendation":["P0009"]}'
```

### Quality inspection request

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5050/quality/inspect" `
  -Form @{
    producer_id = "demo"
    product_type = "Tomato"
    quantity = "1"
    image = Get-Item "C:\path\to\tomato.jpg"
  }
```

### Upload and activate a trained quality model

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:5050/models/upload" `
  -Form @{
    model_type = "quality"
    runtime = "quality_checkpoint"
    name = "yas-quality-model"
    version = "1.0"
    activate = "true"
    metadata = '{"class_names":["fresh","rotten"]}'
    file = Get-Item "E:\path\to\best_quality_model.pt"
  }
```

### Query monitoring data

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5050/admin/overview"
Invoke-RestMethod -Uri "http://127.0.0.1:5050/admin/interactions?event_type=recommendation_shown&limit=10"
```

## Runtime files and git hygiene

At runtime the service generates:

* `data/model_registry.json`
* `data/service_models/`
* `logs/service_interactions.jsonl`

These are local runtime artefacts and are intentionally ignored by git so the
repository stays clean for GitHub and assessment hand-in.

## Verification

The service changes were verified with tests in this repo, including:

* recommendation endpoint tests
* quality inspection endpoint tests
* model upload tests
* service bootstrap tests

Run the service tests with:

```bash
pytest tests/test_service_api.py -v
```
