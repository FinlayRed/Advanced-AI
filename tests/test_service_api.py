from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.service.api import create_app


def _image_bytes(color: tuple[int, int, int] = (180, 120, 90)) -> BytesIO:
    img = Image.new("RGB", (32, 32), color=color)
    payload = BytesIO()
    img.save(payload, format="JPEG")
    payload.seek(0)
    return payload


def test_health_endpoint_bootstraps_models(tmp_path):
    app = create_app(
        models_dir=str(tmp_path / "models"),
        registry_path=str(tmp_path / "registry.json"),
        interaction_log_path=str(tmp_path / "interactions.jsonl"),
    )
    client = app.test_client()

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["active_models"]["recommender"]["name"] == "bootstrap-recommender"
    assert payload["active_models"]["quality"]["name"] == "best-quality-model"
    assert payload["active_models"]["quality"]["runtime"] == "quality_checkpoint"


def test_recommendation_and_outcome_flow(tmp_path):
    app = create_app(
        models_dir=str(tmp_path / "models"),
        registry_path=str(tmp_path / "registry.json"),
        interaction_log_path=str(tmp_path / "interactions.jsonl"),
    )
    client = app.test_client()

    response = client.post("/recommendations/next-order", json={"customer_id": "C0001", "k": 3})
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["recommendations"]) == 3
    assert "reason_text" in payload["recommendations"][0]

    outcome = client.post(
        "/recommendations/outcome",
        json={
            "event_id": payload["event_id"],
            "accepted": [payload["recommendations"][0]["product_id"]],
            "added_outside_recommendation": ["P0001"],
        },
    )
    assert outcome.status_code == 200

    overview = client.get("/admin/overview")
    overview_payload = overview.get_json()
    assert overview_payload["metrics"]["recommendation_events"] == 1
    assert overview_payload["metrics"]["override_count"] == 1


def test_quality_inspection_logs_breakdown(tmp_path):
    app = create_app(
        models_dir=str(tmp_path / "models"),
        registry_path=str(tmp_path / "registry.json"),
        interaction_log_path=str(tmp_path / "interactions.jsonl"),
    )
    client = app.test_client()

    response = client.post(
        "/quality/inspect",
        data={
            "producer_id": "producer-1",
            "product_type": "Tomato",
            "quantity": "8",
            "image": (_image_bytes(), "tomato.jpg"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["inspection"]["condition"] in {"healthy", "rotten"}
    assert payload["inspection"]["decision_mode"] == "healthy_rotten"
    assert payload["inspection"]["grade"] in {"A", "B", "C"}
    assert "reason_text" in payload["inspection"]
    assert payload["inspection"]["action"]

    interactions = client.get("/admin/interactions?event_type=quality_inspection&producer_id=producer-1")
    interactions_payload = interactions.get_json()
    assert len(interactions_payload["interactions"]) == 1
    assert interactions_payload["interactions"][0]["condition"] in {"healthy", "rotten"}


def test_model_upload_endpoint(tmp_path):
    app = create_app(
        models_dir=str(tmp_path / "models"),
        registry_path=str(tmp_path / "registry.json"),
        interaction_log_path=str(tmp_path / "interactions.jsonl"),
    )
    client = app.test_client()

    response = client.post(
        "/models/upload",
        data={
            "model_type": "quality",
            "runtime": "quality_checkpoint",
            "name": "uploaded-quality",
            "version": "2.0",
            "file": (BytesIO(b"pretend checkpoint"), "quality.pt"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    payload = response.get_json()
    assert payload["model"]["name"] == "uploaded-quality"


def test_legacy_quality_heuristic_runtime_is_rejected(tmp_path):
    app = create_app(
        models_dir=str(tmp_path / "models"),
        registry_path=str(tmp_path / "registry.json"),
        interaction_log_path=str(tmp_path / "interactions.jsonl"),
    )
    client = app.test_client()

    response = client.post(
        "/models/upload",
        data={
            "model_type": "quality",
            "runtime": "quality_heuristic",
            "name": "legacy-quality",
            "version": "1.0",
            "file": (BytesIO(b"pretend checkpoint"), "quality.pt"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "quality_checkpoint" in response.get_json()["error"]
