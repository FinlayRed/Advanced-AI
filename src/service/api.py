from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import UnidentifiedImageError

from src.quality.inventory import ProducerInventory
from src.quality.grading import QualityThresholds
from src.service.quality_runtime import QualityRuntime
from src.service.storage import InteractionStore, ModelRegistry, utc_now_iso


def create_app(
    *,
    models_dir: str = "data/service_models",
    registry_path: str = "data/model_registry.json",
    interaction_log_path: str = "logs/service_interactions.jsonl",
) -> Flask:
    app = Flask(__name__)
    interaction_store = InteractionStore(interaction_log_path)
    registry = ModelRegistry(
        models_dir=models_dir,
        registry_path=registry_path,
        interaction_store=interaction_store,
    )
    app.config["interaction_store"] = interaction_store
    app.config["model_registry"] = registry

    @app.get("/health")
    def health() -> tuple:
        recommender = registry.get_active_model("recommender")
        quality = registry.get_active_model("quality")
        return jsonify(
            {
                "status": "ok",
                "active_models": {
                    "recommender": _model_summary(recommender),
                    "quality": _model_summary(quality),
                },
            }
        )

    @app.get("/models")
    def list_models() -> tuple:
        return jsonify({"models": [_model_summary(model, full=True) for model in registry.list_models()]})

    @app.post("/models/upload")
    def upload_model() -> tuple:
        payload = request.form if request.form else request.json or {}
        model_type = payload.get("model_type")
        name = payload.get("name") or f"{model_type}-model"
        version = payload.get("version") or "1.0"
        runtime = payload.get("runtime")
        activate = str(payload.get("activate", "false")).lower() in {"1", "true", "yes", "on"}
        metadata = _parse_metadata(payload.get("metadata"))
        if not model_type or not runtime:
            return jsonify({"error": "model_type and runtime are required."}), 400

        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "file is required."}), 400

        record = registry.upload_model(
            file_obj=file.stream,
            filename=file.filename or f"{model_type}.bin",
            model_type=model_type,
            name=name,
            version=version,
            runtime=runtime,
            metadata=metadata,
            activate=activate,
        )
        interaction_store.append(
            {
                "event_id": f"model-upload-{uuid.uuid4().hex[:8]}",
                "event_type": "model_uploaded",
                "model_id": record["model_id"],
                "model_type": model_type,
                "name": name,
                "version": version,
            }
        )
        return jsonify({"model": _model_summary(record, full=True)}), 201

    @app.post("/models/activate")
    def activate_model() -> tuple:
        payload = request.get_json(silent=True) or {}
        model_id = payload.get("model_id")
        if not model_id:
            return jsonify({"error": "model_id is required."}), 400
        try:
            record = registry.activate(model_id)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        interaction_store.append(
            {
                "event_id": f"model-activate-{uuid.uuid4().hex[:8]}",
                "event_type": "model_activated",
                "model_id": record["model_id"],
                "model_type": record["model_type"],
                "name": record["name"],
                "version": record["version"],
            }
        )
        return jsonify({"model": _model_summary(record, full=True)})

    @app.post("/recommendations/quick-reorder")
    def quick_reorder() -> tuple:
        payload = request.get_json(silent=True) or {}
        customer_id = payload.get("customer_id")
        k = int(payload.get("k", 5))
        if not customer_id:
            return jsonify({"error": "customer_id is required."}), 400

        active, runtime = registry.load_active_runtime("recommender")
        event_id = f"rec-{uuid.uuid4().hex[:12]}"
        recommendations = runtime.explain_quick_reorder(customer_id, k=k)
        interaction_store.append(
            {
                "event_id": event_id,
                "event_type": "recommendation_shown",
                "route": "quick_reorder",
                "customer_id": customer_id,
                "model_id": active["model_id"],
                "model_name": active["name"],
                "model_version": active["version"],
                "recommended": [item["product_id"] for item in recommendations],
            }
        )
        return jsonify({"event_id": event_id, "model": _model_summary(active), "recommendations": recommendations})

    @app.post("/recommendations/next-order")
    def next_order() -> tuple:
        payload = request.get_json(silent=True) or {}
        customer_id = payload.get("customer_id")
        k = int(payload.get("k", 10))
        fairness = bool(payload.get("fairness", True))
        if not customer_id:
            return jsonify({"error": "customer_id is required."}), 400

        active, runtime = registry.load_active_runtime("recommender")
        event_id = f"rec-{uuid.uuid4().hex[:12]}"
        recommendations = runtime.explain_next_order(customer_id, k=k, fairness=fairness)
        interaction_store.append(
            {
                "event_id": event_id,
                "event_type": "recommendation_shown",
                "route": "next_order",
                "customer_id": customer_id,
                "model_id": active["model_id"],
                "model_name": active["name"],
                "model_version": active["version"],
                "fairness": fairness,
                "recommended": [item["product_id"] for item in recommendations],
            }
        )
        return jsonify({"event_id": event_id, "model": _model_summary(active), "recommendations": recommendations})

    @app.post("/recommendations/outcome")
    def recommendation_outcome() -> tuple:
        payload = request.get_json(silent=True) or {}
        event_id = payload.get("event_id")
        if not event_id:
            return jsonify({"error": "event_id is required."}), 400
        accepted = payload.get("accepted") or []
        added = payload.get("added_outside_recommendation") or []
        interaction_store.append(
            {
                "event_id": event_id,
                "event_type": "recommendation_outcome",
                "accepted": accepted,
                "added_outside_recommendation": added,
            }
        )
        return jsonify({"status": "recorded", "event_id": event_id})

    @app.post("/quality/inspect")
    def inspect_quality() -> tuple:
        payload = request.form if request.form else request.get_json(silent=True) or {}
        producer_id = payload.get("producer_id")
        product_type = payload.get("product_type")
        quantity = int(payload.get("quantity", 1))
        if not producer_id or not product_type:
            return jsonify({"error": "producer_id and product_type are required."}), 400

        active, runtime_record = registry.load_active_runtime("quality")
        runtime = _build_quality_runtime(registry, active, runtime_record)

        image_path = payload.get("image_path")
        temp_path = None
        uploaded = request.files.get("image")
        try:
            if uploaded is not None:
                suffix = Path(uploaded.filename or "inspection.jpg").suffix or ".jpg"
                temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_handle.close()
                temp_path = Path(temp_handle.name)
                uploaded.stream.seek(0)
                uploaded.save(str(temp_path))
                image_path = str(temp_path)
            if not image_path:
                return jsonify({"error": "image file or image_path is required."}), 400

            inspection = runtime.inspect(image_path)
            inventory = _rebuild_inventory(interaction_store.read_all())
            record = inventory.process_inspection(
                producer_id=producer_id,
                product_type=product_type,
                quantity=quantity,
                color_score=inspection["quality"]["color_score"],
                size_score=inspection["quality"]["size_score"],
                ripeness_score=inspection["quality"]["ripeness_score"],
                model_confidence=inspection["confidence"],
            )
            interaction_store.append(
                {
                    "event_id": f"quality-{uuid.uuid4().hex[:12]}",
                    "event_type": "quality_inspection",
                    "producer_id": producer_id,
                    "product_type": product_type,
                    "quantity": quantity,
                    "model_id": active["model_id"],
                    "model_name": active["name"],
                    "model_version": active["version"],
                    "predicted_class_label": inspection["predicted_class_label"],
                    "confidence": inspection["confidence"],
                    "quality": inspection["quality"],
                    "grade": record["grade"],
                    "action": record["action"],
                }
            )
            response = {
                "model": _model_summary(active),
                "inspection": {
                    **inspection,
                    "producer_id": producer_id,
                    "product_type": product_type,
                    "quantity": quantity,
                    "action": record["action"],
                    "reason_text": _quality_reason_text(inspection["quality"], record["grade"], record["action"]),
                    "thresholds": {
                        "grade_b": {"color": 75, "size": 80, "ripeness": 70},
                        "grade_c": {"color": 65, "size": 70, "ripeness": 60},
                    },
                },
            }
            return jsonify(response)
        except UnidentifiedImageError:
            return jsonify({"error": "Unsupported or invalid image file. Please upload a valid PNG or JPEG image."}), 400
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @app.get("/admin/overview")
    def admin_overview() -> tuple:
        interactions = interaction_store.read_all()
        recommendation_events = [record for record in interactions if record.get("event_type") == "recommendation_shown"]
        outcomes = [record for record in interactions if record.get("event_type") == "recommendation_outcome"]
        quality_events = [record for record in interactions if record.get("event_type") == "quality_inspection"]
        low_confidence = [record for record in quality_events if float(record.get("confidence", 0.0)) < 0.60]
        manual_review = [record for record in quality_events if record.get("action") == "manual_review"]

        accepted_count = sum(len(record.get("accepted", [])) for record in outcomes)
        shown_count = sum(len(record.get("recommended", [])) for record in recommendation_events)
        override_count = sum(len(record.get("added_outside_recommendation", [])) for record in outcomes)

        return jsonify(
            {
                "generated_at": utc_now_iso(),
                "active_models": {
                    "recommender": _model_summary(registry.get_active_model("recommender")),
                    "quality": _model_summary(registry.get_active_model("quality")),
                },
                "metrics": {
                    "recommendation_events": len(recommendation_events),
                    "accepted_recommendations": accepted_count,
                    "recommended_items_shown": shown_count,
                    "override_count": override_count,
                    "override_rate": round(override_count / max(len(outcomes), 1), 4),
                    "acceptance_rate": round(accepted_count / max(shown_count, 1), 4),
                    "quality_inspections": len(quality_events),
                    "manual_review_cases": len(manual_review),
                    "low_confidence_cases": len(low_confidence),
                    "grade_distribution": _count_by_key(quality_events, "grade"),
                },
                "recent_examples": {
                    "recommendations": list(reversed(recommendation_events[-5:])),
                    "quality_inspections": list(reversed(quality_events[-5:])),
                },
            }
        )

    @app.get("/admin/interactions")
    def admin_interactions() -> tuple:
        event_type = request.args.get("event_type")
        producer_id = request.args.get("producer_id")
        limit = int(request.args.get("limit", 50))
        return jsonify({"interactions": interaction_store.query(event_type=event_type, producer_id=producer_id, limit=limit)})

    return app


def _model_summary(model: dict | None, *, full: bool = False) -> dict | None:
    if model is None:
        return None
    summary = {
        "model_id": model["model_id"],
        "model_type": model["model_type"],
        "name": model["name"],
        "version": model["version"],
        "runtime": model["runtime"],
        "is_active": model.get("is_active", False),
        "uploaded_at": model.get("uploaded_at"),
        "activated_at": model.get("activated_at"),
    }
    if full:
        summary["metadata"] = model.get("metadata", {})
        summary["file_path"] = model.get("file_path")
    return summary


def _build_quality_runtime(registry: ModelRegistry, active: dict, runtime_record):
    if isinstance(runtime_record, QualityRuntime):
        return runtime_record
    file_path = active.get("file_path")
    runtime = QualityRuntime(
        runtime=active["runtime"],
        file_path=None if not file_path else registry.registry_path.parent / file_path,
        class_names=active.get("metadata", {}).get("class_names"),
    )
    registry._cache[active["model_id"]] = runtime
    return runtime


def _rebuild_inventory(interactions: list[dict]) -> ProducerInventory:
    inventory = ProducerInventory()
    for record in interactions:
        if record.get("event_type") != "quality_inspection":
            continue
        inventory.process_inspection(
            producer_id=record["producer_id"],
            product_type=record["product_type"],
            quantity=int(record.get("quantity", 1)),
            color_score=float(record["quality"]["color_score"]),
            size_score=float(record["quality"]["size_score"]),
            ripeness_score=float(record["quality"]["ripeness_score"]),
            model_confidence=float(record.get("confidence", 0.0)),
        )
    return inventory


def _count_by_key(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _quality_reason_text(quality: dict, grade: str, action: str) -> str:
    thresholds = QualityThresholds()
    color = float(quality["color_score"])
    size = float(quality["size_score"])
    ripeness = float(quality["ripeness_score"])
    if grade == "C":
        trigger = []
        if color < thresholds.color_c:
            trigger.append(f"color below {thresholds.color_c:.0f}")
        if size < thresholds.size_c:
            trigger.append(f"size below {thresholds.size_c:.0f}")
        if ripeness < thresholds.ripeness_c:
            trigger.append(f"ripeness below {thresholds.ripeness_c:.0f}")
        return f"Grade C because {', '.join(trigger)}. Suggested action: {action.replace('_', ' ')}."
    if grade == "B":
        trigger = []
        if color < thresholds.color_b:
            trigger.append(f"color below {thresholds.color_b:.0f}")
        if size < thresholds.size_b:
            trigger.append(f"size below {thresholds.size_b:.0f}")
        if ripeness < thresholds.ripeness_b:
            trigger.append(f"ripeness below {thresholds.ripeness_b:.0f}")
        return f"Grade B because {', '.join(trigger)}. Suggested action: {action.replace('_', ' ')}."
    return f"Grade A because color, size, and ripeness all cleared the threshold rules. Suggested action: {action.replace('_', ' ')}."


def _parse_metadata(raw_metadata) -> dict:
    if raw_metadata is None:
        return {}
    if isinstance(raw_metadata, dict):
        return raw_metadata
    try:
        return json.loads(raw_metadata)
    except json.JSONDecodeError:
        return {"raw": str(raw_metadata)}


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5050, debug=True)
