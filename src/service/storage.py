from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import joblib

from src.data_generator import generate
from src.recommender import RecommendationService


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InteractionStore:
    log_path: Path | str

    def __post_init__(self) -> None:
        self.log_path = Path(self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> dict:
        record = dict(record)
        record.setdefault("timestamp", utc_now_iso())
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        records: list[dict] = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def query(self, *, event_type: str | None = None, producer_id: str | None = None, limit: int = 50) -> list[dict]:
        records = self.read_all()
        if event_type:
            records = [record for record in records if record.get("event_type") == event_type]
        if producer_id:
            records = [record for record in records if record.get("producer_id") == producer_id]
        return list(reversed(records[-limit:]))


@dataclass
class ModelRegistry:
    models_dir: Path | str = "data/service_models"
    registry_path: Path | str = "data/model_registry.json"
    interaction_store: InteractionStore | None = None
    _cache: dict[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.models_dir = Path(self.models_dir)
        self.registry_path = Path(self.registry_path)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write({"models": []})
        self.ensure_bootstrap_models()

    def _read(self) -> dict:
        with self.registry_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, payload: dict) -> None:
        with self.registry_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def list_models(self) -> list[dict]:
        return self._read()["models"]

    def get_model(self, model_id: str) -> dict:
        for model in self.list_models():
            if model["model_id"] == model_id:
                return model
        raise KeyError(f"Unknown model id: {model_id}")

    def get_active_model(self, model_type: str) -> dict | None:
        for model in self.list_models():
            if model["model_type"] == model_type and model.get("is_active"):
                return model
        return None

    def _persist_model(self, record: dict) -> dict:
        payload = self._read()
        replaced = False
        for index, existing in enumerate(payload["models"]):
            if existing["model_id"] == record["model_id"]:
                payload["models"][index] = record
                replaced = True
                break
        if not replaced:
            payload["models"].append(record)
        self._write(payload)
        return record

    def register_existing(
        self,
        *,
        model_type: str,
        runtime: str,
        source_path: Path | str | None,
        name: str,
        version: str,
        is_active: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        source_file = Path(source_path) if source_path else None
        destination_rel_path = None
        if source_file is not None:
            ext = source_file.suffix or ".bin"
            model_id = uuid.uuid4().hex[:12]
            destination_dir = self.models_dir / model_type
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{model_id}{ext}"
            if source_file.resolve() != destination.resolve():
                shutil.copy2(source_file, destination)
            destination_rel_path = str(destination.relative_to(self.registry_path.parent))
        else:
            model_id = uuid.uuid4().hex[:12]

        record = {
            "model_id": model_id,
            "model_type": model_type,
            "runtime": runtime,
            "name": name,
            "version": version,
            "file_path": destination_rel_path,
            "uploaded_at": utc_now_iso(),
            "activated_at": utc_now_iso() if is_active else None,
            "is_active": bool(is_active),
            "metadata": metadata or {},
        }
        if is_active:
            self._deactivate_model_type(model_type)
        return self._persist_model(record)

    def upload_model(
        self,
        *,
        file_obj: BinaryIO,
        filename: str,
        model_type: str,
        name: str,
        version: str,
        runtime: str,
        metadata: dict | None = None,
        activate: bool = False,
    ) -> dict:
        model_id = uuid.uuid4().hex[:12]
        ext = Path(filename).suffix or ".bin"
        destination_dir = self.models_dir / model_type
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{model_id}{ext}"
        with destination.open("wb") as handle:
            shutil.copyfileobj(file_obj, handle)

        if activate:
            self._deactivate_model_type(model_type)
        record = {
            "model_id": model_id,
            "model_type": model_type,
            "runtime": runtime,
            "name": name,
            "version": version,
            "file_path": str(destination.relative_to(self.registry_path.parent)),
            "uploaded_at": utc_now_iso(),
            "activated_at": utc_now_iso() if activate else None,
            "is_active": bool(activate),
            "metadata": metadata or {},
        }
        return self._persist_model(record)

    def _deactivate_model_type(self, model_type: str) -> None:
        payload = self._read()
        for model in payload["models"]:
            if model["model_type"] == model_type:
                model["is_active"] = False
                model["activated_at"] = None
        self._write(payload)

    def activate(self, model_id: str) -> dict:
        payload = self._read()
        target = None
        for model in payload["models"]:
            if model["model_id"] == model_id:
                target = model
                break
        if target is None:
            raise KeyError(f"Unknown model id: {model_id}")

        for model in payload["models"]:
            if model["model_type"] == target["model_type"]:
                model["is_active"] = False
                model["activated_at"] = None
        target["is_active"] = True
        target["activated_at"] = utc_now_iso()
        self._write(payload)
        self._cache.pop(target["model_type"], None)
        return target

    def load_active_runtime(self, model_type: str):
        active = self.get_active_model(model_type)
        if active is None:
            raise RuntimeError(f"No active model for type '{model_type}'")
        cached = self._cache.get(active["model_id"])
        if cached is not None:
            return active, cached

        file_path = active.get("file_path")
        absolute_path = None if not file_path else self.registry_path.parent / file_path
        if active["runtime"] == "recommendation_service":
            runtime = RecommendationService.load(absolute_path)
        elif active["runtime"] in {"quality_checkpoint", "quality_heuristic"}:
            runtime = active
        else:
            raise RuntimeError(f"Unsupported runtime: {active['runtime']}")
        self._cache[active["model_id"]] = runtime
        return active, runtime

    def ensure_bootstrap_models(self) -> None:
        if self.get_active_model("recommender") is None:
            bootstrap_dir = self.models_dir / "bootstrap"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)
            data = generate(bootstrap_dir / "sample_data")
            service = RecommendationService(products=data["products"]).fit(data["orders"])
            artefact = bootstrap_dir / "bootstrap_recommendation_service.joblib"
            joblib.dump(service, artefact)
            self.register_existing(
                model_type="recommender",
                runtime="recommendation_service",
                source_path=artefact,
                name="bootstrap-recommender",
                version="1.0",
                is_active=True,
                metadata={"source": "synthetic bootstrap", "records": len(data["orders"])} ,
            )

        if self.get_active_model("quality") is None:
            self.register_existing(
                model_type="quality",
                runtime="quality_heuristic",
                source_path=None,
                name="bootstrap-quality-heuristic",
                version="1.0",
                is_active=True,
                metadata={"class_names": ["fresh", "rotten"], "source": "image heuristics bootstrap"},
            )
