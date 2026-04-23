from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from src.quality.grading import QualityThresholds


@dataclass
class QualityRuntime:
    runtime: str
    file_path: Path | None = None
    class_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.device = None
        self.model = None
        if self.runtime == "quality_checkpoint":
            if self.file_path is None:
                raise ValueError("Checkpoint runtime requires a file path.")
            import torch

            from src.quality.infer import load_model

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = load_model(str(self.file_path), device=self.device)

    def inspect(self, image_path: str) -> dict:
        if self.runtime == "quality_checkpoint":
            from src.quality.infer import predict_with_model

            return predict_with_model(
                model=self.model,
                image_path=image_path,
                device=self.device,
                class_names=self.class_names,
            )
        return self._heuristic_inspection(image_path)

    def _heuristic_inspection(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.asarray(image, dtype=np.float32)
        rgb_mean = arr.mean(axis=(0, 1))
        brightness = float(arr.mean() / 255.0)
        contrast = float(arr.std() / 128.0)
        channel_spread = float(np.std(rgb_mean) / 64.0)

        color_score = np.clip((0.55 * brightness + 0.45 * channel_spread) * 100.0, 0.0, 100.0)
        size_score = np.clip((0.65 + min(contrast, 1.0) * 0.35) * 100.0, 0.0, 100.0)
        ripeness_midpoint = 1.0 - abs(brightness - 0.62) / 0.62
        ripeness_score = np.clip((0.6 * ripeness_midpoint + 0.4 * min(channel_spread, 1.0)) * 100.0, 0.0, 100.0)

        average_quality = float((color_score + size_score + ripeness_score) / 3.0)
        confidence = round(float(np.clip(0.55 + average_quality / 250.0, 0.55, 0.98)), 4)
        grade = "A"
        thresholds = QualityThresholds()
        if color_score < thresholds.color_c or size_score < thresholds.size_c or ripeness_score < thresholds.ripeness_c:
            grade = "C"
        elif color_score < thresholds.color_b or size_score < thresholds.size_b or ripeness_score < thresholds.ripeness_b:
            grade = "B"

        predicted_class_idx = 0 if grade == "A" else 1
        class_names = self.class_names or ["fresh", "rotten"]
        predicted_class_label = class_names[min(predicted_class_idx, len(class_names) - 1)]

        return {
            "image_path": str(Path(image_path)),
            "predicted_class_idx": predicted_class_idx,
            "predicted_class_label": predicted_class_label,
            "confidence": confidence,
            "quality": {
                "color_score": round(float(color_score), 2),
                "size_score": round(float(size_score), 2),
                "ripeness_score": round(float(ripeness_score), 2),
            },
            "grade": grade,
        }
