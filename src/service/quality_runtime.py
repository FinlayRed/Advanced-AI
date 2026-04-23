from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityRuntime:
    runtime: str
    file_path: Path | None = None
    class_names: list[str] | None = None

    def __post_init__(self) -> None:
        if self.runtime != "quality_checkpoint":
            raise ValueError(f"Unsupported quality runtime: {self.runtime}")
        if self.file_path is None:
            raise ValueError("Checkpoint runtime requires a file path.")
        if not self.file_path.exists():
            raise FileNotFoundError(f"Quality checkpoint not found: {self.file_path}")

        import torch

        from src.quality.infer import load_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(str(self.file_path), device=self.device)

    def inspect(self, image_path: str) -> dict:
        from src.quality.infer import predict_with_model

        return predict_with_model(
            model=self.model,
            image_path=image_path,
            device=self.device,
            class_names=self.class_names,
        )
