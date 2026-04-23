from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetColumns:
    image_path: str = "image_path"
    class_idx: str = "class_idx"
    color_score: str = "color_score"
    size_score: str = "size_score"
    ripeness_score: str = "ripeness_score"


class FruitVegDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        split: str = "train",
        columns: DatasetColumns = DatasetColumns(),
        imagenet_normalize: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.columns = columns

        if split == "train":
            transform_steps = [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
        else:
            transform_steps = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]

        if imagenet_normalize:
            transform_steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        self.transform = transforms.Compose(transform_steps)

        self._validate_schema()

    def _validate_schema(self) -> None:
        required = {
            self.columns.image_path,
            self.columns.class_idx,
            self.columns.color_score,
            self.columns.size_score,
            self.columns.ripeness_score,
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_root / str(row[self.columns.image_path])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        class_idx = torch.tensor(int(row[self.columns.class_idx]), dtype=torch.long)
        scores = torch.tensor(
            [
                float(row[self.columns.color_score]),
                float(row[self.columns.size_score]),
                float(row[self.columns.ripeness_score]),
            ],
            dtype=torch.float32,
        )

        return image_tensor, class_idx, scores

