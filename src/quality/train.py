from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from .dataset import FruitVegDataset


@dataclass
class TrainConfig:
    train_csv: str
    val_csv: str
    image_root: str
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    cls_weight: float = 1.0
    reg_weight: float = 0.5
    num_classes: int = 2
    save_dir: str = "models/quality"


class QualityNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        self.regressor = nn.Linear(in_features, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        scores = self.regressor(features)
        return logits, scores


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    ce_loss: nn.Module,
    mse_loss: nn.Module,
    device: torch.device,
    cls_weight: float,
    reg_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_mae = torch.zeros(3, device=device)

    with torch.no_grad():
        for images, y_class, y_scores in loader:
            images = images.to(device)
            y_class = y_class.to(device)
            y_scores = y_scores.to(device)

            logits, pred_scores = model(images)
            loss_cls = ce_loss(logits, y_class)
            loss_reg = mse_loss(pred_scores, y_scores)
            loss = cls_weight * loss_cls + reg_weight * loss_reg

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == y_class).sum().item()
            total_items += images.size(0)
            total_mae += torch.abs(pred_scores - y_scores).sum(dim=0)

    denom = max(total_items, 1)
    return {
        "loss": total_loss / denom,
        "acc": total_correct / denom,
        "mae_color": (total_mae[0] / denom).item(),
        "mae_size": (total_mae[1] / denom).item(),
        "mae_ripeness": (total_mae[2] / denom).item(),
    }


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    train_ds = FruitVegDataset(csv_path=cfg.train_csv, image_root=cfg.image_root, split="train")
    val_ds = FruitVegDataset(csv_path=cfg.val_csv, image_root=cfg.image_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = QualityNet(num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, y_class, y_scores in train_loader:
            images = images.to(device)
            y_class = y_class.to(device)
            y_scores = y_scores.to(device)

            logits, pred_scores = model(images)
            loss_cls = ce_loss(logits, y_class)
            loss_reg = mse_loss(pred_scores, y_scores)
            loss = cfg.cls_weight * loss_cls + cfg.reg_weight * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            ce_loss=ce_loss,
            mse_loss=mse_loss,
            device=device,
            cls_weight=cfg.cls_weight,
            reg_weight=cfg.reg_weight,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_mae(c,s,r)=({val_metrics['mae_color']:.2f},"
            f"{val_metrics['mae_size']:.2f},{val_metrics['mae_ripeness']:.2f})"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = Path(cfg.save_dir) / "best_quality_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint -> {checkpoint_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="models/quality")
    args = parser.parse_args()

    return TrainConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_root=args.image_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    train(parse_args())

