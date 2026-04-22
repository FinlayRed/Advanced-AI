from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .grading import assign_grade, quality_breakdown
from .train import QualityNet


def load_model(checkpoint_path: str, device: torch.device) -> QualityNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    num_classes = int(config.get("num_classes", 2))

    model = QualityNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(image)
    return tensor.unsqueeze(0)


def predict(checkpoint_path: str, image_path: str) -> dict:
    """Predict class, quality scores, and final grade for one image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device=device)
    image_batch = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits, raw_scores = model(image_batch)
        probs = torch.softmax(logits, dim=1)
        predicted_class_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(torch.max(probs, dim=1).values.item())

        scores = raw_scores.squeeze(0).cpu().numpy()
        color = float(min(max(scores[0], 0.0), 100.0))
        size = float(min(max(scores[1], 0.0), 100.0))
        ripeness = float(min(max(scores[2], 0.0), 100.0))

    grade = assign_grade(color=color, size=size, ripeness=ripeness)

    return {
        "image_path": str(Path(image_path)),
        "predicted_class_idx": predicted_class_idx,
        "confidence": round(confidence, 4),
        "quality": quality_breakdown(color=color, size=size, ripeness=ripeness),
        "grade": grade,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_quality_model.pt")
    parser.add_argument("--image", required=True, help="Path to a fruit/vegetable image")
    args = parser.parse_args()

    result = predict(checkpoint_path=args.checkpoint, image_path=args.image)
    print("Prediction result:")
    print(result)


if __name__ == "__main__":
    main()

