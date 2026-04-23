from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def label_for_folder(folder_name: str) -> str | None:
    if "__" not in folder_name:
        return None
    label = folder_name.rsplit("__", 1)[1].lower()
    if label == "healthy":
        return "healthy"
    if label == "rotten":
        return "rotten"
    return None


def row_for_image(image_path: Path, image_root: Path) -> dict[str, str]:
    label = label_for_folder(image_path.parent.name)
    if label == "healthy":
        class_idx = 0
        score = "90"
    elif label == "rotten":
        class_idx = 1
        score = "20"
    else:
        raise ValueError(f"Unsupported label folder: {image_path.parent.name}")

    return {
        "image_path": image_path.relative_to(image_root).as_posix(),
        "class_idx": str(class_idx),
        "color_score": score,
        "size_score": score,
        "ripeness_score": score,
    }


def build_rows(image_root: Path, *, max_per_class: int | None, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for condition in ("healthy", "rotten"):
        images = [
            path
            for folder in image_root.iterdir()
            if folder.is_dir() and label_for_folder(folder.name) == condition
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        images.sort()
        rng.shuffle(images)
        if max_per_class is not None:
            images = images[:max_per_class]
        rows.extend(row_for_image(path, image_root) for path in images)

    rng.shuffle(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "class_idx", "color_score", "size_score", "ripeness_score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    rows = build_rows(image_root, max_per_class=args.max_per_class, seed=args.seed)
    if not rows:
        raise ValueError(f"No Healthy/Rotten images found under {image_root}")

    split_idx = int(len(rows) * (1.0 - args.val_fraction))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]
    write_csv(output_dir / "quality_train.csv", train_rows)
    write_csv(output_dir / "quality_val.csv", val_rows)
    print(f"Wrote {len(train_rows)} train rows and {len(val_rows)} val rows to {output_dir}")


if __name__ == "__main__":
    main()
