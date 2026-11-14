"""
Validates a processed Food-11 dataset against a data contract.

Usage:
    uv run python src/food11/validate_data.py data/food11_processed_mini
    uv run python src/food11/validate_data.py data/food11_processed --min-per-class 50 --tau 0.05

Exits with status 1 if any check fails.
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

CLASSES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
           "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
SPLITS = ["training", "validation", "evaluation"]
IMAGE_SIZE = (128, 128)


def list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_file())


def check_class_folders(root: Path) -> list[str]:
    errors = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            errors.append(f"{split}: folder missing")
            continue
        found = {p.name for p in split_dir.iterdir() if p.is_dir()}
        missing = set(CLASSES) - found
        extra = found - set(CLASSES)
        if missing:
            errors.append(f"{split}: missing {sorted(missing)}")
        if extra:
            errors.append(f"{split}: unexpected {sorted(extra)}")
    return errors


def check_images(root: Path) -> list[str]:
    errors = []
    for split in SPLITS:
        for cls in CLASSES:
            class_dir = root / split / cls
            if not class_dir.is_dir():
                continue
            for path in list_images(class_dir):
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB":
                            errors.append(f"{path}: mode {img.mode}, expected RGB")
                        if img.size != IMAGE_SIZE:
                            errors.append(f"{path}: size {img.size}, expected {IMAGE_SIZE}")
                except (OSError, ValueError) as exc:  # unreadable or truncated file
                    errors.append(f"{path}: cannot open ({exc})")
    return errors


def check_min_per_class(root: Path, min_per_class: int) -> list[str]:
    errors = []
    for split in SPLITS:
        for cls in CLASSES:
            class_dir = root / split / cls
            count = len(list_images(class_dir)) if class_dir.is_dir() else 0
            if count < min_per_class:
                errors.append(f"{split}/{cls}: {count} images < {min_per_class}")
    return errors


def check_no_cross_split_duplicates(root: Path) -> list[str]:
    seen = defaultdict(set)  # md5 -> {(split, relative path)}
    for split in SPLITS:
        for cls in CLASSES:
            class_dir = root / split / cls
            if not class_dir.is_dir():
                continue
            for path in list_images(class_dir):
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                seen[digest].add((split, str(path.relative_to(root))))
    errors = []
    for files in seen.values():
        if len({split for split, _ in files}) > 1:
            errors.append("same image in several splits: " + ", ".join(sorted(p for _, p in files)))
    return errors


def check_class_balance(root: Path, tau: float) -> list[str]:
    def shares(split: str) -> dict[str, float]:
        counts = {}
        for cls in CLASSES:
            class_dir = root / split / cls
            counts[cls] = len(list_images(class_dir)) if class_dir.is_dir() else 0
        total = sum(counts.values()) or 1
        return {cls: n / total for cls, n in counts.items()}

    train, val = shares("training"), shares("validation")
    errors = []
    for cls in CLASSES:
        diff = abs(train[cls] - val[cls])
        if diff > tau:
            errors.append(f"{cls}: training {train[cls]:.3f} vs validation {val[cls]:.3f} (diff {diff:.3f} > {tau})")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a processed Food-11 dataset")
    parser.add_argument("root", type=Path, help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    checks = [
        ("1. 11 class folders per split", check_class_folders(args.root)),
        ("2. images open, RGB, 128x128", check_images(args.root)),
        (f"3. at least {args.min_per_class} images per class", check_min_per_class(args.root, args.min_per_class)),
        ("4. no image in two splits", check_no_cross_split_duplicates(args.root)),
        (f"5. class shares train vs val within {args.tau}", check_class_balance(args.root, args.tau)),
    ]

    failed = False
    print(f"Data contract for {args.root}")
    for name, errors in checks:
        print(f"[{'PASS' if not errors else 'FAIL'}] {name}")
        for err in errors[:10]:
            print(f"       - {err}")
        if len(errors) > 10:
            print(f"       ... and {len(errors) - 10} more")
        failed = failed or bool(errors)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()