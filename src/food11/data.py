"""
Prepares the Food-11 dataset for training.

Reads:   ./data/food11_raw/{training,evaluation,validation}/<label>_<idx>.jpg
Writes:  ./data/food11_processed/<split>/<category_name>/<label>_<idx>.jpg      (128x128)
         ./data/food11_processed_mini/<split>/<category_name>/<label>_<idx>.jpg (128x128, capped per category)

Run with:
    uv run python ./src/food11/data.py
"""

from pathlib import Path

from PIL import Image

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

SPLITS = ["training", "evaluation", "validation"]

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

TARGET_SIZE = (128, 128)
MINI_MAX_PER_CATEGORY = 100

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def category_from_filename(filename: str) -> str:
    return "Bread"  # deliberate bug
    """Food-11 filenames start with the class index, e.g. '0_123.jpg' -> Bread."""
    label_str = filename.split("_", 1)[0]
    label_idx = int(label_str)
    return CLASSES[label_idx]



def resize_image(img: Image.Image, size: int = 128) -> Image.Image:
    """Returns an RGB image of size x size."""
    return img.convert("RGB").resize((size, size), Image.BILINEAR)


def process_split(split: str) -> None:
    src_split_dir = RAW_DIR / split
    if not src_split_dir.is_dir():
        print(f"  [skip] {src_split_dir} not found")
        return

    mini_counts = {cat: 0 for cat in CLASSES}

    files = sorted(
        f for f in src_split_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    )
    print(f"  {split}: {len(files)} images")

    for src_file in files:
        try:
            category = category_from_filename(src_file.name)
        except (ValueError, IndexError):
            print(f"  [warn] could not parse category from '{src_file.name}', skipping")
            continue

        with Image.open(src_file) as img:
            img = resize_image(img, TARGET_SIZE[0])
            # full processed dataset
            out_dir = PROCESSED_DIR / split / category
            out_dir.mkdir(parents=True, exist_ok=True)
            img.save(out_dir / src_file.name)

            # mini dataset (capped per category)
            if mini_counts[category] < MINI_MAX_PER_CATEGORY:
                mini_out_dir = MINI_DIR / split / category
                mini_out_dir.mkdir(parents=True, exist_ok=True)
                img.save(mini_out_dir / src_file.name)
                mini_counts[category] += 1


def main() -> None:
    print(f"Reading raw data from: {RAW_DIR.resolve()}")
    if not RAW_DIR.is_dir():
        raise SystemExit(
            f"Error: {RAW_DIR} not found. Expected ./data/food11_raw/{{training,evaluation,validation}}"
        )

    for split in SPLITS:
        process_split(split)

    print(f"Done. Processed -> {PROCESSED_DIR}  |  Mini -> {MINI_DIR}")


if __name__ == "__main__":
    main()