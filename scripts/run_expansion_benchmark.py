"""Run a controlled 5-image vs. 30-image dataset-expansion benchmark.

COCO128 supplies reviewed labels so this measures the benefit of successful
collection/annotation separately from label noise. Both arms use the same
20-image holdout and deterministic ordering.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(SETTINGS["datasets_dir"]) / "coco128"
OUT = ROOT / "benchmark_data" / "expansion_30"
RUNS = ROOT / "runs" / "expansion_benchmark"


def copy_pair(stem: str, split: str) -> None:
    image = next((SOURCE / "images" / "train2017").glob(f"{stem}.*"))
    label = SOURCE / "labels" / "train2017" / f"{stem}.txt"
    shutil.copy2(image, OUT / split / "images" / image.name)
    # SearchVision trains one requested class. Keep only COCO class 0 (person),
    # already numbered zero in the generated single-class dataset.
    person_rows = [row for row in label.read_text(encoding="utf-8").splitlines() if row.startswith("0 ")]
    (OUT / split / "labels" / label.name).write_text("\n".join(person_rows) + "\n", encoding="utf-8")


def prepare() -> tuple[Path, Path]:
    if not SOURCE.exists():
        # A validation call lets Ultralytics resolve and download coco128.yaml.
        YOLO(ROOT / "yolov8n.pt").val(data="coco128.yaml", imgsz=640, batch=8, device="cpu", workers=0,
                                      project=str(RUNS), name="dataset_download", plots=False)
    images = sorted((SOURCE / "images" / "train2017").glob("*.jpg"))
    eligible = []
    for image in images:
        label = SOURCE / "labels" / "train2017" / f"{image.stem}.txt"
        if label.exists() and any(row.startswith("0 ") for row in label.read_text(encoding="utf-8").splitlines()):
            eligible.append(image.stem)
    random.Random(42).shuffle(eligible)
    train, val = eligible[:30], eligible[30:50]

    if OUT.exists():
        shutil.rmtree(OUT)
    for arm in ("seed5", "expanded30"):
        for split in ("train", "val"):
            (OUT / arm / split / "images").mkdir(parents=True)
            (OUT / arm / split / "labels").mkdir(parents=True)
        for stem in train[: 5 if arm == "seed5" else 30]:
            copy_pair(stem, f"{arm}/train")
        for stem in val:
            copy_pair(stem, f"{arm}/val")

        yaml = OUT / arm / "data.yaml"
        yaml.write_text(
            f"path: {str((OUT / arm).resolve()).replace(chr(92), '/')}\n"
            "train: train/images\nval: val/images\n"
            "names:\n  0: person\n",
            encoding="utf-8",
        )
    return OUT / "seed5" / "data.yaml", OUT / "expanded30" / "data.yaml"


def best_metrics(run: Path) -> dict[str, float]:
    with (run / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key = next(k for k in rows[0] if "mAP50-95" in k)
    row = max(rows, key=lambda r: float(r[key]))
    return {k.strip(): float(v) for k, v in row.items() if v.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    seed_yaml, expanded_yaml = prepare()
    RUNS.mkdir(parents=True, exist_ok=True)
    for name, data in (("seed5", seed_yaml), ("expanded30", expanded_yaml)):
        YOLO(ROOT / "yolov8n.pt").train(
            data=str(data), epochs=args.epochs, imgsz=640, batch=4, device="cpu", workers=0,
            patience=0, seed=42, deterministic=True, project=str(RUNS), name=name, exist_ok=True,
        )
        print(name, best_metrics(RUNS / name))


if __name__ == "__main__":
    main()
