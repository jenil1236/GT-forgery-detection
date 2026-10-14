"""
download.py - Auto-download CORD-v2 and RVL-CDIP subset.
Sets HF_HOME to data/models_cache so all HuggingFace downloads stay in the project.
Run directly:  python src/data/download.py
Or called by train.py pipeline.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List


def _set_hf_home(project_root: str) -> None:
    cache = os.path.join(project_root, "data", "models_cache")
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache, "datasets")


def download_cord(raw_dir: str, max_samples: int = 800, seed: int = 42) -> Dict:
    from datasets import load_dataset
    from PIL import Image

    out_dir = os.path.join(raw_dir, "cord")
    os.makedirs(out_dir, exist_ok=True)
    splits_info = {}

    for split in ["train", "test", "validation"]:
        split_out = os.path.join(out_dir, split)
        os.makedirs(split_out, exist_ok=True)
        done_file = os.path.join(split_out, ".done")

        if os.path.exists(done_file):
            count = len([f for f in os.listdir(split_out) if f.endswith(".png")])
            print(f"  CORD/{split}: already downloaded ({count} images), skipping.")
            splits_info[split] = count
            continue

        print(f"  Downloading CORD-v2 [{split}] ...")
        ds = load_dataset("naver-clova-ix/cord-v2", split=split, trust_remote_code=True)
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        if len(indices) > max_samples and split == "train":
            indices = rng.sample(indices, max_samples)

        saved = 0
        for idx in indices:
            sample = ds[idx]
            img = sample["image"]
            img_path = os.path.join(split_out, f"cord_{split}_{idx:05d}.png")
            if not os.path.exists(img_path):
                img.save(img_path)
            saved += 1

        Path(done_file).touch()
        print(f"  CORD/{split}: saved {saved} images -> {split_out}")
        splits_info[split] = saved

    return splits_info


def download_rvl_cdip(
    raw_dir: str,
    train_classes: List[str],
    zeroshot_classes: List[str],
    samples_per_class_train: int = 500,
    samples_per_class_zeroshot: int = 300,
    seed: int = 42,
) -> Dict:
    from datasets import load_dataset

    CLASS_NAMES = [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific_report", "scientific_publication", "specification",
        "file_folder", "news_article", "budget", "invoice",
        "presentation", "questionnaire", "resume", "memo"
    ]
    class_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    all_classes = list(dict.fromkeys(train_classes + zeroshot_classes))
    target_ids = {class_to_id[c] for c in all_classes if c in class_to_id}
    samples_needed = {}
    for c in train_classes:
        if c in class_to_id:
            samples_needed[class_to_id[c]] = samples_per_class_train
    for c in zeroshot_classes:
        if c in class_to_id:
            samples_needed[class_to_id[c]] = samples_per_class_zeroshot

    out_dir = os.path.join(raw_dir, "rvl_cdip")
    os.makedirs(out_dir, exist_ok=True)
    done_file = os.path.join(out_dir, ".done")

    if os.path.exists(done_file):
        print("  RVL-CDIP: already downloaded, skipping.")
        counts = {}
        for c in all_classes:
            d = os.path.join(out_dir, c)
            if os.path.isdir(d):
                counts[c] = len([f for f in os.listdir(d) if f.endswith((".png", ".tif"))])
        return counts

    print("  Downloading RVL-CDIP (streaming) ...")
    ds = load_dataset("aharley/rvl_cdip", split="train", streaming=True, trust_remote_code=True)
    counters: Dict[int, int] = {cid: 0 for cid in target_ids}

    for sample in ds:
        label = sample["label"]
        if label not in target_ids:
            continue
        needed = samples_needed.get(label, 0)
        if counters[label] >= needed:
            if all(counters[cid] >= samples_needed.get(cid, 0) for cid in target_ids):
                break
            continue

        class_name = CLASS_NAMES[label]
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        idx = counters[label]
        img_path = os.path.join(class_dir, f"{class_name}_{idx:05d}.png")
        if not os.path.exists(img_path):
            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path)

        counters[label] += 1
        if counters[label] % 50 == 0:
            print(f"    {class_name}: {counters[label]}/{needed}")

    Path(done_file).touch()
    counts = {CLASS_NAMES[cid]: counters[cid] for cid in target_ids}
    for name, cnt in counts.items():
        print(f"  RVL-CDIP/{name}: {cnt} images saved.")
    return counts


def main(args) -> None:
    project_root = args.project_root
    _set_hf_home(project_root)
    print("="*60)
    print("  GT Forgery Detection - Dataset Download")
    print("="*60)

    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("\n[1/2] CORD-v2 (receipts)")
    cord_stats = download_cord(raw_dir, max_samples=args.cord_samples)

    train_classes = ["letter", "form", "invoice", "budget", "memo", "resume"]
    zeroshot_classes = ["advertisement", "scientific_report"]
    print("[2/2] RVL-CDIP subset")
    rvl_stats = download_rvl_cdip(
        raw_dir,
        train_classes=train_classes,
        zeroshot_classes=zeroshot_classes,
        samples_per_class_train=args.rvl_train_samples,
        samples_per_class_zeroshot=args.rvl_zeroshot_samples,
    )

    manifest = {
        "cord": cord_stats,
        "rvl_cdip": rvl_stats,
        "train_doc_types": train_classes + ["cord"],
        "zeroshot_doc_types": zeroshot_classes,
    }
    manifest_path = os.path.join(project_root, "data", "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved -> {manifest_path}")
    print("Download complete.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--cord_samples", type=int, default=800)
    p.add_argument("--rvl_train_samples", type=int, default=500)
    p.add_argument("--rvl_zeroshot_samples", type=int, default=300)
    args = p.parse_args()
    main(args)