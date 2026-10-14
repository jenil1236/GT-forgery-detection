"""
forge.py - Synthetic forgery generation pipeline.

Three forgery types:
  1. TextReplace  - detect text regions, overwrite with similar text
  2. CopyMove     - copy a patch from one region, paste elsewhere
  3. NoiseInject  - add JPEG artefacts + Gaussian noise in a local region

Each real image produces one forged version.
Outputs:
  data/processed/{doc_type}/real/{split}/img_XXXXX.png
  data/processed/{doc_type}/forged/{split}/img_XXXXX.png
  data/processed/{doc_type}/forged/{split}/img_XXXXX_mask.png
  data/processed/{doc_type}/forged/{split}/img_XXXXX_meta.json
"""

from __future__ import annotations
import io
import json
import os
import random
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


class ForgeType(str, Enum):
    TEXT_REPLACE = "text_replace"
    COPY_MOVE = "copy_move"
    NOISE_INJECT = "noise_inject"


class ForgeryGenerator:
    """
    Generates synthetic document forgeries.
    All randomness is controlled via a seeded RNG for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate(
        self,
        image: Image.Image,
        forge_type: Optional[ForgeType] = None,
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """
        Generate a forged version of the input image.

        Returns:
            forged_image: PIL Image
            mask:         PIL Image (binary, white = tampered region)
            meta:         dict with forge_type, region, etc.
        """
        if forge_type is None:
            forge_type = self.rng.choice(list(ForgeType))

        img = image.copy().convert("RGB")

        if forge_type == ForgeType.TEXT_REPLACE:
            return self._text_replace(img)
        elif forge_type == ForgeType.COPY_MOVE:
            return self._copy_move(img)
        else:
            return self._noise_inject(img)

    # ── Forgery methods ───────────────────────────────────────────────────────

    def _text_replace(
        self, img: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """
        Detect a rectangular text-like region using edge/contour detection,
        then overwrite it with random-length digit/text strings.
        """
        W, H = img.size
        arr = np.array(img)

        # Find text-like regions via simple thresholding on a grayscale crop
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Use a horizontal strip in the middle third of the image
        strip_y1 = H // 4
        strip_y2 = 3 * H // 4
        strip = gray[strip_y1:strip_y2, :]

        # Find bounding boxes of non-white (text) regions
        _, thresh = cv2.threshold(strip, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        region = None
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, w, h = cv2.boundingRect(cnt)
            # look for regions that look like text lines
            if w > W * 0.1 and h > 8 and h < H * 0.1:
                # expand slightly
                pad = 4
                rx1 = max(0, x - pad)
                ry1 = max(0, strip_y1 + y - pad)
                rx2 = min(W, x + w + pad)
                ry2 = min(H, strip_y1 + y + h + pad)
                region = (rx1, ry1, rx2, ry2)
                break

        if region is None:
            # fallback: random small rectangle
            rx1 = self.rng.randint(W // 4, W // 2)
            ry1 = self.rng.randint(H // 4, H // 2)
            rx2 = rx1 + self.rng.randint(50, 150)
            ry2 = ry1 + self.rng.randint(15, 35)
            rx2, ry2 = min(rx2, W - 1), min(ry2, H - 1)
            region = (rx1, ry1, rx2, ry2)

        rx1, ry1, rx2, ry2 = region
        region_w, region_h = rx2 - rx1, ry2 - ry1

        # Sample background colour from nearby pixels
        bg_sample = arr[max(0, ry1 - 3):ry1, rx1:rx2]
        if bg_sample.size > 0:
            bg_color = tuple(int(v) for v in bg_sample.mean(axis=(0, 1)))
        else:
            bg_color = (255, 255, 255)

        # Draw replacement text
        draw = ImageDraw.Draw(img)
        draw.rectangle([rx1, ry1, rx2, ry2], fill=bg_color)

        fake_text = self._random_text(region_w)
        font_size = max(8, min(region_h - 4, 18))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        text_color = (30, 30, 30)
        draw.text((rx1 + 2, ry1 + 2), fake_text, fill=text_color, font=font)

        # Build mask
        mask = Image.new("L", (W, H), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([rx1, ry1, rx2, ry2], fill=255)

        meta = {
            "forge_type": ForgeType.TEXT_REPLACE.value,
            "region": [rx1, ry1, rx2, ry2],
            "fake_text": fake_text,
        }
        return img, mask, meta

    def _copy_move(
        self, img: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """Copy a patch from one area and paste it at a different location."""
        W, H = img.size
        patch_w = self.rng.randint(30, min(100, W // 4))
        patch_h = self.rng.randint(30, min(100, H // 4))

        # source patch
        sx = self.rng.randint(0, W - patch_w - 1)
        sy = self.rng.randint(0, H - patch_h - 1)
        patch = img.crop((sx, sy, sx + patch_w, sy + patch_h))

        # slight transform on patch
        angle = self.rng.uniform(-5, 5)
        patch = patch.rotate(angle, resample=Image.BILINEAR, expand=False)
        # slight brightness shift
        factor = self.rng.uniform(0.92, 1.08)
        patch_arr = (np.array(patch).astype(float) * factor).clip(0, 255).astype(np.uint8)
        patch = Image.fromarray(patch_arr)

        # destination (ensure no overlap with source)
        for _ in range(20):
            dx = self.rng.randint(0, W - patch_w - 1)
            dy = self.rng.randint(0, H - patch_h - 1)
            # check non-overlapping
            if abs(dx - sx) > patch_w or abs(dy - sy) > patch_h:
                break

        img_copy = img.copy()
        img_copy.paste(patch, (dx, dy))

        mask = Image.new("L", (W, H), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([dx, dy, dx + patch_w, dy + patch_h], fill=255)

        meta = {
            "forge_type": ForgeType.COPY_MOVE.value,
            "source_region": [sx, sy, sx + patch_w, sy + patch_h],
            "dest_region": [dx, dy, dx + patch_w, dy + patch_h],
            "angle": angle,
        }
        return img_copy, mask, meta

    def _noise_inject(
        self, img: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """
        Add JPEG compression artefacts + Gaussian noise in a local region,
        simulating low-quality pasted content.
        """
        W, H = img.size
        rx1 = self.rng.randint(W // 6, W // 2)
        ry1 = self.rng.randint(H // 6, H // 2)
        rx2 = rx1 + self.rng.randint(60, W // 3)
        ry2 = ry1 + self.rng.randint(60, H // 3)
        rx2, ry2 = min(rx2, W - 1), min(ry2, H - 1)

        region = img.crop((rx1, ry1, rx2, ry2))

        # JPEG re-compression artefact
        quality = self.rng.randint(10, 40)
        buf = io.BytesIO()
        region.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        region_compressed = Image.open(buf).convert("RGB")
        region_compressed = region_compressed.resize(region.size, Image.LANCZOS)

        # Gaussian noise
        arr = np.array(region_compressed).astype(float)
        noise_std = self.rng.uniform(5, 20)
        noise = self.np_rng.normal(0, noise_std, arr.shape)
        arr = (arr + noise).clip(0, 255).astype(np.uint8)
        region_noisy = Image.fromarray(arr)

        img_copy = img.copy()
        img_copy.paste(region_noisy, (rx1, ry1))

        mask = Image.new("L", (W, H), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([rx1, ry1, rx2, ry2], fill=255)

        meta = {
            "forge_type": ForgeType.NOISE_INJECT.value,
            "region": [rx1, ry1, rx2, ry2],
            "jpeg_quality": quality,
            "noise_std": noise_std,
        }
        return img_copy, mask, meta

    # ── helpers ───────────────────────────────────────────────────────────────

    def _random_text(self, width_hint: int) -> str:
        """Generate a plausible-looking replacement string (digits/amounts)."""
        n_chars = max(3, width_hint // 10)
        choices = [
            lambda: str(self.rng.randint(1, 999)) + "." + str(self.rng.randint(0, 99)).zfill(2),
            lambda: "".join(str(self.rng.randint(0, 9)) for _ in range(self.rng.randint(4, 8))),
            lambda: self.rng.choice(["TOTAL", "AMOUNT", "NET", "TAX", "PAID"]) + " " + str(self.rng.randint(10, 9999)),
        ]
        return self.rng.choice(choices)()


def run_forge_pipeline(
    raw_dir: str,
    processed_dir: str,
    doc_types: List[str],
    seed: int = 42,
    force: bool = False,
) -> Dict:
    """
    For every real image in raw_dir/{doc_type}/,
    generate one forged counterpart and save both to processed_dir.
    """
    generator = ForgeryGenerator(seed=seed)
    stats = {}

    for doc_type in doc_types:
        # Determine source directory
        if doc_type == "cord":
            src_dirs = {
                "train": os.path.join(raw_dir, "cord", "train"),
                "test": os.path.join(raw_dir, "cord", "test"),
                "val": os.path.join(raw_dir, "cord", "validation"),
            }
        else:
            src_dirs = {"all": os.path.join(raw_dir, "rvl_cdip", doc_type)}

        doc_stats = {}

        for split, src_dir in src_dirs.items():
            if not os.path.isdir(src_dir):
                print(f"  [forge] Source dir not found: {src_dir}, skipping.")
                continue

            real_out = os.path.join(processed_dir, doc_type, "real", split)
            forged_out = os.path.join(processed_dir, doc_type, "forged", split)
            os.makedirs(real_out, exist_ok=True)
            os.makedirs(forged_out, exist_ok=True)

            done_file = os.path.join(forged_out, ".done")
            if os.path.exists(done_file) and not force:
                n = len([f for f in os.listdir(forged_out) if f.endswith(".png") and "mask" not in f])
                print(f"  [forge] {doc_type}/{split}: already processed ({n} forged), skipping.")
                doc_stats[split] = n
                continue

            src_files = sorted([
                f for f in os.listdir(src_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ])

            n_saved = 0
            for fname in src_files:
                src_path = os.path.join(src_dir, fname)
                stem = Path(fname).stem

                real_dest = os.path.join(real_out, f"{stem}.png")
                forged_dest = os.path.join(forged_out, f"{stem}.png")
                mask_dest = os.path.join(forged_out, f"{stem}_mask.png")
                meta_dest = os.path.join(forged_out, f"{stem}_meta.json")

                try:
                    img = Image.open(src_path).convert("RGB")
                except Exception as e:
                    print(f"  [forge] Failed to open {src_path}: {e}")
                    continue

                # save real
                if not os.path.exists(real_dest):
                    img.save(real_dest)

                # generate and save forged
                if not os.path.exists(forged_dest):
                    try:
                        forged, mask, meta = generator.generate(img)
                        forged.save(forged_dest)
                        mask.save(mask_dest)
                        with open(meta_dest, "w") as f:
                            json.dump(meta, f)
                    except Exception as e:
                        print(f"  [forge] Failed to forge {fname}: {e}")
                        continue

                n_saved += 1

            Path(done_file).touch()
            print(f"  [forge] {doc_type}/{split}: {n_saved} real+forged pairs saved.")
            doc_stats[split] = n_saved

        stats[doc_type] = doc_stats

    return stats


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    raw_dir = os.path.join(args.project_root, "data", "raw")
    processed_dir = os.path.join(args.project_root, "data", "processed")

    all_types = ["cord", "letter", "form", "invoice", "budget", "memo",
                 "resume", "advertisement", "scientific_report"]
    stats = run_forge_pipeline(raw_dir, processed_dir, all_types, force=args.force)
    print("\nForgery generation complete.")
    print(json.dumps(stats, indent=2))