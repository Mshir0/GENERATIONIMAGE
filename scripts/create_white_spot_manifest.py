#!/usr/bin/env python3
"""Create repeatable white-spot tasks at random positions for normal fabric images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--seed", type=int, default=202600)
    parser.add_argument(
        "--pattern-id",
        help="Use one pattern id for all images; otherwise each image's parent directory is used",
    )
    args = parser.parse_args()

    for path, label in (
        (args.normal_dir, "normal directory"),
        (args.reference, "reference image"),
        (args.reference_mask, "reference mask"),
    ):
        if not path.exists():
            print(f"FAILED: missing {label}: {path}")
            return 1
    if args.copies < 1:
        print("FAILED: --copies must be at least 1")
        return 1

    images = sorted(
        path for path in args.normal_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"FAILED: no images found under {args.normal_dir}")
        return 1

    records = []
    for image_index, image in enumerate(images):
        pattern_id = safe_id(args.pattern_id or image.parent.name)
        image_id = safe_id(image.stem)
        for copy_index in range(args.copies):
            seed = args.seed + image_index * args.copies + copy_index
            records.append(
                {
                    "sample_id": f"{pattern_id}_{image_id}_white_spot_{copy_index:03d}",
                    "pattern_id": pattern_id,
                    "normal_path": str(image.resolve()),
                    "reference_path": str(args.reference.resolve()),
                    "reference_mask_path": str(args.reference_mask.resolve()),
                    "defect_type": "white_spot",
                    "split": "train",
                    "seed": seed,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"PASSED: wrote {len(records)} tasks from {len(images)} normal image(s)")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

