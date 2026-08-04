#!/usr/bin/env python3
"""Validate JSONL samples, images, masks, references, and pattern splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


O2MAG_TYPES = {"stain", "ink_smear"}
SUPPORTED_TYPES = {"nozzle_line", "banding", "white_spot", "speckle", *O2MAG_TYPES}


def load_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
        record["_line"] = line_number
        records.append(record)
    return records


def check_image(path_value: str, label: str, line: int, grayscale: bool = False) -> list[str]:
    errors = []
    path = Path(path_value).expanduser()
    if not path.is_file():
        return [f"Line {line}: missing {label}: {path}"]
    try:
        with Image.open(path) as image:
            image.verify()
        if grayscale:
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L"))
            if not np.any(mask > 127):
                errors.append(f"Line {line}: {label} is empty: {path}")
    except Exception as exc:
        errors.append(f"Line {line}: cannot read {label} {path}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    ids: list[str] = []
    split_patterns: dict[str, set[str]] = defaultdict(set)
    defect_counts: Counter[str] = Counter()

    for manifest in args.manifests:
        split_name = manifest.stem
        try:
            records = load_records(manifest)
        except Exception as exc:
            errors.append(f"{manifest}: {exc}")
            continue
        print(f"{manifest}: {len(records)} sample(s)")
        for record in records:
            line = record["_line"]
            for key in ("sample_id", "pattern_id", "normal_path", "defect_type"):
                if not record.get(key):
                    errors.append(f"{manifest} line {line}: missing {key}")
            if not all(record.get(key) for key in ("sample_id", "pattern_id", "defect_type")):
                continue
            ids.append(record["sample_id"])
            split = record.get("split", split_name)
            split_patterns[split].add(record["pattern_id"])
            defect_type = record["defect_type"]
            defect_counts[defect_type] += 1
            if defect_type not in SUPPORTED_TYPES:
                errors.append(f"{manifest} line {line}: unsupported defect_type {defect_type}")
            if record.get("normal_path"):
                errors.extend(check_image(record["normal_path"], "normal image", line))
            if defect_type in O2MAG_TYPES:
                for key, grayscale in (("reference_path", False), ("reference_mask_path", True)):
                    if not record.get(key):
                        errors.append(f"{manifest} line {line}: {defect_type} requires {key}")
                    else:
                        errors.extend(check_image(record[key], key, line, grayscale))

    duplicates = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate sample_id values: {duplicates[:20]}")
    split_names = sorted(split_patterns)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_patterns[left] & split_patterns[right]
            if overlap:
                errors.append(f"Pattern leakage between {left} and {right}: {sorted(overlap)[:20]}")

    print("\nDefect counts:")
    for defect_type, count in sorted(defect_counts.items()):
        print(f"  {defect_type}: {count}")
    print("\nPattern counts:")
    for split, patterns in sorted(split_patterns.items()):
        print(f"  {split}: {len(patterns)}")
    if errors:
        print("\nFAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

