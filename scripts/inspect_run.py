#!/usr/bin/env python3
"""Check a generated run and print a concise experiment summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    manifest = args.run_dir / "manifest.jsonl"
    if not manifest.is_file():
        print(f"FAILED: missing {manifest}")
        return 1
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    errors = []
    shapes: Counter[tuple[int, int]] = Counter()
    routes: Counter[str] = Counter()
    defects: Counter[str] = Counter()
    for record in records:
        routes[record["route"]] += 1
        defects[record["defect_type"]] += 1
        for key in ("generated_path", "mask_path", "original_path", "metadata_path"):
            if not Path(record[key]).is_file():
                errors.append(f"{record['sample_id']}: missing {key}: {record[key]}")
        if Path(record["generated_path"]).is_file():
            with Image.open(record["generated_path"]) as image:
                shapes[image.size] += 1
    print(f"Run: {args.run_dir.resolve()}")
    print(f"Samples: {len(records)}")
    print(f"Routes: {dict(routes)}")
    print(f"Defects: {dict(defects)}")
    print(f"Image sizes: {dict(shapes)}")
    metrics = args.run_dir / "metrics.json"
    if metrics.is_file():
        summary = json.loads(metrics.read_text(encoding="utf-8")).get("summary", {})
        print("Metrics:")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}")
    if errors:
        print("FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

