#!/usr/bin/env python3
"""Split a JSONL dataset by pattern_id without installing a Python package."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_by_pattern(records: list[dict], ratios: list[float], seed: int) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        pattern_id = record.get("pattern_id")
        if not pattern_id:
            raise ValueError(f"Record is missing pattern_id: {record.get('sample_id', '<unknown>')}")
        grouped[pattern_id].append(record)
    patterns = sorted(grouped)
    random.Random(seed).shuffle(patterns)
    train_end = round(len(patterns) * ratios[0])
    val_end = train_end + round(len(patterns) * ratios[1])
    assigned = {
        "train": patterns[:train_end],
        "val": patterns[train_end:val_end],
        "test": patterns[val_end:],
    }
    output = {}
    for split, pattern_ids in assigned.items():
        output[split] = []
        for pattern_id in pattern_ids:
            for source in grouped[pattern_id]:
                record = dict(source)
                record["split"] = split
                output[split].append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ratios", default="0.7,0.1,0.2")
    args = parser.parse_args()
    ratios = [float(value) for value in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        print("FAILED: --ratios must contain train,val,test values summing to 1")
        return 1
    try:
        splits = split_by_pattern(read_records(args.manifest), ratios, args.seed)
        for split, records in splits.items():
            write_records(args.output_dir / f"{split}.jsonl", records)
            patterns = {record["pattern_id"] for record in records}
            print(f"{split}: {len(records)} samples, {len(patterns)} patterns")
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

