from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .manifest import read_manifest, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split samples by pattern_id")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ratios", default="0.7,0.1,0.2")
    args = parser.parse_args()
    ratios = [float(value) for value in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("--ratios must contain train,val,test values summing to 1")
    grouped = defaultdict(list)
    for sample in read_manifest(args.manifest):
        grouped[sample.pattern_id].append(sample)
    patterns = sorted(grouped)
    random.Random(args.seed).shuffle(patterns)
    n_train = round(len(patterns) * ratios[0])
    n_val = round(len(patterns) * ratios[1])
    assignments = {
        "train": patterns[:n_train],
        "val": patterns[n_train : n_train + n_val],
        "test": patterns[n_train + n_val :],
    }
    output = Path(args.output_dir)
    for split, pattern_ids in assignments.items():
        records = []
        for pattern_id in pattern_ids:
            for sample in grouped[pattern_id]:
                record = sample.to_dict()
                record["split"] = split
                records.append(record)
        write_jsonl(output / f"{split}.jsonl", records)


if __name__ == "__main__":
    main()

