#!/usr/bin/env python3
"""Run generation and optional quality evaluation without shell helper logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from fabric_o2mag.config import load_config
from fabric_o2mag.evaluate import evaluate_record
from fabric_o2mag.manifest import read_manifest, write_jsonl
from fabric_o2mag.pipeline import FabricGenerationPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--route", choices=("all", "procedural", "o2mag"), default="all")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    pipeline = FabricGenerationPipeline(config, args.output)
    allowed = None if args.route == "all" else {args.route}
    records, failures = [], []
    for sample in tqdm(read_manifest(args.manifest), desc="Generating"):
        try:
            records.append(pipeline.generate(sample, allowed))
        except RuntimeError as exc:
            if "disabled by --routes" in str(exc):
                continue
            failures.append({"sample_id": sample.sample_id, "error": str(exc)})
            if not args.continue_on_error:
                raise
        except Exception as exc:
            failures.append({"sample_id": sample.sample_id, "error": repr(exc)})
            if not args.continue_on_error:
                raise

    write_jsonl(args.output / "manifest.jsonl", records)
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if args.evaluate and records:
        samples = [evaluate_record(record) for record in tqdm(records, desc="Evaluating")]
        numeric = [key for key in samples[0] if key not in {"sample_id", "defect_type"}]
        summary = {key: float(np.nanmean([row[key] for row in samples])) for key in numeric}
        (args.output / "metrics.json").write_text(
            json.dumps({"summary": summary, "samples": samples}, indent=2), encoding="utf-8"
        )
    print(f"Completed: {len(records)}")
    print(f"Failed: {len(failures)}")
    print(f"Output: {args.output.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

