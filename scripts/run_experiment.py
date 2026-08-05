#!/usr/bin/env python3
"""Run generation and optional quality evaluation without shell helper logic."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fabric_o2mag.config import load_config
from fabric_o2mag.evaluate import evaluate_record
from fabric_o2mag.manifest import read_manifest, write_jsonl
from fabric_o2mag.pipeline import FabricGenerationPipeline


EXPECTED_RUNTIME = {
    "diffusers": "0.29.2",
    "transformers": "4.26.1",
    "huggingface-hub": "0.23.4",
    "tokenizers": "0.13.3",
    "safetensors": "0.4.3",
}


def validate_runtime() -> list[str]:
    errors = []
    if sys.version_info[:2] != (3, 10):
        errors.append(
            f"Python 3.10 is required, but this interpreter is {sys.version.split()[0]}"
        )
    for distribution, expected in EXPECTED_RUNTIME.items():
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            errors.append(f"{distribution} is not installed (required: {expected})")
            continue
        if installed != expected:
            errors.append(f"{distribution}=={expected} is required, but {installed} is installed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--route", choices=("all", "procedural", "o2mag"), default="all")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    runtime_errors = validate_runtime()
    if runtime_errors:
        print("FAILED: incompatible O2MAG runtime")
        for error in runtime_errors:
            print(f"  - {error}")
        print("Create a clean Python 3.10 environment and install requirements.txt")
        return 3

    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    pipeline = FabricGenerationPipeline(config, args.output)
    allowed = None if args.route == "all" else {args.route}
    samples = read_manifest(args.manifest)
    configured_routes = Counter(
        config.get("routes", {}).get(sample.defect_type, "procedural") for sample in samples
    )
    print(f"Configured routes: {dict(configured_routes)}")
    if allowed and not any(route in allowed for route in configured_routes):
        print(
            f"FAILED: no manifest samples are configured for route {args.route!r}. "
            "Regenerate the remote config or choose the configured route."
        )
        return 2

    records, failures, skipped = [], [], []
    for sample in tqdm(samples, desc="Generating"):
        try:
            records.append(pipeline.generate(sample, allowed))
        except RuntimeError as exc:
            if "disabled by --routes" in str(exc):
                skipped.append(
                    {
                        "sample_id": sample.sample_id,
                        "defect_type": sample.defect_type,
                        "reason": str(exc),
                    }
                )
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
    (args.output / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    if args.evaluate and records:
        samples = [evaluate_record(record) for record in tqdm(records, desc="Evaluating")]
        numeric = [key for key in samples[0] if key not in {"sample_id", "defect_type"}]
        summary = {key: float(np.nanmean([row[key] for row in samples])) for key in numeric}
        (args.output / "metrics.json").write_text(
            json.dumps({"summary": summary, "samples": samples}, indent=2), encoding="utf-8"
        )
    print(f"Completed: {len(records)}")
    print(f"Failed: {len(failures)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Output: {args.output.resolve()}")
    if not records and skipped:
        print("FAILED: every sample was skipped by the route filter")
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
