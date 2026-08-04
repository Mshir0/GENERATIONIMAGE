from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from .config import load_config
from .manifest import read_manifest, write_jsonl
from .pipeline import FabricGenerationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate process-aware fabric anomalies")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--routes", choices=["all", "procedural", "o2mag"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    pipeline = FabricGenerationPipeline(config, output)
    allowed = None if args.routes == "all" else {args.routes}
    records = []
    for sample in tqdm(read_manifest(args.manifest), desc="Generating"):
        try:
            records.append(pipeline.generate(sample, allowed))
        except RuntimeError as exc:
            if "disabled by --routes" not in str(exc):
                raise
    write_jsonl(output / "manifest.jsonl", records)


if __name__ == "__main__":
    main()
