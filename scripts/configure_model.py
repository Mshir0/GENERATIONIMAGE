#!/usr/bin/env python3
"""Locate a complete Hugging Face SD 1.5 snapshot and create a remote config."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT_CACHE_ROOT = Path(
    "/mnt/sda1/HuggingfaceDownload/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
)
REQUIRED_ENTRIES = ("model_index.json", "scheduler", "text_encoder", "tokenizer", "unet", "vae")


def merge_config(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_name = config.pop("base", None)
    if base_name:
        config = merge_config(load_config(path.parent / base_name), config)
    return config


def missing_entries(snapshot: Path) -> list[str]:
    return [entry for entry in REQUIRED_ENTRIES if not (snapshot / entry).exists()]


def referenced_snapshot(cache_root: Path) -> Path | None:
    main_ref = cache_root / "refs" / "main"
    if not main_ref.is_file():
        return None
    revision = main_ref.read_text(encoding="utf-8").strip()
    if not revision:
        return None
    candidate = cache_root / "snapshots" / revision
    return candidate if candidate.is_dir() else None


def find_snapshot(cache_root: Path) -> Path:
    cache_root = cache_root.expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    candidates: list[Path] = []
    preferred = referenced_snapshot(cache_root)
    if preferred:
        candidates.append(preferred)
    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.is_dir():
        others = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir() and path != preferred),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(others)

    failures = []
    for candidate in candidates:
        missing = missing_entries(candidate)
        if not missing:
            return candidate.resolve()
        failures.append(f"{candidate.name}: missing {', '.join(missing)}")
    details = "; ".join(failures) if failures else "no snapshot directories found"
    raise FileNotFoundError(f"No complete Diffusers snapshot under {cache_root}: {details}")


def validate_model_index(snapshot: Path) -> dict:
    path = snapshot / "model_index.json"
    try:
        model_index = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot parse {path}: {exc}") from exc
    pipeline_class = model_index.get("_class_name")
    if pipeline_class not in {"StableDiffusionPipeline", "FlaxStableDiffusionPipeline"}:
        raise ValueError(f"Unexpected pipeline class in model_index.json: {pipeline_class!r}")
    return model_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--base-config", type=Path, default=Path("configs/fabric_base.yaml"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/fabric_remote.yaml"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    try:
        snapshot = find_snapshot(args.cache_root)
        model_index = validate_model_index(snapshot)
        config = load_config(args.base_config)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    config["model_path"] = str(snapshot)
    config["device"] = args.device
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    print("PASSED")
    print(f"Repository cache: {args.cache_root.expanduser().resolve()}")
    print(f"Selected snapshot: {snapshot}")
    print(f"Pipeline class: {model_index.get('_class_name')}")
    print(f"Device: {args.device}")
    print(f"Config written: {args.output_config.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
