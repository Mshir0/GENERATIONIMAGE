#!/usr/bin/env python3
"""Validate the remote Python, CUDA, package, and model environment."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "Pillow": "PIL",
    "OpenCV": "cv2",
    "PyWavelets": "pywt",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
}

MODEL_ENTRIES = ("model_index.json", "scheduler", "text_encoder", "tokenizer", "unet", "vae")


def package_version(module) -> str:
    return str(getattr(module, "__version__", "installed"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="Stable Diffusion 1.5 directory")
    args = parser.parse_args()

    errors: list[str] = []
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    if sys.version_info[:2] != (3, 10):
        errors.append("Python 3.10 is required for the pinned O2MAG environment")

    imported = {}
    print("\nPackages:")
    for label, import_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            imported[import_name] = module
            print(f"  [OK] {label}: {package_version(module)}")
        except Exception as exc:
            errors.append(f"Cannot import {label}: {exc}")
            print(f"  [FAIL] {label}: {exc}")

    torch = imported.get("torch")
    print("\nCUDA:")
    if torch is None:
        print("  [FAIL] torch is unavailable")
    elif not torch.cuda.is_available():
        errors.append("PyTorch cannot access CUDA")
        print("  [FAIL] CUDA unavailable")
    else:
        print(f"  [OK] PyTorch CUDA: {torch.version.cuda}")
        print(f"  [OK] GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / 1024**3
            print(f"  [OK] cuda:{index}: {properties.name}, {memory_gib:.1f} GiB")

    if args.model_path:
        model_path = args.model_path.expanduser().resolve()
        print(f"\nModel: {model_path}")
        for entry in MODEL_ENTRIES:
            target = model_path / entry
            if target.exists():
                print(f"  [OK] {entry}")
            else:
                errors.append(f"Missing model entry: {target}")
                print(f"  [FAIL] {entry}")

    print("\nResult:")
    if errors:
        for error in errors:
            print(f"  - {error}")
        print(f"FAILED with {len(errors)} problem(s)")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

