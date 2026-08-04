from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pywt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _wavelet_distance(a: np.ndarray, b: np.ndarray) -> float:
    distances = []
    for channel in range(3):
        ac = pywt.wavedec2(a[..., channel], "db2", level=2)[1:]
        bc = pywt.wavedec2(b[..., channel], "db2", level=2)[1:]
        for ad, bd in zip(ac, bc):
            distances.extend(float(np.mean(np.abs(x - y))) for x, y in zip(ad, bd))
    return float(np.mean(distances))


def evaluate_record(record: dict) -> dict:
    normal = np.asarray(Image.open(record["original_path"]).convert("RGB"), dtype=np.float32) / 255
    generated = np.asarray(Image.open(record["generated_path"]).convert("RGB"), dtype=np.float32) / 255
    mask = np.asarray(Image.open(record["mask_path"]).convert("L")) > 0
    guard = cv2.dilate(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    background = ~guard
    if background.any():
        n_bg, g_bg = normal.copy(), generated.copy()
        g_bg[~background] = n_bg[~background]
        psnr = peak_signal_noise_ratio(n_bg, g_bg, data_range=1.0)
        ssim = structural_similarity(n_bg, g_bg, channel_axis=2, data_range=1.0)
        leakage = float(np.mean(np.abs(normal[background] - generated[background])))
    else:
        psnr, ssim, leakage = float("nan"), float("nan"), float("nan")
    changed = np.mean(np.abs(normal - generated), axis=2) > (10 / 255)
    intersection = np.logical_and(changed, mask).sum()
    union = np.logical_or(changed, mask).sum()
    return {
        "sample_id": record["sample_id"],
        "defect_type": record["defect_type"],
        "background_psnr": float(psnr),
        "background_ssim": float(ssim),
        "background_leakage_l1": leakage,
        "wavelet_high_frequency_distance": _wavelet_distance(normal, generated),
        "change_mask_iou": float(intersection / max(union, 1)),
        "mask_area_ratio": float(mask.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line]
    per_sample = [evaluate_record(record) for record in records]
    numeric = [key for key in per_sample[0] if key not in {"sample_id", "defect_type"}] if per_sample else []
    summary = {key: float(np.nanmean([row[key] for row in per_sample])) for key in numeric}
    Path(args.output).write_text(
        json.dumps({"summary": summary, "samples": per_sample}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

