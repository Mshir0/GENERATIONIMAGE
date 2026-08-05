from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class MaskResult:
    mask: np.ndarray
    parameters: dict[str, Any]


def _range_int(value: Any, rng: np.random.Generator) -> int:
    if isinstance(value, (list, tuple)):
        return int(rng.integers(int(value[0]), int(value[1]) + 1))
    return int(value)


def generate_process_mask(
    shape: tuple[int, int],
    defect_type: str,
    rng: np.random.Generator,
    params: dict[str, Any],
    placement_mask: np.ndarray | None = None,
) -> MaskResult:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    p = dict(params)
    direction = p.get("direction", "vertical")

    if defect_type == "nozzle_line":
        width = _range_int(p.get("width", [1, 4]), rng)
        count = _range_int(p.get("count", [1, 4]), rng)
        positions = []
        for _ in range(count):
            pos = int(rng.integers(0, w if direction == "vertical" else h))
            positions.append(pos)
            if direction == "vertical":
                cv2.line(mask, (pos, 0), (pos, h - 1), 255, width)
            else:
                cv2.line(mask, (0, pos), (w - 1, pos), 255, width)
        p.update(width=width, count=count, positions=positions)

    elif defect_type == "banding":
        width = _range_int(p.get("width", [2, 8]), rng)
        period = _range_int(p.get("period", [18, 60]), rng)
        phase = int(rng.integers(0, period))
        limit = h if direction == "horizontal" else w
        for pos in range(phase, limit, period):
            if direction == "horizontal":
                mask[pos : min(h, pos + width), :] = 255
            else:
                mask[:, pos : min(w, pos + width)] = 255
        p.update(width=width, period=period, phase=phase)

    elif defect_type in {"white_spot", "speckle"}:
        count = _range_int(p.get("count", [1, 8]), rng)
        radius_range = p.get("radius", [2, 10])
        spots = []
        for _ in range(count):
            rx = _range_int(radius_range, rng)
            ry = max(1, int(rx * rng.uniform(0.4, 1.4)))
            if placement_mask is not None:
                if placement_mask.shape[:2] != (h, w):
                    raise ValueError("placement_mask shape must match the target image")
                margin = int(p.get("placement", {}).get("edge_margin", 3))
                kernel_radius = max(rx, ry) + margin
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_radius * 2 + 1, kernel_radius * 2 + 1)
                )
                safe_region = cv2.erode((placement_mask > 0).astype(np.uint8), kernel)
                ys, xs = np.nonzero(safe_region)
                if not len(xs):
                    raise ValueError(
                        "No valid dark-region position can contain the requested white spot"
                    )
                selected = int(rng.integers(0, len(xs)))
                x, y = int(xs[selected]), int(ys[selected])
            else:
                x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(mask, (x, y), (rx, ry), angle, 0, 360, 255, -1)
            spots.append({"x": x, "y": y, "rx": rx, "ry": ry, "angle": angle})
        p.update(count=count, spots=spots)

    elif defect_type in {"stain", "ink_smear"}:
        sigma = float(p.get("sigma", rng.uniform(10, 30)))
        coverage = float(p.get("coverage", rng.uniform(0.005, 0.04)))
        noise = rng.random((h, w)).astype(np.float32)
        field = cv2.GaussianBlur(noise, (0, 0), sigma)
        threshold = float(np.quantile(field, 1.0 - coverage))
        mask[field >= threshold] = 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        p.update(sigma=sigma, coverage=coverage, threshold=threshold)
    else:
        raise ValueError(f"Unsupported defect type: {defect_type}")

    return MaskResult(mask=mask, parameters=p)


def build_placement_mask(image_rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Build a valid placement region from the target image."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("Expected an RGB target image")
    mode = config.get("mode", "anywhere")
    h, w = image_rgb.shape[:2]
    if mode == "anywhere":
        return np.full((h, w), 255, dtype=np.uint8)
    if mode != "dark_regions":
        raise ValueError(f"Unsupported placement mode: {mode}")

    rgb = image_rgb[..., :3].astype(np.uint8)
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    region = (luminance <= int(config.get("luminance_max", 90))).astype(np.uint8)
    close_kernel = int(config.get("close_kernel", 9))
    if close_kernel > 1:
        if close_kernel % 2 == 0:
            close_kernel += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)

    min_area = int(config.get("min_component_area", 120))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    filtered = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            filtered[labels == label] = 255
    if not filtered.any():
        raise ValueError(
            "No dark printed region was found; increase placement.luminance_max or provide a mask"
        )
    return filtered
