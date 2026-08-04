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
    shape: tuple[int, int], defect_type: str, rng: np.random.Generator, params: dict[str, Any]
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
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            rx = _range_int(radius_range, rng)
            ry = max(1, int(rx * rng.uniform(0.4, 1.4)))
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

