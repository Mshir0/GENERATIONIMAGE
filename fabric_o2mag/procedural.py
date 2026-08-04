from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def render_procedural(
    image: Image.Image,
    mask: Image.Image,
    defect_type: str,
    rng: np.random.Generator,
    parameters: dict,
) -> Image.Image:
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    edge = float(parameters.get("edge_sigma", 0.8))
    if edge > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), edge)
    alpha = alpha[..., None]
    if defect_type in {"white_spot", "nozzle_line"}:
        strength = float(parameters.get("strength", rng.uniform(0.55, 0.9)))
        target = np.full_like(source, 255.0)
    elif defect_type == "banding":
        strength = float(parameters.get("strength", rng.uniform(0.08, 0.25)))
        target = np.clip(source * (1.0 - strength), 0, 255)
        strength = 1.0
    elif defect_type == "speckle":
        strength = float(parameters.get("strength", rng.uniform(0.45, 0.85)))
        color = np.asarray(parameters.get("color", [35, 35, 35]), dtype=np.float32)
        target = np.broadcast_to(color, source.shape)
    else:
        raise ValueError(f"No procedural renderer for {defect_type}")
    output = source * (1.0 - alpha * strength) + target * alpha * strength
    return Image.fromarray(np.round(output.clip(0, 255)).astype(np.uint8))

