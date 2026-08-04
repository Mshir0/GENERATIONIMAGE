from __future__ import annotations

import cv2
import numpy as np
import pywt
from PIL import Image


def soft_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    value = binary.astype(np.float32)
    if radius > 0:
        # Feather inward so pixels outside the annotation remain bit-exact.
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        value = np.clip(distance / float(radius), 0.0, 1.0)
    return value[..., None]


def wavelet_fuse(
    normal: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    lambda_high: float,
    wavelet: str = "db2",
    levels: int = 2,
    feather_radius: int = 5,
) -> Image.Image:
    normal_np = np.asarray(normal.convert("RGB"), dtype=np.float32) / 255.0
    generated_np = np.asarray(generated.convert("RGB"), dtype=np.float32) / 255.0
    if normal_np.shape != generated_np.shape:
        raise ValueError("normal and generated images must have the same shape")
    fused_channels = []
    for channel in range(3):
        n_coeffs = pywt.wavedec2(normal_np[..., channel], wavelet, level=levels)
        g_coeffs = pywt.wavedec2(generated_np[..., channel], wavelet, level=levels)
        coeffs = [g_coeffs[0]]
        for n_detail, g_detail in zip(n_coeffs[1:], g_coeffs[1:]):
            coeffs.append(tuple(lambda_high * n + (1.0 - lambda_high) * g
                                for n, g in zip(n_detail, g_detail)))
        reconstructed = pywt.waverec2(coeffs, wavelet)
        fused_channels.append(reconstructed[: normal_np.shape[0], : normal_np.shape[1]])
    fused = np.stack(fused_channels, axis=-1).clip(0, 1)
    alpha = soft_mask(np.asarray(mask.convert("L")), feather_radius)
    output = normal_np * (1.0 - alpha) + fused * alpha
    return Image.fromarray(np.round(output.clip(0, 1) * 255).astype(np.uint8))


def strict_composite(normal: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    n = np.asarray(normal.convert("RGB"))
    g = np.asarray(generated.convert("RGB"))
    m = np.asarray(mask.convert("L")) > 0
    out = n.copy()
    out[m] = g[m]
    return Image.fromarray(out)
