from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .schemas import ROIPlan


def estimate_thickness(mask: np.ndarray) -> float:
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        raise ValueError("ROI cannot be planned from an empty mask")
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    values = distance[distance > 0]
    return max(1.0, float(np.percentile(values, 75) * 2.0))


def plan_roi(
    mask: np.ndarray,
    model_size: int = 512,
    min_defect_px: int = 24,
    context_ratio: float = 3.0,
    min_crop_size: int = 64,
) -> ROIPlan:
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        raise ValueError("Target mask is empty")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    thickness = estimate_thickness(mask)
    bbox_side = max(x1 - x0, y1 - y0)
    context_side = max(min_crop_size, int(round(bbox_side * context_ratio)))
    scale_limited_side = max(min_crop_size, int(thickness * model_size / min_defect_px))
    side = min(max(h, w), max(bbox_side, min(context_side, scale_limited_side)))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = left + side, top + side
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - w), max(0, bottom - h)
    left, top, right, bottom = max(0, left), max(0, top), min(w, right), min(h, bottom)
    actual_side = max(right - left + pad_left + pad_right, bottom - top + pad_top + pad_bottom)
    return ROIPlan(
        xyxy=(left, top, right, bottom),
        padding=(pad_left, pad_top, pad_right, pad_bottom),
        source_size=(w, h),
        model_size=model_size,
        estimated_thickness=thickness,
        scale=model_size / float(actual_side),
    )


def extract_roi(image: Image.Image, plan: ROIPlan, is_mask: bool = False) -> Image.Image:
    crop = np.asarray(image.crop(plan.xyxy))
    pl, pt, pr, pb = plan.padding
    if any(plan.padding):
        border = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
        value = 0 if is_mask else None
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, border, value=value)
    interpolation = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    mode = "L" if is_mask else "RGB"
    return Image.fromarray(crop).convert(mode).resize((plan.model_size, plan.model_size), interpolation)


def paste_roi(base: Image.Image, generated: Image.Image, plan: ROIPlan) -> Image.Image:
    left, top, right, bottom = plan.xyxy
    pl, pt, pr, pb = plan.padding
    full_side = generated.width
    valid = generated.crop((pl * full_side // (right - left + pl + pr),
                            pt * full_side // (bottom - top + pt + pb),
                            full_side - pr * full_side // (right - left + pl + pr),
                            full_side - pb * full_side // (bottom - top + pt + pb)))
    valid = valid.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    result = base.copy()
    result.paste(valid, (left, top))
    return result

