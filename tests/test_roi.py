import numpy as np
from PIL import Image

from fabric_o2mag.roi import extract_roi, paste_roi, plan_roi


def test_roi_handles_border_and_roundtrip_size():
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[:3, :3] = 255
    plan = plan_roi(mask, model_size=512, min_defect_px=24)
    image = Image.fromarray(np.full((100, 160, 3), 127, dtype=np.uint8))
    roi = extract_roi(image, plan)
    assert roi.size == (512, 512)
    assert paste_roi(image, roi, plan).size == image.size

