import numpy as np
from PIL import Image

from fabric_o2mag.fusion import strict_composite, wavelet_fuse


def test_background_is_unchanged():
    normal_np = np.full((64, 64, 3), 80, dtype=np.uint8)
    generated_np = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask_np = np.zeros((64, 64), dtype=np.uint8)
    mask_np[24:40, 24:40] = 255
    normal, generated, mask = map(Image.fromarray, (normal_np, generated_np, mask_np))
    strict = np.asarray(strict_composite(normal, generated, mask))
    assert np.array_equal(strict[mask_np == 0], normal_np[mask_np == 0])
    fused = np.asarray(wavelet_fuse(normal, generated, mask, 0.8, feather_radius=0))
    assert np.array_equal(fused[mask_np == 0], normal_np[mask_np == 0])

