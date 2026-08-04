import numpy as np

from fabric_o2mag.masks import generate_process_mask


def test_masks_are_reproducible_and_nonempty():
    for defect in ("nozzle_line", "banding", "white_spot", "speckle", "stain"):
        first = generate_process_mask((256, 384), defect, np.random.default_rng(7), {})
        second = generate_process_mask((256, 384), defect, np.random.default_rng(7), {})
        assert first.mask.any()
        assert np.array_equal(first.mask, second.mask)

