import numpy as np

from fabric_o2mag.masks import generate_process_mask


def test_masks_are_reproducible_and_nonempty():
    for defect in ("nozzle_line", "banding", "white_spot", "speckle", "stain"):
        first = generate_process_mask((256, 384), defect, np.random.default_rng(7), {})
        second = generate_process_mask((256, 384), defect, np.random.default_rng(7), {})
        assert first.mask.any()
        assert np.array_equal(first.mask, second.mask)


def test_white_spot_stays_inside_placement_region():
    placement = np.zeros((128, 128), dtype=np.uint8)
    placement[30:100, 40:110] = 255
    result = generate_process_mask(
        (128, 128),
        "white_spot",
        np.random.default_rng(11),
        {"count": 1, "radius": [4, 7], "placement": {"edge_margin": 3}},
        placement_mask=placement,
    )
    assert np.all(placement[result.mask > 0] > 0)
