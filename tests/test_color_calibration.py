import numpy as np

from src.color_calibration import (
    apply_neutral_card_calibration,
    estimate_neutral_card_calibration,
)


def test_neutral_card_removes_known_channel_cast():
    reference = np.full((100, 100, 3), (150, 120, 100), dtype=np.uint8)
    calibration = estimate_neutral_card_calibration(reference)
    face = np.full((20, 20, 3), (150, 120, 100), dtype=np.uint8)

    corrected = apply_neutral_card_calibration(face, calibration)

    assert calibration.success
    assert calibration.confidence > 0.9
    assert np.ptp(corrected[0, 0].astype(int)) <= 1


def test_clipped_card_is_rejected_and_image_preserved():
    reference = np.full((100, 100, 3), 250, dtype=np.uint8)
    calibration = estimate_neutral_card_calibration(reference)
    image = np.full((10, 10, 3), 80, dtype=np.uint8)

    corrected = apply_neutral_card_calibration(image, calibration)

    assert not calibration.success
    assert np.array_equal(corrected, image)
