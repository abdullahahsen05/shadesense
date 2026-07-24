from io import BytesIO

from PIL import Image

from src.image_io import open_rgb_image


def _jpeg_with_orientation(orientation: int | None) -> BytesIO:
    image = Image.new("RGB", (8, 4), color=(120, 80, 40))
    buffer = BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    image.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)
    return buffer


def test_exif_orientation_five_is_transposed_to_portrait():
    loaded = open_rgb_image(_jpeg_with_orientation(5))

    assert loaded.mode == "RGB"
    assert loaded.size == (4, 8)


def test_image_without_orientation_preserves_dimensions():
    loaded = open_rgb_image(_jpeg_with_orientation(None))

    assert loaded.mode == "RGB"
    assert loaded.size == (8, 4)
