from io import BytesIO

from PIL import Image, ImageCms

from src.image_io import open_rgb_image, open_rgb_image_with_metadata


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


def test_embedded_icc_profile_is_converted_to_srgb():
    image = Image.new("RGB", (8, 4), color=(120, 80, 40))
    profile = ImageCms.ImageCmsProfile(
        ImageCms.createProfile("sRGB")
    ).tobytes()
    buffer = BytesIO()
    image.save(buffer, format="JPEG", icc_profile=profile)
    buffer.seek(0)

    loaded, metadata = open_rgb_image_with_metadata(buffer)

    assert loaded.mode == "RGB"
    assert metadata.embedded_icc_present
    assert metadata.icc_converted_to_srgb
    assert not metadata.assumed_srgb
    assert "sRGB" in metadata.source_profile_description


def test_invalid_icc_profile_falls_back_with_warning():
    image = Image.new("RGB", (8, 4), color=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", icc_profile=b"not-a-profile")
    buffer.seek(0)

    loaded, metadata = open_rgb_image_with_metadata(buffer)

    assert loaded.mode == "RGB"
    assert metadata.embedded_icc_present
    assert not metadata.icc_converted_to_srgb
    assert metadata.assumed_srgb
    assert metadata.warnings
