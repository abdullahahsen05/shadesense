"""Image loading helpers for preserving camera orientation."""

from PIL import Image, ImageOps


def open_rgb_image(source) -> Image.Image:
    """Open an image, apply its EXIF orientation, and return RGB pixels."""
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
