"""Image loading helpers for camera orientation and embedded color profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO

from PIL import Image, ImageCms, ImageOps


@dataclass(frozen=True)
class ImageColorMetadata:
    embedded_icc_present: bool = False
    icc_converted_to_srgb: bool = False
    assumed_srgb: bool = True
    source_profile_description: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _profile_description(profile) -> str:
    try:
        return str(ImageCms.getProfileDescription(profile)).strip()
    except Exception:
        return "embedded ICC profile"


def open_rgb_image_with_metadata(source) -> tuple[Image.Image, ImageColorMetadata]:
    """Open an image, apply orientation, and convert embedded ICC data to sRGB."""
    with Image.open(source) as image:
        oriented = ImageOps.exif_transpose(image)
        icc_bytes = oriented.info.get("icc_profile") or image.info.get("icc_profile")
        if not icc_bytes:
            return (
                oriented.convert("RGB"),
                ImageColorMetadata(
                    embedded_icc_present=False,
                    icc_converted_to_srgb=False,
                    assumed_srgb=True,
                ),
            )
        try:
            source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_bytes))
            destination_profile = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(
                oriented,
                source_profile,
                destination_profile,
                renderingIntent=ImageCms.Intent.PERCEPTUAL,
                outputMode="RGB",
            )
            return (
                converted,
                ImageColorMetadata(
                    embedded_icc_present=True,
                    icc_converted_to_srgb=True,
                    assumed_srgb=False,
                    source_profile_description=_profile_description(
                        source_profile
                    ),
                ),
            )
        except Exception as exc:
            return (
                oriented.convert("RGB"),
                ImageColorMetadata(
                    embedded_icc_present=True,
                    icc_converted_to_srgb=False,
                    assumed_srgb=True,
                    source_profile_description="unreadable embedded ICC profile",
                    warnings=[
                        "The embedded camera color profile could not be converted "
                        f"to sRGB ({type(exc).__name__}); RGB values were preserved."
                    ],
                ),
            )


def open_rgb_image(source) -> Image.Image:
    """Backward-compatible RGB loader with orientation and ICC handling."""
    image, _ = open_rgb_image_with_metadata(source)
    return image
