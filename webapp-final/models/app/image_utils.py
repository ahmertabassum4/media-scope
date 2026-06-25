import base64
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = None

TARGET_ASPECT_RATIO = 16 / 9
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ImageValidationError(ValueError):
    pass


def load_image_from_bytes(payload: bytes, filename: str | None = None) -> tuple[Image.Image, str]:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ImageValidationError(f"Unsupported image extension. Supported: {supported}")

    try:
        with Image.open(io.BytesIO(payload)) as image:
            source_format = image.format or "unknown"
            return normalize_rgb(image), source_format.lower()
    except UnidentifiedImageError as exc:
        raise ImageValidationError("Uploaded file is not a readable image") from exc


def normalize_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if image.mode in {"RGBA", "LA"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def crop_to_first_screen(image: Image.Image) -> tuple[Image.Image, dict[str, int | float]]:
    width, height = image.size
    target_height = min(height, round(width / TARGET_ASPECT_RATIO))

    if target_height < height:
        left = 0
        top = 0
        right = width
        bottom = target_height
    else:
        target_width = min(width, round(height * TARGET_ASPECT_RATIO))
        left = (width - target_width) // 2
        top = 0
        right = left + target_width
        bottom = height

    crop = image.crop((left, top, right, bottom))
    return crop, {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": crop.width,
        "height": crop.height,
        "aspect_ratio": round(crop.width / crop.height, 6),
    }


def resize_for_llm(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    new_height = round(image.height * max_width / image.width)
    return image.resize((max_width, new_height), Image.Resampling.LANCZOS)


def image_to_png_bytes(image: Image.Image, optimize: bool = True) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=optimize)
    return output.getvalue()


def image_to_data_url(image: Image.Image, max_width: int) -> str:
    resized = resize_for_llm(image, max_width=max_width)
    encoded = base64.b64encode(image_to_png_bytes(resized)).decode("ascii")
    return f"data:image/png;base64,{encoded}"
