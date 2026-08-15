"""Image sizing helpers that mirror Anthropic's published resize contract, so that pixel coordinates returned by
Claude map 1:1 onto the images we keep (docs: build-with-claude/vision-coordinates)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

HIGH_RES = dict(max_edge=2576, max_tokens=4784)   # Claude 4.7+, Opus 5, Sonnet 5, Fable 5
STANDARD = dict(max_edge=1568, max_tokens=1568)   # Haiku 4.5 and older


def count_image_tokens(width: int, height: int) -> int:
    """Visual tokens consumed by an image: one token per 28x28 pixel patch."""
    return math.ceil(width / 28) * math.ceil(height / 28)


def resized_size(width: int, height: int, max_edge: int = 2576, max_tokens: int = 4784) -> tuple[int, int]:
    """Verbatim port of Anthropic's reference `resized_size` (defaults set to the high-res tier)."""

    def fits(w: int, h: int) -> bool:
        return (math.ceil(w / 28) * 28 <= max_edge and math.ceil(h / 28) * 28 <= max_edge
                and count_image_tokens(w, h) <= max_tokens)

    if fits(width, height):
        return (width, height)
    if height > width:
        resized_h, resized_w = resized_size(height, width, max_edge, max_tokens)
        return (resized_w, resized_h)
    aspect_ratio = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            lo = mid
        else:
            hi = mid
    return (lo, max(round(lo / aspect_ratio), 1))


@dataclass
class PreparedImage:
    image: Image.Image          # exactly what we send (RGB)
    scale: float                # sent_px / source_px (uniform)
    tokens: int
    source_size: tuple[int, int]

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        """Map a coordinate returned by Claude (in sent-image pixels) back to source-image pixels."""
        return x / self.scale, y / self.scale


def prepare_for_claude(img: Image.Image, tier: dict = HIGH_RES, min_long_edge: int = 1400,
                       max_upscale: float = 4.0) -> PreparedImage:
    """Resize an image so Claude sees exactly this (coordinates 1:1). Small rasters (e.g. 500 px figure PNGs) are
    upscaled with Lanczos so the model has enough pixels to localize (upscaling adds no information but measurably
    helps read-out), capped so we never exceed the tier limits."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    scale = 1.0
    long_edge = max(w, h)
    if long_edge < min_long_edge:
        scale = min(max_upscale, min_long_edge / long_edge)
        w2, h2 = round(w * scale), round(h * scale)
        rw, rh = resized_size(w2, h2, **tier)
        if (rw, rh) != (w2, h2):        # would be downscaled anyway → pick the size Claude would use
            w2, h2 = rw, rh
            scale = w2 / w
        img = img.resize((w2, h2), Image.LANCZOS)
    else:
        rw, rh = resized_size(w, h, **tier)
        if (rw, rh) != (w, h):
            scale = rw / w
            img = img.resize((rw, rh), Image.LANCZOS)
    return PreparedImage(image=img, scale=scale, tokens=count_image_tokens(*img.size), source_size=(w, h))
