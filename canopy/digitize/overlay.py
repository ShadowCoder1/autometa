"""Verification overlays: draw the marks a route resolved onto the figure crop.

The overlay is what a vision model is shown when asked "is mark 3 really on the bar top?", so every mark
is numbered and drawn small, crisp and high-contrast — the underlying figure must stay readable.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_COLOUR = "#e6194b"
_HALO = (255, 255, 255)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """DejaVu (shipped with matplotlib, already a dependency) at the requested size, else PIL's default."""
    try:
        import matplotlib
        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        if path.exists():
            return ImageFont.truetype(str(path), size)
    except Exception:                                   # pragma: no cover - font lookup is best effort
        pass
    try:
        return ImageFont.load_default(size=size)        # Pillow >= 10.1
    except TypeError:                                   # pragma: no cover - older Pillow
        return ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, colour: str) -> None:
    """Text with a white halo so it stays readable over ink."""
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        draw.text((x + dx, y + dy), text, font=font, fill=_HALO)
    draw.text((x, y), text, font=font, fill=colour)


def draw_overlay(crop_png: str | Path, marks: list[dict], out_png: str | Path) -> Path:
    """Draw numbered annotations on a copy of `crop_png` and save it to `out_png` (returns that path).

    Each mark is a dict in **crop pixels**::

        {"x": float, "y": float, "label": str, "color": "#rrggbb",
         "kind": "point" | "hline" | "vline" | "box" | "text",
         "x1": float, "y1": float}      # box only: opposite corner

    Marks are numbered 1..n in the order given; the number (and the label, when present) is written next to
    the mark with a white halo. The output image has exactly the same size as the input.
    """
    img = Image.open(crop_png).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(1, round(min(w, h) / 500))
    r = max(4, round(min(w, h) / 90))
    size = max(11, round(min(w, h) / 45))
    font = _font(size)
    for i, mark in enumerate(marks, start=1):
        colour = str(mark.get("color") or DEFAULT_COLOUR)
        kind = str(mark.get("kind") or "point")
        x = float(mark.get("x", 0.0))
        y = float(mark.get("y", 0.0))
        tag = str(i) if not mark.get("label") else f"{i}. {mark['label']}"
        if kind == "hline":
            draw.line([(0, y), (w, y)], fill=colour, width=lw)
            _text(draw, (4, y + lw + 2), tag, font, colour)
        elif kind == "vline":
            draw.line([(x, 0), (x, h)], fill=colour, width=lw)
            _text(draw, (x + lw + 2, 4), tag, font, colour)
        elif kind == "box":
            x1 = float(mark.get("x1", x + 2 * r))
            y1 = float(mark.get("y1", y + 2 * r))
            draw.rectangle([(min(x, x1), min(y, y1)), (max(x, x1), max(y, y1))], outline=colour, width=lw)
            _text(draw, (min(x, x1), min(y, y1) - size - 2), tag, font, colour)
        elif kind == "text":
            _text(draw, (x, y), tag, font, colour)
        else:                                            # point: crosshair + ring, centre left clear
            draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=colour, width=lw)
            draw.line([(x - 2 * r, y), (x - r, y)], fill=colour, width=lw)
            draw.line([(x + r, y), (x + 2 * r, y)], fill=colour, width=lw)
            draw.line([(x, y - 2 * r), (x, y - r)], fill=colour, width=lw)
            draw.line([(x, y + r), (x, y + 2 * r)], fill=colour, width=lw)
            _text(draw, (x + r + 3, y - size / 2), tag, font, colour)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)
    return out
