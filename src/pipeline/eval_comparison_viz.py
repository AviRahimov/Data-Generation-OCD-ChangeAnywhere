"""Multi-column detection overlays for :mod:`scripts.eval_detection_modes`.

Keeps color constants and drawing helpers in one place so the eval script focuses on
metrics and I/O. Each column uses the same downscaled ``before`` for fair comparison.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Column colors (match prior two-way eval: text=red, auto=blue)
COLOR_TEXT = (255, 70, 70)
COLOR_AUTO = (70, 140, 255)
COLOR_SAM2 = (80, 220, 120)  # green for SAM2.1 auto masks
PANEL_TITLE_H = 56


def font(size: int = 24) -> ImageFont.ImageFont:
    for p in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def downscale_for_vis(
        pil_image: Image.Image, target_width: int = 1200,
) -> tuple[Image.Image, float]:
    w, h = pil_image.size
    if w <= target_width:
        return pil_image, 1.0
    scale = target_width / w
    return pil_image.resize(
        (target_width, int(h * scale)), Image.LANCZOS), scale


def overlay_detections(
        base_pil: Image.Image,
        dets: list,
        color: tuple,
        vis_scale: float,
) -> Image.Image:
    """Draw mask fills, bbox outlines, and labels (det coords = full before scale)."""
    out = base_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    f = font(20)

    color_fill = (*color, 90)
    color_line = (*color, 255)

    for i, d in enumerate(dets, 1):
        mask = d.get("mask_fullimg")
        if mask is not None:
            mask_small = np.array(
                Image.fromarray((mask.astype(np.uint8) * 255)
                                ).resize(out.size, Image.NEAREST)
            )
            alpha = np.zeros(out.size[::-1] + (4,), dtype=np.uint8)
            fill = np.array(color_fill, dtype=np.uint8)
            alpha[mask_small > 127] = fill
            overlay = Image.alpha_composite(overlay, Image.fromarray(alpha))
            draw = ImageDraw.Draw(overlay)

        bx1, by1, bx2, by2 = d["bbox_fullimg"]
        sx1, sy1 = int(bx1 * vis_scale), int(by1 * vis_scale)
        sx2, sy2 = int(bx2 * vis_scale), int(by2 * vis_scale)
        draw.rectangle([sx1, sy1, sx2, sy2], outline=color_line, width=3)
        label = f"{i}. {d.get('label', '?')} {d.get('score', 0):.2f}"
        tw = (draw.textlength(label, font=f)
              if hasattr(draw, "textlength")
              else (draw.textbbox((0, 0), label, font=f)[2]
                    - draw.textbbox((0, 0), label, font=f)[0]))
        draw.rectangle(
            [sx1, max(sy1 - 24, 0), sx1 + tw + 8, max(sy1, 24)],
            fill=color_line,
        )
        draw.text((sx1 + 4, max(sy1 - 22, 2)), label, fill=(255, 255, 255), font=f)

    return Image.alpha_composite(out, overlay).convert("RGB")


def build_multi_column_panel(
        before_pil: Image.Image,
        columns: list[tuple[str, list, tuple]],
        pair_id: str,
        vis_scale: float,
) -> Image.Image:
    """``columns`` = list of ``(title_suffix, dets, color)``, e.g. (\"text (3)\", dets, COLOR_TEXT).

    Titles are combined with the detection count from ``len(dets)``.
    """
    title_f = font(30)
    small_f = font(22)

    rendered = [overlay_detections(before_pil, dets, col, vis_scale)
                for _tit, dets, col in columns]

    if not rendered:
        raise ValueError("columns is empty")
    pw, ph = rendered[0].size
    gap = 12
    n = len(rendered)
    out_w = pw * n + gap * (n - 1)
    out_h = ph + PANEL_TITLE_H + 36
    canvas = Image.new("RGB", (out_w, out_h), (25, 25, 25))
    dr = ImageDraw.Draw(canvas)
    dr.text((16, 12), f"{pair_id}", fill=(255, 255, 255), font=title_f)

    x = 0
    for i, (title_part, dets, color) in enumerate(columns):
        sub = f"{title_part} ({len(dets)} objs)"
        dr.text(
            (16 + x, PANEL_TITLE_H - 8), sub, fill=color, font=small_f,
        )
        canvas.paste(rendered[i], (x, PANEL_TITLE_H + 24))
        x += pw + gap

    return canvas
