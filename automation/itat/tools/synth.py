"""Synthetic ITAT captcha generator — training data for automation/itat/model.onnx.

This is development tooling only; the app never imports it. It reproduces the
real ``itat.gov.in/judicial/casestatus`` captcha, profiled from 60 harvested
live samples on 2026-11-23:

* ~ W x H canvas, pure white background;
* a dense tangle of thin LIGHT-BLUE / CYAN bezier scribbles over most of the
  image (a few thicker/lighter strands);
* tiny coloured speckle dots (pink, green, blue-grey) scattered over the canvas;
* a faint light-blue rectangular border on many samples;
* CAPTCHA_LEN dark navy/black glyphs, slightly slanted/italic, variable size,
  drawn ON TOP of the scribbles.

That is a very different style from the GST grey-grid + red-line captcha, so
this generator is its own implementation rather than a tuning of the GST one.
Character set, length and canvas size come from ``charset``.

Rendering is randomised (font, size, slant, overlap, glyph colour jitter,
scribble density, speckle count) so the model generalises instead of memorising
one harvest batch.

API: ``gen_one(rng, style=None) -> (PIL.Image WxH, "A1B2C3")``.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from charset import CAPTCHA_LEN, CHARSET, W, H

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Oblique.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-ExtraBold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arialbi.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
)


def _available_fonts() -> list:
    found = []
    for path in _FONT_CANDIDATES:
        try:
            ImageFont.truetype(path, 20)
        except Exception:
            continue
        found.append(path)
    return found or list(_FONT_CANDIDATES[:1])


FONT_PATHS = _available_fonts()
_FONT_CACHE: dict = {}


def _font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def _blue_scribbles(rng, im, draw):
    """Dense tangle of thin light-blue bezier-like strands over most of the canvas."""
    n_lines = int(rng.integers(120, 190))
    for _ in range(n_lines):
        # light cyan-blue, slightly varying brightness/saturation
        r = int(rng.integers(150, 200))
        g = int(rng.integers(195, 230))
        b = int(rng.integers(225, 250))
        width = int(rng.choice([1, 1, 2]))
        x0 = rng.uniform(-10, W + 10)
        y0 = rng.uniform(-6, H + 6)
        # each strand is a short random polyline / arc
        segs = int(rng.integers(2, 7))
        pts = [(x0, y0)]
        x, y = x0, y0
        ang = rng.uniform(0, 2 * np.pi)
        for _s in range(segs):
            ang += rng.uniform(-1.4, 1.4)
            step = rng.uniform(12, W * 0.7)
            x += np.cos(ang) * step
            y += np.sin(ang) * step
            pts.append((x, y))
        draw.line(pts, fill=(r, g, b), width=width, joint="curve")


def _speckle(rng, draw):
    """Tiny coloured dots scattered over the canvas (pink/green/blue-grey)."""
    n = int(rng.integers(120, 260))
    palette = ((220, 90, 140), (120, 190, 120), (150, 170, 205),
               (200, 160, 210), (110, 160, 200))
    for _ in range(n):
        x = rng.uniform(0, W)
        y = rng.uniform(0, H)
        c = palette[int(rng.integers(len(palette)))]
        v = int(rng.integers(0, 40))
        draw.point((x, y), fill=(max(0, c[0] - v), max(0, c[1] - v), max(0, c[2] - v)))


def _border(rng, draw):
    if rng.random() < 0.7:
        c = (int(rng.integers(150, 190)), int(rng.integers(180, 215)),
             int(rng.integers(215, 245)))
        draw.rectangle([1, 1, W - 2, H - 2], outline=c, width=1)


def _glyph_layer(rng, code):
    """Render the dark glyphs on a transparent layer: small, thin, slanted, and
    clustered toward the left/middle rather than spread edge to edge (matches
    the live samples, which leave slack on the right)."""
    path = FONT_PATHS[int(rng.integers(len(FONT_PATHS)))]
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    base_h = rng.uniform(H * 0.38, H * 0.58)
    font = _font(path, 90)
    pad = 24
    size = (W + 2 * pad, H + 2 * pad)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    # glyphs occupy roughly the left 72-88% of the width, with uneven spacing
    span = W * rng.uniform(0.72, 0.88)
    x_start = pad + rng.uniform(3, 12)
    gaps = rng.uniform(0.6, 1.4, CAPTCHA_LEN)
    gaps = gaps / gaps.sum()
    x_cursor = 0.0
    for i, ch in enumerate(code):
        box = probe.textbbox((0, 0), ch, font=font)
        gw, gh = box[2] - box[0], box[3] - box[1]
        tmp = Image.new("RGBA", (gw + 10, gh + 10), (0, 0, 0, 0))
        # dark navy glyph with slight colour jitter
        v = int(rng.integers(22, 48))
        gcol = (v, v + int(rng.integers(2, 10)), v + int(rng.integers(4, 16)), 255)
        ImageDraw.Draw(tmp).text((5 - box[0], 5 - box[1]), ch, font=font, fill=gcol)
        tmp = tmp.crop(tmp.getbbox())
        scale_h = base_h * rng.uniform(0.82, 1.12)
        asp = gw / max(1, gh)
        w = max(2, int(round(scale_h * asp * rng.uniform(0.8, 1.0))))
        tmp = tmp.resize((w, max(2, int(round(scale_h)))), Image.LANCZOS)
        # thinner strokes sometimes (live glyphs are fairly thin)
        if rng.random() < 0.35:
            tmp = tmp.filter(ImageFilter.MinFilter(3))
        # stronger slant (italic-ish)
        shear = rng.uniform(-0.28, 0.28)
        tmp = tmp.transform(tmp.size, Image.AFFINE,
                            (1, shear, -shear * tmp.height / 2, 0, 1, 0),
                            resample=Image.BICUBIC)
        rot = rng.uniform(-10, 10)
        tmp = tmp.rotate(rot, resample=Image.BICUBIC, expand=True)
        cx = x_start + span * (x_cursor + gaps[i] / 2)
        px = int(round(cx - tmp.width / 2))
        py = int(round(pad + H / 2 - tmp.height / 2 + rng.uniform(-4, 4)))
        layer.alpha_composite(tmp, (max(-pad, px), max(-pad, py)))
        x_cursor += gaps[i]
    return layer.crop((pad, pad, pad + W, pad + H))


def gen_one(rng, style=None):
    code = "".join(CHARSET[int(i)] for i in rng.integers(0, len(CHARSET), CAPTCHA_LEN))
    im = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    _blue_scribbles(rng, im, draw)
    _speckle(rng, draw)
    _border(rng, draw)
    layer = _glyph_layer(rng, code)
    # a soft shadow/anti-alias pass so glyphs sit naturally over the scribbles
    if rng.random() < 0.5:
        shadow = layer.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.2)))
        shadow.putalpha(shadow.getchannel("A").point(lambda a: int(a * 0.25)))
        im = Image.alpha_composite(im.convert("RGBA"), shadow).convert("RGB")
    im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")
    # final light speckle overlay (a few dots drawn over the glyphs too, as live)
    d2 = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(10, 40))):
        x = rng.uniform(0, W)
        y = rng.uniform(0, H)
        d2.point((x, y), fill=(int(rng.integers(160, 220)), int(rng.integers(190, 230)),
                               int(rng.integers(210, 245))))
    return im, code


if __name__ == "__main__":
    import csv
    import os
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "synth_out"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        image, code = gen_one(rng)
        name = f"syn_{i:05d}.png"
        image.save(os.path.join(out_dir, name))
        rows.append([name, code])
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)
    print(f"wrote {n} images to {out_dir}")
