#!/usr/bin/env python3
"""Terminal-style GitHub profile banner ("profile.sh --live").

Turns your photo into a 1-bit dithered portrait and places it next to a
SYSTEM.INFO panel filled from config.json. Writes a dark and a light SVG.

    pip install pillow numpy
    python generate_banner.py --photo photo.png --config config.json --out assets
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

W, H = 1180, 610
GRID_W, GRID_H = 300, 340          # portrait lattice, drawn 1:1
PORTRAIT_X, PORTRAIT_Y = 94, 162   # top-left of the portrait inside the left panel
FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
MAX_ROWS = 16

THEMES = {
    "dark": {
        "bg": "#0A101F", "panel": "#0D1628", "panel2": "#101B30",
        "line": "#25344C", "muted": "#8291A8", "text": "#DDE7F5",
        "dots": "#AA9BEF", "chrome": "#22D3EE", "accent": "#10B981",
        "shadow": "#02050B",
    },
    "light": {
        "bg": "#F6F8FA", "panel": "#FFFFFF", "panel2": "#EDF3F7",
        "line": "#CBD7E1", "muted": "#64748B", "text": "#172033",
        "dots": "#4A3D7A", "chrome": "#0891B2", "accent": "#10B981",
        "shadow": "#AAB7C4",
    },
}

DEFAULT_CONFIG = {
    "window_title": "profile.sh --live",
    "map_label": "VISUAL.MAP",
    "info_label": "SYSTEM.INFO",
    "handle": "@username",
    "status_left": "ALL SYSTEMS NOMINAL",
    "status_right": "UTC+0 · MY NODE",
    "photo_crop": {"zoom": 1.0, "x": 0.5, "y": 0.5},
    "rows": [["Subject", "Your Name"]],
}


# ----------------------------------------------------------------- portrait --

def load_portrait(path: Path, zoom: float, cx: float, cy: float) -> Image.Image:
    """Crop the photo to the portrait aspect ratio and resize to the lattice."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    iw, ih = img.size
    aspect = GRID_W / GRID_H
    if iw / ih > aspect:
        bh = float(ih)
        bw = bh * aspect
    else:
        bw = float(iw)
        bh = bw / aspect
    bw /= max(zoom, 1.0)
    bh /= max(zoom, 1.0)
    left = min(max(cx * iw - bw / 2, 0), iw - bw)
    top = min(max(cy * ih - bh / 2, 0), ih - bh)
    box = (round(left), round(top), round(left + bw), round(top + bh))
    return img.crop(box).resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine Floyd-Steinberg dithering. True = lit pixel."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros(work.shape, dtype=bool)
    height, width = work.shape
    for y in range(height):
        forward = y % 2 == 0
        xs = range(width) if forward else range(width - 1, -1, -1)
        step = 1 if forward else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0
            err = old - new
            nx = x + step
            if 0 <= nx < width:
                work[y, nx] += err * 7 / 16
            if y + 1 < height:
                if 0 <= x - step < width:
                    work[y + 1, x - step] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < width:
                    work[y + 1, nx] += err * 1 / 16
    return out


def portrait_dots(img: Image.Image, theme: str) -> np.ndarray:
    """Boolean (GRID_H, GRID_W) array of the dots to draw for this theme."""
    alpha = np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0
    has_alpha = bool(alpha.min() < 0.99)

    if theme == "dark":
        gray = np.asarray(ImageOps.grayscale(img.convert("RGB")), dtype=np.float32)
        prepared = Image.fromarray(np.uint8(np.clip(gray * alpha, 0, 255)), "L")
        mask = None
        if has_alpha:
            mask = Image.fromarray(np.uint8((alpha > 0.08) * 255), "L")
        prepared = ImageOps.equalize(prepared, mask=mask)
    else:
        white = Image.new("RGBA", img.size, "white")
        white.alpha_composite(img)
        prepared = ImageOps.autocontrast(
            ImageOps.grayscale(white.convert("RGB")), cutoff=1
        )

    prepared = ImageEnhance.Contrast(prepared).enhance(1.35)
    prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2, percent=170, threshold=1))
    bits = floyd_steinberg(np.asarray(prepared))

    if theme == "dark":
        return bits & (alpha > 0.08)
    active = ~bits
    if has_alpha:
        active &= alpha > 0.08
    return active


def runs_path(active: np.ndarray, x0: int, y0: int) -> str:
    """Compact SVG path: one 1px-high horizontal stroke per run of dots."""
    parts: list[str] = []
    for y in range(active.shape[0]):
        row = active[y]
        width = row.size
        x = 0
        while x < width:
            if row[x]:
                start = x
                while x < width and row[x]:
                    x += 1
                parts.append(f"M{x0 + start} {y0 + y}.5h{x - start}")
            else:
                x += 1
    return "".join(parts)


# ---------------------------------------------------------------------- svg --

def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def tw(text: str, size: float) -> float:
    """Monospace width estimate (also passed as textLength so fonts can't drift)."""
    return len(text) * size * 0.605


def n(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{n(x)} {n(y)}h1" for x in np.arange(x1, x2, 5.0))


def render_svg(theme: str, dots: np.ndarray, cfg: dict) -> str:
    t = THEMES[theme]
    rows = cfg["rows"]
    handle = cfg["handle"]

    pill_w = tw(handle, 14) + 36
    pill_x = 1128 - pill_w
    live_x = pill_x - 55

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-label="{esc(cfg["window_title"])}">',
        f'<title>{esc(cfg["window_title"])}</title>',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        f'<clipPath id="portraitClip"><rect x="{PORTRAIT_X}" y="{PORTRAIT_Y}" '
        f'width="{GRID_W}" height="{GRID_H}"/></clipPath>',
        '<linearGradient id="scan" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{t["chrome"]}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{t["chrome"]}" stop-opacity=".28"/>'
        "</linearGradient>",
        "</defs>",
        # window
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        f'font-family="{FONT}" font-size="13" letter-spacing=".4">{esc(cfg["window_title"])}</text>',
        # left panel
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="{FONT}" font-size="13" '
        f'font-weight="700" letter-spacing="1.2">{esc(cfg["map_label"])}</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" font-family="{FONT}" '
        f'font-size="11">{GRID_W}×{GRID_H} / 1-BIT</text>',
        # corner brackets
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        # portrait + scan line
        '<g shape-rendering="crispEdges">'
        f'<path d="{runs_path(dots, PORTRAIT_X, PORTRAIT_Y)}" fill="none" '
        f'stroke="{t["dots"]}" stroke-width="1" opacity=".95"/></g>',
        '<g clip-path="url(#portraitClip)">'
        f'<rect x="{PORTRAIT_X}" y="{PORTRAIT_Y - 30}" width="{GRID_W}" height="30" fill="url(#scan)">'
        f'<animateTransform attributeName="transform" type="translate" '
        f'values="0 0;0 {GRID_H + 30}" dur="5s" repeatCount="indefinite"/></rect></g>',
        f'<text x="58" y="551" fill="{t["muted"]}" font-family="{FONT}" font-size="10">'
        f'PTS {int(dots.sum()):05d} · FS/SERPENTINE</text>',
        # right panel
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        f'<text x="490" y="111" fill="{t["chrome"]}" font-family="{FONT}" font-size="13" '
        f'font-weight="700" letter-spacing="1.2">{esc(cfg["info_label"])}</text>',
        f'<circle cx="{n(live_x - 12)}" cy="106" r="4" fill="#FF4D5A">'
        '<animate attributeName="opacity" values="1;.25;1" dur="1.6s" repeatCount="indefinite"/></circle>',
        f'<text x="{n(live_x)}" y="111" fill="#FF4D5A" font-family="{FONT}" font-size="12" '
        'font-weight="700">LIVE</text>',
        f'<rect x="{n(pill_x)}" y="94" width="{n(pill_w)}" height="24" rx="12" '
        f'fill="{t["chrome"]}" fill-opacity=".16" stroke="{t["chrome"]}"/>',
        f'<text x="{n(pill_x + pill_w / 2)}" y="111" text-anchor="middle" fill="{t["chrome"]}" '
        f'font-family="{FONT}" font-size="14" font-weight="700" textLength="{n(tw(handle, 14))}" '
        f'lengthAdjust="spacingAndGlyphs">{esc(handle)}</text>',
    ]

    right = 1127.0
    y = 153.0
    for label, value in rows:
        label, value = str(label), str(value)
        lw, vw = tw(label, 14), tw(value, 14)
        out += [
            f'<text x="491" y="{n(y)}" fill="{t["muted"]}" font-family="{FONT}" font-size="14" '
            f'textLength="{n(lw)}" lengthAdjust="spacingAndGlyphs">{esc(label)}</text>',
            f'<path d="{leader(491 + lw + 12, right - vw - 12, y - 4)}" fill="none" '
            f'stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
            f'<text x="{n(right)}" y="{n(y)}" text-anchor="end" fill="{t["text"]}" '
            f'font-family="{FONT}" font-size="14" textLength="{n(vw)}" '
            f'lengthAdjust="spacingAndGlyphs">{esc(value)}</text>',
        ]
        y += 23

    status = "● " + cfg["status_left"]
    sw = tw(status, 11)
    right_txt = cfg["status_right"]
    out += [
        f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
        f'<text x="491" y="548" fill="{t["accent"]}" font-family="{FONT}" font-size="11" '
        f'textLength="{n(sw)}" lengthAdjust="spacingAndGlyphs">{esc(status)}</text>',
        f'<rect x="{n(491 + sw + 6)}" y="539" width="6" height="10" fill="{t["accent"]}">'
        '<animate attributeName="opacity" values="1;0;1" dur="1.1s" repeatCount="indefinite"/></rect>',
        f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" font-family="{FONT}" '
        f'font-size="11" textLength="{n(tw(right_txt, 11))}" lengthAdjust="spacingAndGlyphs">'
        f"{esc(right_txt)}</text>",
        "</svg>",
    ]
    return "".join(out)


# --------------------------------------------------------------------- main --

def load_config(path: Path) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    user = json.loads(path.read_text(encoding="utf-8"))
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    if len(cfg["rows"]) > MAX_ROWS:
        sys.exit(f"config.json has {len(cfg['rows'])} rows; the panel fits {MAX_ROWS}.")
    return cfg


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Generate the profile.sh --live banner.")
    ap.add_argument("--photo", default=str(here / "photo.png"), help="your picture (PNG/JPG)")
    ap.add_argument("--config", default=str(here / "config.json"))
    ap.add_argument("--out", default=str(here / "assets"), help="output folder")
    args = ap.parse_args()

    photo, config, out_dir = Path(args.photo), Path(args.config), Path(args.out)
    if not photo.exists():
        sys.exit(f"Photo not found: {photo}")
    if not config.exists():
        sys.exit(f"Config not found: {config}")

    cfg = load_config(config)
    crop = cfg["photo_crop"]
    img = load_portrait(photo, float(crop["zoom"]), float(crop["x"]), float(crop["y"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        dots = portrait_dots(img, theme)
        target = out_dir / f"banner-{theme}.svg"
        target.write_text(render_svg(theme, dots, cfg), encoding="utf-8")
        print(f"{target}: {target.stat().st_size / 1024:.0f} KiB, {int(dots.sum()):,} dots")


if __name__ == "__main__":
    main()
