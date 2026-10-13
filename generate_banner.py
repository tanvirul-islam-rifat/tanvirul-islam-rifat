#!/usr/bin/env python3
"""Terminal-style GitHub profile banner ("profile.sh --live") with morphing dots.

Loop:   photo 1  ->  photo 2  ->  sign 1  ->  sign 2  ->  back to photo 1
Each picture is a 1-bit dithered dot image. Between pictures ~1000 dots fly
from their old position to a new one (optimal assignment, so nothing crosses
wildly). Everything is plain SVG animation, so it runs inside a GitHub README.

    pip install pillow numpy scipy
    python generate_banner.py --config config.json --out assets
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

W, H = 1180, 610
GRID_W, GRID_H = 300, 340          # photo lattice, drawn 1:1
PORTRAIT_X, PORTRAIT_Y = 94, 162   # top-left of the photo inside the left panel
SIGN_BOX = 270                     # signs are scaled to fit a square this big
SIGN_CX, SIGN_CY = PORTRAIT_X + GRID_W / 2, PORTRAIT_Y + GRID_H / 2
FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
MAX_ROWS = 16
EASE = "0.55 0 0.25 1"             # easing used while the dots travel

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
    "photos": [],
    "signs": ["code", "window"],
    "animation": {"hold": 2.4, "move": 1.3, "dots": 1000, "seed": 314159},
    "rows": [["Subject", "Your Name"]],
}


# ------------------------------------------------------------ built-in signs --

def sign_code() -> Image.Image:
    """A </> mark."""
    im = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    w = 44
    d.line([(150, 92), (62, 200), (150, 308)], fill="black", width=w, joint="curve")
    d.line([(250, 92), (338, 200), (250, 308)], fill="black", width=w, joint="curve")
    d.line([(228, 70), (172, 330)], fill="black", width=w)
    return im


def sign_window() -> Image.Image:
    """A square split into four panes."""
    im = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((60, 60, 340, 340), radius=18, outline="black", width=34)
    d.rectangle((183, 60, 217, 340), fill="black")
    d.rectangle((60, 183, 340, 217), fill="black")
    return im


BUILTIN_SIGNS = {"code": sign_code, "window": sign_window}


# --------------------------------------------------------------- file lookup --

def resolve(name: str, base: Path) -> Path:
    p = Path(name).expanduser()
    if p.is_absolute():
        return p
    for cand in (base / p, Path.cwd() / p):
        if cand.exists():
            return cand
    return base / p


# -------------------------------------------------------------------- photos --

def load_photo(path: Path, zoom: float, cx: float, cy: float) -> Image.Image:
    """Crop to the photo aspect ratio and resize to the dot lattice."""
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


def photo_dots(img: Image.Image, theme: str) -> np.ndarray:
    """Boolean (GRID_H, GRID_W) array: which lattice cells get a dot."""
    alpha = np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0
    has_alpha = bool(alpha.min() < 0.99)

    if theme == "dark":
        gray = np.asarray(ImageOps.grayscale(img.convert("RGB")), dtype=np.float32)
        prepared = Image.fromarray(np.uint8(np.clip(gray * alpha, 0, 255)), "L")
        mask = None
        if has_alpha:
            mask = Image.fromarray(np.uint8((alpha > 0.08) * 255), "L")
        # Stretch the range without flooding the highlights, then apply a gamma
        # curve so midtones dither into sparse dots and features stay visible.
        prepared = ImageOps.autocontrast(prepared, cutoff=1, mask=mask)
        curved = (np.asarray(prepared, dtype=np.float32) / 255.0) ** 1.6
        prepared = Image.fromarray(np.uint8(np.clip(curved * 255, 0, 255)), "L")
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


def dots_to_points(active: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(active)
    return np.column_stack((PORTRAIT_X + xs + 0.5, PORTRAIT_Y + ys + 0.5)).astype(np.float32)


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


# --------------------------------------------------------------------- signs --

def sign_points(spec: str, base: Path) -> np.ndarray:
    """All candidate dot positions (banner coordinates) for one sign."""
    if spec in BUILTIN_SIGNS:
        img = BUILTIN_SIGNS[spec]()
    else:
        path = resolve(spec, base)
        if not path.exists():
            sys.exit(f"Sign not found: {spec} (use 'code', 'window' or an image file)")
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    img.thumbnail((400, 400), Image.Resampling.LANCZOS)

    alpha = np.asarray(img.getchannel("A"))
    if alpha.min() < 128:                       # transparent background
        mask = alpha > 127
    else:                                       # solid background: find the glyph
        gray = np.asarray(ImageOps.grayscale(img.convert("RGB"))).astype(np.int16)
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        mask = np.abs(gray - int(np.median(border))) > 70
        if mask.mean() > 0.6:
            mask = ~mask
    if not mask.any():
        sys.exit(f"Could not find any shape in sign: {spec}")

    ys, xs = np.nonzero(mask)
    minx, maxx, miny, maxy = xs.min(), xs.max(), ys.min(), ys.max()
    scale = SIGN_BOX / max(maxx - minx + 1, maxy - miny + 1)
    cx, cy = (minx + maxx + 1) / 2, (miny + maxy + 1) / 2
    px = SIGN_CX + (xs + 0.5 - cx) * scale
    py = SIGN_CY + (ys + 0.5 - cy) * scale
    return np.column_stack((px, py)).astype(np.float32)


# ------------------------------------------------------------------ morphing --

def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reorder target so source[i] flies to out[i] with minimum total distance."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    out = np.empty_like(target)
    out[rows] = target[cols]
    return out


# ---------------------------------------------------------------------- svg --

def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def tw(text: str, size: float) -> float:
    """Monospace width estimate (also used as textLength so fonts can't drift)."""
    return len(text) * size * 0.605


def num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".") or "0"


def kt(t: float, total: float) -> str:
    s = f"{min(max(t / total, 0.0), 1.0):.4f}".rstrip("0").rstrip(".")
    return s or "0"


def leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{num(x)} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def clean(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for t, v in points:
        if out and abs(t - out[-1][0]) < 1e-9:
            out[-1] = (t, v)
        else:
            out.append((t, v))
    return out


def opacity_anim(points: list[tuple[float, float]], total: float) -> str:
    pts = clean(points)
    keys = ";".join(kt(t, total) for t, _ in pts)
    vals = ";".join(f"{v:g}" for _, v in pts)
    return (
        f'<animate attributeName="opacity" dur="{total:g}s" repeatCount="indefinite" '
        f'keyTimes="{keys}" values="{vals}"/>'
    )


def render_svg(
    theme: str,
    dots_list: list[np.ndarray],
    photo_pts: list[np.ndarray],
    sign_pts: list[np.ndarray],
    cfg: dict,
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme]
    rows = cfg["rows"]
    handle = cfg["handle"]
    anim = cfg["animation"]
    hold, move, count = float(anim["hold"]), float(anim["move"]), int(anim["dots"])
    fade = min(0.35, move / 3)

    kinds = ["photo"] * len(photo_pts) + ["sign"] * len(sign_pts)
    pools = photo_pts + sign_pts
    N = len(kinds)
    total = N * (hold + move)
    animated = N > 1

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
        '<clipPath id="visualClip"><rect x="49" y="130" width="390" height="400"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        f'font-family="{FONT}" font-size="13" letter-spacing=".4">{esc(cfg["window_title"])}</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="{FONT}" font-size="13" '
        f'font-weight="700" letter-spacing="1.2">{esc(cfg["map_label"])}</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" font-family="{FONT}" '
        f'font-size="11">{GRID_W}×{GRID_H} / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
    ]

    # --- dense dot layers, one per photo; each fades in/out around its slot ---
    for i, dots in enumerate(dots_list):
        pts: list[tuple[float, float]] = []
        for j in range(N):
            a = j * (hold + move)
            hold_end = a + hold
            arrive = hold_end + move
            v = 1.0 if j == i else 0.0
            pts += [(a, v), (hold_end, v)]
            if j == i:
                pts.append((hold_end + fade, 0.0))
            if (j + 1) % N == i:
                pts.append((arrive - fade, 0.0))
        pts.append((total, 1.0 if i == 0 else 0.0))
        out.append(
            f'<g opacity="{1 if i == 0 else 0}">'
            + (opacity_anim(pts, total) if animated else "")
            + f'<path d="{runs_path(dots, PORTRAIT_X, PORTRAIT_Y)}" fill="none" '
            f'stroke="{t["dots"]}" stroke-width="1" opacity=".95"/></g>'
        )

    # --- travelling dots ---
    if animated:
        def pick(pool: np.ndarray) -> np.ndarray:
            return pool[rng.choice(len(pool), count, replace=len(pool) < count)]

        states = [pick(pools[0])]
        for pool in pools[1:]:
            states.append(transport(states[-1], pick(pool)))

        pos_times: list[float] = []
        pos_idx: list[int] = []
        for i in range(N):
            a = i * (hold + move)
            pos_times += [a, a + hold]
            pos_idx += [i, i]
        pos_times.append(total)
        pos_idx.append(0)
        keys = ";".join(kt(x, total) for x in pos_times)
        splines = ";".join("0 0 1 1" if k % 2 == 0 else EASE for k in range(2 * N))

        vis: list[tuple[float, float]] = []
        for i in range(N):
            a = i * (hold + move)
            hold_end = a + hold
            arrive = hold_end + move
            is_photo = kinds[i] == "photo"
            v = 0.0 if is_photo else 1.0
            vis += [(a, v), (hold_end, v)]
            if is_photo:
                vis.append((hold_end + fade, 1.0))
            if kinds[(i + 1) % N] == "photo":
                vis.append((arrive - fade, 1.0))
        vis.append((total, 0.0 if kinds[0] == "photo" else 1.0))

        out.append(f'<g fill="{t["dots"]}" opacity="0">{opacity_anim(vis, total)}')
        for k in range(count):
            vals = ";".join(
                f"{num(states[i][k, 0])} {num(states[i][k, 1])}" for i in pos_idx
            )
            out.append(
                '<path d="M-.65-.65h1.3v1.3h-1.3z">'
                f'<animateTransform attributeName="transform" type="translate" '
                f'dur="{total:g}s" repeatCount="indefinite" calcMode="spline" '
                f'keyTimes="{keys}" keySplines="{splines}" values="{vals}"/></path>'
            )
        out.append("</g>")

    out.append("</g>")

    out += [
        f'<text x="58" y="551" fill="{t["muted"]}" font-family="{FONT}" font-size="10">'
        f'PTS {int(dots_list[0].sum()):05d} · FS/SERPENTINE</text>',
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        f'<text x="490" y="111" fill="{t["chrome"]}" font-family="{FONT}" font-size="13" '
        f'font-weight="700" letter-spacing="1.2">{esc(cfg["info_label"])}</text>',
        f'<circle cx="{num(live_x - 12)}" cy="106" r="4" fill="#FF4D5A">'
        '<animate attributeName="opacity" values="1;.25;1" dur="1.6s" repeatCount="indefinite"/></circle>',
        f'<text x="{num(live_x)}" y="111" fill="#FF4D5A" font-family="{FONT}" font-size="12" '
        'font-weight="700">LIVE</text>',
        f'<rect x="{num(pill_x)}" y="94" width="{num(pill_w)}" height="24" rx="12" '
        f'fill="{t["chrome"]}" fill-opacity=".16" stroke="{t["chrome"]}"/>',
        f'<text x="{num(pill_x + pill_w / 2)}" y="111" text-anchor="middle" fill="{t["chrome"]}" '
        f'font-family="{FONT}" font-size="14" font-weight="700" textLength="{num(tw(handle, 14))}" '
        f'lengthAdjust="spacingAndGlyphs">{esc(handle)}</text>',
    ]

    right = 1127.0
    y = 153.0
    for label, value in rows:
        label, value = str(label), str(value)
        lw, vw = tw(label, 14), tw(value, 14)
        out += [
            f'<text x="491" y="{num(y)}" fill="{t["muted"]}" font-family="{FONT}" font-size="14" '
            f'textLength="{num(lw)}" lengthAdjust="spacingAndGlyphs">{esc(label)}</text>',
            f'<path d="{leader(491 + lw + 12, right - vw - 12, y - 4)}" fill="none" '
            f'stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
            f'<text x="{num(right)}" y="{num(y)}" text-anchor="end" fill="{t["text"]}" '
            f'font-family="{FONT}" font-size="14" textLength="{num(vw)}" '
            f'lengthAdjust="spacingAndGlyphs">{esc(value)}</text>',
        ]
        y += 23

    status = "● " + cfg["status_left"]
    sw = tw(status, 11)
    right_txt = cfg["status_right"]
    out += [
        f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
        f'<text x="491" y="548" fill="{t["accent"]}" font-family="{FONT}" font-size="11" '
        f'textLength="{num(sw)}" lengthAdjust="spacingAndGlyphs">{esc(status)}</text>',
        f'<rect x="{num(491 + sw + 6)}" y="539" width="6" height="10" fill="{t["accent"]}">'
        '<animate attributeName="opacity" values="1;0;1" dur="1.1s" repeatCount="indefinite"/></rect>',
        f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" font-family="{FONT}" '
        f'font-size="11" textLength="{num(tw(right_txt, 11))}" lengthAdjust="spacingAndGlyphs">'
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
    if not cfg["photos"]:
        sys.exit('config.json needs at least one entry in "photos".')
    if float(cfg["animation"]["hold"]) <= 0 or float(cfg["animation"]["move"]) <= 0:
        sys.exit('"hold" and "move" in "animation" must be greater than 0.')
    return cfg


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Generate the profile.sh --live banner.")
    ap.add_argument("--config", default=str(here / "config.json"))
    ap.add_argument("--out", default=str(here / "assets"), help="output folder")
    args = ap.parse_args()

    config, out_dir = Path(args.config), Path(args.out)
    if not config.exists():
        sys.exit(f"Config not found: {config}")
    base = config.resolve().parent
    cfg = load_config(config)

    photos: list[Image.Image] = []
    for spec in cfg["photos"]:
        if isinstance(spec, str):
            spec = {"file": spec}
        opts = {"zoom": 1.0, "x": 0.5, "y": 0.4}
        opts.update(spec)
        path = resolve(opts["file"], base)
        if not path.exists():
            sys.exit(f"Photo not found: {opts['file']}")
        photos.append(load_photo(path, float(opts["zoom"]), float(opts["x"]), float(opts["y"])))

    signs = [sign_points(spec, base) for spec in cfg["signs"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["animation"]["seed"])
    for index, theme in enumerate(THEMES):
        dots_list = [photo_dots(img, theme) for img in photos]
        photo_pts = [dots_to_points(d) for d in dots_list]
        rng = np.random.default_rng(seed + index)
        svg = render_svg(theme, dots_list, photo_pts, signs, cfg, rng)
        target = out_dir / f"banner-{theme}.svg"
        target.write_text(svg, encoding="utf-8")
        print(f"{target}: {target.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()