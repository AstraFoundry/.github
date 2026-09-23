"""Render the looping AstraFoundry profile banner (assets/astrafoundry_banner.gif).

Requires: Pillow, numpy, ffmpeg. Fonts (SIL OFL) are fetched from google/fonts.
Usage: python3 scripts/make_banner.py
"""

import math
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "astrafoundry_banner.gif"
FONT_DIR = Path(tempfile.gettempdir()) / "astrafoundry-fonts"
FONTS = {
    "title": "ofl/pinyonscript/PinyonScript-Regular.ttf",
    "caption": "ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
}

W, H = 1200, 400
FRAMES = 90
FPS = 20
SEED = 7
PAD = (190.0, H - 30.0)

IVORY = np.array([244, 232, 208], dtype=np.float32)
FLAME = np.array([255, 196, 128], dtype=np.float32)
SMOKE = np.array([150, 160, 185], dtype=np.float32)


def font_path(key):
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    path = FONT_DIR / Path(FONTS[key]).name.replace("%5B", "[").replace("%5D", "]")
    if not path.exists():
        url = "https://raw.githubusercontent.com/google/fonts/main/" + FONTS[key]
        urllib.request.urlretrieve(url, path)
    return path


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def background():
    y = np.linspace(0, 1, H)[:, None, None]
    x = np.linspace(-1, 1, W)[None, :, None]
    top = np.array([4, 6, 14], dtype=np.float32)
    bottom = np.array([14, 18, 38], dtype=np.float32)
    img = top + (bottom - top) * y ** 1.6
    # Faint horizon glow and a diffuse band of milky way across the sky.
    img = img + np.array([40, 30, 34]) * np.exp(-((1 - y) / 0.10) ** 2) * 0.35
    band = np.exp(-((y - 0.25 - 0.18 * x) / 0.16) ** 2)
    img = img + np.array([18, 20, 34]) * band * 0.6
    return np.broadcast_to(img, (H, W, 3)).astype(np.float32).copy()


def make_stars(rng):
    n = 520
    return {
        "x": rng.uniform(0, W, n),
        "y": rng.uniform(0, H * 0.92, n),
        "b": rng.uniform(0.15, 1.0, n) ** 3,
        "phase": rng.uniform(0, 2 * math.pi, n),
        # Integer cycles per loop keep the twinkle seamless.
        "cycles": rng.integers(1, 4, n),
        "tint": rng.uniform(0, 1, n),
    }


def splat(img, x, y, radius, color, strength):
    """Additively draw a soft gaussian point."""
    r = int(max(2, radius * 3))
    x0, x1 = int(max(0, x - r)), int(min(W, x + r + 1))
    y0, y1 = int(max(0, y - r)), int(min(H, y + r + 1))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * radius ** 2)) * strength
    img[y0:y1, x0:x1] += g[..., None] * color


def blend(img, x, y, radius, color, alpha):
    """Alpha-blend a soft gaussian puff (used for smoke)."""
    r = int(max(2, radius * 2.5))
    x0, x1 = int(max(0, x - r)), int(min(W, x + r + 1))
    y0, y1 = int(max(0, y - r)), int(min(H, y + r + 1))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    a = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * radius ** 2)) * alpha
    region = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = region * (1 - a[..., None]) + color * a[..., None]


def draw_stars(img, stars, t):
    warm = np.array([255, 236, 210], dtype=np.float32)
    cool = np.array([205, 220, 255], dtype=np.float32)
    for x, y, b, ph, cyc, tint in zip(
        stars["x"], stars["y"], stars["b"], stars["phase"], stars["cycles"], stars["tint"]
    ):
        tw = 0.65 + 0.35 * math.sin(2 * math.pi * cyc * t + ph)
        color = warm * tint + cool * (1 - tint)
        if b > 0.45:
            splat(img, x, y, 0.9 + b * 0.8, color, b * tw * 0.9)
            splat(img, x, y, 3.5, color, b * tw * 0.06)
        else:
            xi, yi = int(x), int(y)
            img[yi, xi] += color * b * tw * 0.8


def draw_shooting_star(img, t):
    # A single meteor, fully contained in [0.62, 0.74] of the loop.
    p = (t - 0.62) / 0.12
    if not 0 <= p <= 1:
        return
    fade = math.sin(math.pi * p)
    sx, sy = 1080 - 260 * p, 40 + 90 * p
    for i in range(40):
        k = i / 40
        splat(img, sx + 90 * k * 0.95, sy - 33 * k, 0.9, IVORY, fade * (1 - k) ** 2 * 0.9)


def rocket_state(t):
    """Liftoff at 0.08, out of frame by 0.60; p is flight progress in [0, 1]."""
    p = min(max((t - 0.08) / 0.52, 0.0), 1.0)
    ease = p ** 2.2
    base_x, base_y = PAD
    y = base_y - ease * (H + 180)
    x = base_x + 70 * ease ** 1.6
    angle = math.atan2(70 * 1.6 * ease ** 0.6, H + 180)
    return x, y, angle, p


def draw_ground(img):
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    pts = [(0, H)]
    for i in range(0, W + 1, 20):
        pts.append((i, H - 18 - 6 * math.sin(i / 97) - 4 * math.sin(i / 31 + 1)))
    pts.append((W, H))
    d.polygon(pts, fill=255)
    # Launch tower beside the pad.
    tx = 168
    d.rectangle([tx, H - 70, tx + 5, H - 18], fill=255)
    for yy in range(H - 70, H - 18, 8):
        d.line([tx, yy, tx + 5, yy + 8], fill=255)
    d.rectangle([tx + 5, H - 62, tx + 14, H - 60], fill=255)
    a = np.asarray(layer, dtype=np.float32)[..., None] / 255
    ground = np.array([3, 4, 9], dtype=np.float32)
    img[:] = img * (1 - a) + ground * a


def rocket_sprite(scale=1.0):
    w, h = int(12 * scale), int(52 * scale)
    im = Image.new("RGBA", (w * 3, h + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = w * 1.5
    body = [
        (cx, 0),
        (cx + w * 0.5, h * 0.22),
        (cx + w * 0.5, h * 0.86),
        (cx + w * 1.1, h),
        (cx - w * 1.1, h),
        (cx - w * 0.5, h * 0.86),
        (cx - w * 0.5, h * 0.22),
    ]
    d.polygon(body, fill=(214, 214, 222, 255))
    d.polygon([(cx, 0), (cx + w * 0.5, h * 0.22), (cx, h * 0.22)], fill=(150, 150, 165, 255))
    d.polygon([(cx, h * 0.22), (cx + w * 0.5, h * 0.22), (cx + w * 0.5, h * 0.86), (cx, h * 0.86)],
              fill=(160, 162, 176, 255))
    return im, (cx, h)


def blit_sprite(img, sprite, x, y, ang, alpha, warm):
    spr = sprite.rotate(-math.degrees(ang), resample=Image.BICUBIC, expand=True)
    arr = np.asarray(spr, dtype=np.float32)
    a = arr[..., 3:4] / 255 * alpha
    sh, sw = arr.shape[:2]
    ox, oy = int(x - sw / 2), int(y - sh + 8)
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(W, ox + sw), min(H, oy + sh)
    if x0 < x1 and y0 < y1:
        sub = arr[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        sa = a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        col = sub[..., :3] * 0.55 + FLAME * 0.12 * warm
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - sa) + col * sa


def draw_rocket(img, t, sprite):
    x, y, ang, p = rocket_state(t)
    smoke_fade = 1 - smoothstep(0.62, 0.97, t)
    ignition = smoothstep(0.02, 0.08, t)
    base_x, base_y = PAD

    # The next rocket rolls onto the pad at the end so the loop closes seamlessly.
    if t >= 0.80:
        blit_sprite(img, sprite, base_x, base_y, 0.0, smoothstep(0.80, 0.95, t), 0.0)

    # Pad glow and ground smoke billowing out at ignition.
    if smoke_fade > 0 and t >= 0.02:
        glow = ignition * (1 - smoothstep(0.18, 0.5, t))
        splat(img, base_x, base_y, 38, FLAME, 0.35 * glow)
        rng = np.random.default_rng(3)
        for i in range(26):
            spread = smoothstep(0.03, 0.55, t)
            dx = rng.uniform(-1, 1) * (30 + 150 * spread)
            dy = -rng.uniform(0, 1) * 30 * spread
            rad = 10 + 20 * spread * rng.uniform(0.6, 1.2)
            blend(img, base_x + dx, base_y + dy, rad, SMOKE * 0.55, 0.30 * ignition * smoke_fade)

    if p >= 1:
        return

    # Exhaust trail: sample past positions of the rocket.
    for i in range(60):
        tp = t - i * 0.006
        if tp < 0.08:
            break
        px, py, _, _ = rocket_state(tp)
        age = t - tp
        rad = 4 + age * 90
        alpha = 0.22 * math.exp(-age * 6) * smoke_fade
        blend(img, px, py + 8, rad, SMOKE * 0.8, alpha)

    if ignition > 0:
        flick = (0.85 + 0.15 * math.sin(t * 2 * math.pi * 23)) * ignition
        nx, ny = x - math.sin(ang) * 2, y + 4
        splat(img, nx, ny + 6, 5, FLAME, 1.6 * flick)
        splat(img, nx, ny + 14, 9, FLAME, 0.6 * flick)
        splat(img, nx, ny + 10, 26, FLAME, 0.18 * flick)
        splat(img, nx, ny + 3, 2.2, np.array([255, 250, 235]), 2.0 * ignition)

    blit_sprite(img, sprite, x, y, ang, 1.0, ignition)


def text_layer():
    title_font = ImageFont.truetype(str(font_path("title")), 118)
    cap_font = ImageFont.truetype(str(font_path("caption")), 17)
    try:
        cap_font.set_variation_by_axes([500])
    except Exception:
        pass

    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    title = "AstraFoundry"
    bx = d.textbbox((0, 0), title, font=title_font)
    tw, th = bx[2] - bx[0], bx[3] - bx[1]
    tx, ty = (W - tw) / 2 - bx[0], (H - th) / 2 - bx[1] - 22
    d.text((tx, ty), title, font=title_font, fill=255)

    caption = " ".join("INFRASTRUCTURE FOR HUMANS & AGENTS")
    cb = d.textbbox((0, 0), caption, font=cap_font)
    cw = cb[2] - cb[0]
    cy = (H + th) / 2 + 6
    d.text(((W - cw) / 2 - cb[0], cy), caption, font=cap_font, fill=170)

    line_y = cy - 12
    half = 60
    d.line([(W / 2 - half, line_y), (W / 2 + half, line_y)], fill=90, width=1)

    mask = np.asarray(layer, dtype=np.float32) / 255
    glow = np.asarray(layer.filter(ImageFilter.GaussianBlur(14)), dtype=np.float32) / 255
    return mask, glow


def render():
    rng = np.random.default_rng(SEED)
    bg = background()
    stars = make_stars(rng)
    sprite, _ = rocket_sprite()
    yy, xx = np.mgrid[0:H, 0:W]
    vignette = 1 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 * 0.6 + ((yy - H / 2) / (H / 2)) ** 2 * 0.4)
    mask, glow = text_layer()

    tmp = Path(tempfile.mkdtemp())
    for f in range(FRAMES):
        t = f / FRAMES
        img = bg.copy()
        draw_stars(img, stars, t)
        draw_shooting_star(img, t)
        draw_ground(img)
        draw_rocket(img, t, sprite)

        breathe = 0.85 + 0.15 * math.sin(2 * math.pi * t)
        img += glow[..., None] * IVORY * 0.22 * breathe
        img = img * (1 - mask[..., None]) + IVORY * mask[..., None]

        img *= vignette[..., None]

        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(tmp / f"f{f:03d}.png")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", str(tmp / "f%03d.png"),
            "-filter_complex",
            "split[a][b];[a]palettegen=max_colors=160:stats_mode=full[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-loop", "0", str(OUT),
        ],
        check=True,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    render()
