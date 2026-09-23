"""Render the looping AstraFoundry profile banner (assets/astrafoundry_banner.gif).

Footage and photography are public-domain NASA media, fetched on demand:
- Artemis I launch, viewed from Banana Creek (NASA/KSC, "Artemis I Isolated Launch Views")
- Night sky over Kennedy Space Center (NASA/KSC, KSC-20191031-PH-GEB01_0003)
The title is set in Newsreader (SIL OFL), fetched from google/fonts.

Requires: Pillow, numpy, ffmpeg; gifsicle (optional) for smaller output.
Usage: python3 scripts/make_banner.py
"""

import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "astrafoundry_banner.gif"
CACHE = Path(tempfile.gettempdir()) / "astrafoundry-banner"

LAUNCH_ID = "KSC-20221116-MH-AJN01-0001-Artemis_I_Isolated_Launch_Views-3314595"
LAUNCH_URL = f"https://images-assets.nasa.gov/video/{LAUNCH_ID}/{LAUNCH_ID}~orig.mp4"
SKY_URL = (
    "https://images-assets.nasa.gov/image/KSC-20191031-PH-GEB01_0003/"
    "KSC-20191031-PH-GEB01_0003~orig.jpg"
)
FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/newsreader/" + urllib.parse.quote(
    "Newsreader[opsz,wght].ttf"
)

# The Banana Creek shot of the launch starts at 8:18 in the NASA reel (3840x2160).
CLIP_START, CLIP_LEN = 498.0, 24.0
SPEEDUP = 2.4
# Keep the rocket near the left edge so it never runs behind the centered title.
CROP = (1558, 667, 2282, 913)  # x, y, w, h in source pixels
# Region of the night-sky photo that is free of foreground grass.
SKY_CROP = (0, 0, 3000, 1200)

STAR_THRESHOLD = 14

W, H = 900, 360
FPS = 10
FADE_IN, FADE_OUT = 6, 12  # frames
NOISE_HOLD = 4

IVORY = np.array([242, 236, 226], dtype=np.float32)


def fetch(url, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def launch_clip():
    path = CACHE / "launch.mp4"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(CLIP_START), "-t", str(CLIP_LEN),
             "-i", LAUNCH_URL, "-an", "-c", "copy", str(path)],
            check=True,
        )
    return path


def video_frames():
    x, y, w, h = CROP
    vf = (
        f"crop={w}:{h}:{x}:{y},setpts=PTS/{SPEEDUP},fps={FPS},"
        f"scale={W}:{H}:flags=lanczos,hqdn3d=4:3:10:10"
    )
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(launch_clip()), "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32)


def star_layer():
    img = Image.open(fetch(SKY_URL, "night_sky.jpg")).convert("RGB")
    x, y, w, h = SKY_CROP
    img = img.crop((x, y, x + w, y + h)).resize((W, H), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    # Drop the sky glow and keep the stars and the Milky Way.
    floor = np.asarray(img.filter(ImageFilter.GaussianBlur(24)), dtype=np.float32)
    stars = np.clip(a - floor - STAR_THRESHOLD, 0, 255) * 2.4
    # Fade out toward the horizon of the launch shot.
    fade = np.clip((H * 0.9 - np.arange(H)) / (H * 0.2), 0, 1)[:, None, None]
    return stars * fade


def title_layer():
    font = ImageFont.truetype(str(fetch(FONT_URL, "Newsreader.ttf")), 84)
    font.set_variation_by_axes([380, 72])
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    text = "AstraFoundry"
    bx = d.textbbox((0, 0), text, font=font)
    tw, th = bx[2] - bx[0], bx[3] - bx[1]
    d.text(((W - tw) / 2 - bx[0], (H - th) / 2 - bx[1] - 10), text, font=font, fill=255)
    mask = np.asarray(layer, dtype=np.float32) / 255
    shadow = np.asarray(layer.filter(ImageFilter.GaussianBlur(18)), dtype=np.float32) / 255
    return mask[..., None], np.clip(shadow * 1.6, 0, 1)[..., None]


def envelope(n):
    """Fade the footage in and out so both ends of the loop are the bare night sky."""
    i = np.arange(n)
    return np.clip(np.minimum((i + 1) / FADE_IN, (n - i) / FADE_OUT), 0, 1)


def compose(frames):
    stars = star_layer()
    mask, shadow = title_layer()
    out = []
    for f, k in zip(frames, envelope(len(frames))):
        f = f * k
        lum = f.mean(axis=2, keepdims=True) / 255
        # Stars wash out as the launch glow lights up the sky.
        s = stars * np.clip(1 - lum * 3.0, 0, 1)
        img = 255 - (255 - f) * (255 - s) / 255
        img = img * (1 - shadow * 0.45)
        img = img * (1 - mask) + IVORY * mask
        out.append(img)
    return out


def stabilize(frames):
    """Hold pixels that barely change so sensor noise does not bloat the GIF."""
    prev = frames[0]
    out = [prev]
    for f in frames[1:]:
        still = (np.abs(f - prev).max(axis=2, keepdims=True) < NOISE_HOLD)
        prev = np.where(still, prev, f)
        out.append(prev)
    return out


def render():
    frames = stabilize(compose(video_frames()))
    tmp = Path(tempfile.mkdtemp())
    for i, f in enumerate(frames):
        Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).save(tmp / f"f{i:03d}.png")

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
    if shutil.which("gifsicle"):
        subprocess.run(["gifsicle", "-b", "-O3", "--lossy=25", str(OUT)], check=True)
    print(f"wrote {OUT} ({len(frames)} frames, {OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    render()
