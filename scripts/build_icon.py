"""
Build frontend/assets/icon.ico from the BioSpark-Light web logo design.

The logo (matching the SVG in frontend/index.html lines 20-32):
  - Rounded square background, semi-transparent indigo fill
  - Diagonal gradient stroke (cyan #22d3ee → indigo #6366f1 → pink #f472b6)
  - ECG-style zigzag line in the center, same gradient stroke

PIL doesn't natively support gradient strokes, so we render the shapes as
a pure-white mask first, then composite a gradient image through that mask.
This gives clean edges at every output resolution.

Usage:
    python scripts/build_icon.py

Output:
    frontend/assets/icon.ico   (multi-res ICO: 16/32/48/64/128/256)

Re-run this whenever the logo design changes; the .ico is then picked up
by biospark-light.spec on the next PyInstaller build.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

# ─── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "frontend" / "assets" / "icon.ico"

# ─── Design constants (mirror the SVG in index.html) ──────────────────
# We render at 1024x1024 then downsample for crisp edges at every ICO size
RENDER_SIZE = 1024
PADDING = 64                 # outer breathing room (so the corner radius is visible)
CORNER_RADIUS = 256          # rx="12" on a 42px viewBox → ~28% radius
RECT_STROKE_W = 14           # SVG used 1.5 on 42px → scaled to 1024
ECG_STROKE_W = 60            # SVG used 2.5 on 42px → scaled, slightly bumped for icon legibility

# Gradient stops (matching --neon-green / --primary / --neon-pink in style.css)
GRADIENT_STOPS = [
    (0.00, (0x22, 0xd3, 0xee)),  # cyan
    (0.50, (0x63, 0x66, 0xf1)),  # indigo
    (1.00, (0xf4, 0x72, 0xb6)),  # pink
]

# Background fill: matches rgba(99,102,241,0.15) but bumped to 0.55 alpha
# so the icon doesn't disappear against dark Windows themes
BG_FILL = (0x63, 0x66, 0xf1, 140)

# ECG zigzag path — direct port from the SVG path "M5 21 L10 21 L13 12 ..."
# Coordinates here are in 42-unit space; we scale to RENDER_SIZE-PADDING space below
ECG_PATH = [
    (5,  21),  (10, 21),
    (13, 12),  (17, 30),  (21, 7),  (25, 30),  (29, 12),
    (32, 21),  (37, 21),
]

ICO_SIZES = [(s, s) for s in (16, 32, 48, 64, 128, 256)]


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def sample_gradient(t: float) -> tuple[int, int, int]:
    """Sample the multi-stop gradient at position t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    for i in range(len(GRADIENT_STOPS) - 1):
        t0, c0 = GRADIENT_STOPS[i]
        t1, c1 = GRADIENT_STOPS[i + 1]
        if t0 <= t <= t1:
            local = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return lerp_color(c0, c1, local)
    return GRADIENT_STOPS[-1][1]


def make_diagonal_gradient(size: int) -> Image.Image:
    """Generate an opaque RGBA gradient image, top-left → bottom-right."""
    img = Image.new("RGBA", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2.0 * (size - 1))
            r, g, b = sample_gradient(t)
            px[x, y] = (r, g, b, 255)
    return img


def scale_path(path: list[tuple[int, int]], src_extent: int, dst_extent: int, offset: int) -> list[tuple[int, int]]:
    s = dst_extent / src_extent
    return [(int(round(x * s)) + offset, int(round(y * s)) + offset) for x, y in path]


def render_logo(size: int = RENDER_SIZE) -> Image.Image:
    inner = size - 2 * PADDING

    # ─── Layer 1: background fill (semi-transparent indigo rounded square) ───
    base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base_draw = ImageDraw.Draw(base)
    base_draw.rounded_rectangle(
        (PADDING, PADDING, size - PADDING, size - PADDING),
        radius=CORNER_RADIUS,
        fill=BG_FILL,
    )

    # ─── Layer 2: white mask of all stroked shapes ───
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    # Rounded-rect stroke
    mask_draw.rounded_rectangle(
        (PADDING, PADDING, size - PADDING, size - PADDING),
        radius=CORNER_RADIUS,
        outline=255,
        width=RECT_STROKE_W,
    )
    # ECG zigzag (anti-aliased thick line; PIL's draw.line is jaggy at thick widths,
    # so we draw it twice — once as line, once with overlapping ellipses at vertices for round joins)
    ecg_pts = scale_path(ECG_PATH, src_extent=42, dst_extent=inner, offset=PADDING)
    mask_draw.line(ecg_pts, fill=255, width=ECG_STROKE_W, joint="curve")
    # Round caps + smooth joints
    r = ECG_STROKE_W // 2
    for x, y in ecg_pts:
        mask_draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    # ─── Layer 3: gradient pasted through the mask ───
    grad = make_diagonal_gradient(size)
    base.paste(grad, (0, 0), mask)

    return base


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering logo at {RENDER_SIZE}×{RENDER_SIZE}…")
    logo = render_logo(RENDER_SIZE)

    # Pillow's ICO encoder will downsample to each requested size with a high-quality
    # filter (LANCZOS by default in recent versions).
    print(f"Writing multi-res ICO ({len(ICO_SIZES)} sizes) → {OUT_PATH}")
    logo.save(OUT_PATH, format="ICO", sizes=ICO_SIZES)

    sz = OUT_PATH.stat().st_size
    print(f"Done. icon.ico = {sz:,} bytes ({sz / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
