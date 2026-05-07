"""
Build frontend/assets/download-qr.png — a QR code that points at the
BioSpark-Light "latest release" page on GitHub.

Pointing at /releases/latest (rather than a versioned tag) means the same
printed/shared QR keeps working across version bumps; users see the latest
release notes first, then click the .zip asset to download.

Re-run only if the URL changes (e.g. repo rename); version bumps don't
require regenerating.

Usage:
    python scripts/build_download_qr.py

Output:
    frontend/assets/download-qr.png
"""
from __future__ import annotations

from pathlib import Path

import qrcode

URL = "https://github.com/xjhveteran199-bit/BioSpark-Light/releases/latest"
OUT = Path(__file__).resolve().parent.parent / "frontend" / "assets" / "download-qr.png"


def main() -> int:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(URL)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)

    sz = OUT.stat().st_size
    print(f"wrote {OUT} ({sz:,} bytes) -> {URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
