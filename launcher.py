"""
BioSpark-Light launcher — desktop entry point.

Boots an embedded uvicorn server on 127.0.0.1:8765 in a background thread,
then opens the default browser to the local UI. A system tray icon (when
pystray + pillow are available) lets the user quit cleanly; otherwise the
process holds in the foreground until Ctrl+C / window close.

Usage:
    python launcher.py            # normal — opens browser, starts tray
    python launcher.py --no-tray  # no tray (e.g. headless smoke test)
    python launcher.py --no-open  # do not auto-open browser
"""

import argparse
import logging
import socket
import sys
import threading
import time
import webbrowser
from typing import Optional

import uvicorn

from backend.config import HOST, PORT, USER_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
log = logging.getLogger("biospark.launcher")


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    """Poll until uvicorn binds the socket — protects against opening the
    browser before the server is ready (which would 502 / connection refused)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.2)
    return False


def _run_server(host: str, port: int):
    """Run uvicorn in the calling thread. Started from a daemon thread."""
    # Import lazily so --help doesn't pay the cost of loading torch
    config = uvicorn.Config(
        "backend.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,  # noisy in a desktop app
        # Single worker — torch + multi-worker forking is asking for OOM,
        # and a desktop app has exactly one local user.
        workers=1,
    )
    server = uvicorn.Server(config)
    server.run()


def _build_tray(quit_callback):
    """Return a pystray.Icon ready to .run(), or None if pystray/PIL missing."""
    try:
        import pystray  # type: ignore
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        log.info("pystray / pillow not installed — skipping tray icon")
        return None

    # Generate a tiny 64x64 icon programmatically — avoids shipping an asset
    img = Image.new("RGB", (64, 64), color=(37, 99, 235))
    draw = ImageDraw.Draw(img)
    # ECG-ish zigzag in white
    pts = [(6, 32), (18, 32), (24, 16), (30, 48), (36, 8), (42, 48), (48, 32), (58, 32)]
    draw.line(pts, fill=(255, 255, 255), width=3)

    def _on_open(_icon, _item):
        webbrowser.open(f"http://{HOST}:{PORT}")

    def _on_quit(icon, _item):
        icon.stop()
        quit_callback()

    menu = pystray.Menu(
        pystray.MenuItem("Open BioSpark-Light", _on_open, default=True),
        pystray.MenuItem("Data folder…", lambda *_: webbrowser.open(USER_DATA_DIR.as_uri())),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _on_quit),
    )
    return pystray.Icon("BioSpark-Light", img, "BioSpark-Light", menu)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="biospark-light")
    parser.add_argument("--no-tray", action="store_true", help="Skip system tray icon")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--host", default=HOST, help=f"Bind host (default {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Bind port (default {PORT})")
    args = parser.parse_args(argv)

    log.info("BioSpark-Light starting…")
    log.info("Data dir: %s", USER_DATA_DIR)
    log.info("Server:   http://%s:%s", args.host, args.port)

    # Server thread
    server_thread = threading.Thread(
        target=_run_server,
        args=(args.host, args.port),
        daemon=True,
        name="biospark-uvicorn",
    )
    server_thread.start()

    if not _wait_for_port(args.host, args.port, timeout=20.0):
        log.error("Server did not come up on %s:%s within 20s", args.host, args.port)
        return 1

    log.info("Server ready.")

    if not args.no_open:
        webbrowser.open(f"http://{args.host}:{args.port}")

    # Tray loop blocks the main thread. Quit callback returns control here so
    # the process can exit cleanly. Without tray, just block on the server thread.
    quit_event = threading.Event()

    if args.no_tray:
        log.info("Press Ctrl+C to quit.")
        try:
            while server_thread.is_alive():
                server_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            log.info("Interrupted — shutting down.")
        return 0

    icon = _build_tray(quit_event.set)
    if icon is None:
        # Fall back to no-tray behavior
        log.info("Press Ctrl+C to quit.")
        try:
            while server_thread.is_alive():
                server_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            log.info("Interrupted — shutting down.")
        return 0

    try:
        icon.run()  # blocks until quit menu item clicked
    except KeyboardInterrupt:
        pass
    log.info("Quit requested — exiting.")
    # uvicorn is in a daemon thread, so process exit will tear it down
    return 0


if __name__ == "__main__":
    sys.exit(main())
