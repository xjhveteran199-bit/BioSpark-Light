# Building BioSpark-Light into a single executable

We use **PyInstaller one-folder mode** (not `--onefile`). The output is a
self-contained `dist/BioSpark-Light/` directory that runs on a target
machine **without Python installed**.

## Why one-folder, not one-file?

- Torch's many `.dll` / `.pyd` files don't always survive the `--onefile`
  temp-extract on Windows.
- Cold start is much faster (no extraction on every launch).
- Easier to inspect / patch a broken bundle.

## Build prerequisites

```bash
pip install pyinstaller platformdirs pystray pillow
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

A CPU-only torch wheel (~150 MB) keeps the bundle under ~700 MB. The CUDA
wheel adds 2-3 GB you don't need for inference-free local training on a
laptop.

## Build

From the repo root:

```bash
python -m PyInstaller --noconfirm --clean biospark-light.spec
```

This takes 3-8 minutes on a typical laptop. Output:

```
dist/
  BioSpark-Light/
    BioSpark-Light.exe        ← the entry point
    _internal/
      frontend/...            ← static UI (HTML/CSS/JS)
      torch/...               ← bundled torch DLLs
      ...                     ← all other deps
```

## Run

Double-click `dist/BioSpark-Light/BioSpark-Light.exe`, or from a shell:

```bash
./dist/BioSpark-Light/BioSpark-Light.exe
```

A console window will appear (we ship `console=True` until the UI is fully
trusted), then the default browser opens to `http://127.0.0.1:8765`. A
system tray icon lets you quit.

## Distribution

Zip the entire `dist/BioSpark-Light/` folder. Recipients extract it
anywhere (Desktop, Program Files, USB stick) and double-click the .exe.
No installer needed.

To make a real installer (MSI / NSIS / Inno Setup), wrap the folder
afterwards — out of scope for v0.1.

## Common issues

- **`ImportError: DLL load failed` on launch** — the bundled torch
  is missing a Visual C++ runtime. Install the **Microsoft Visual C++
  Redistributable for Visual Studio 2015-2022** on the target machine.
- **Antivirus quarantines the .exe** — known PyInstaller false positive.
  Sign the binary, or whitelist the folder. Will revisit when we
  publish releases on GitHub.
- **Bundle too big** — verify you installed the **CPU-only** torch wheel
  before running PyInstaller. The CUDA wheel inflates the bundle by
  2-3 GB even though we never use it.

## Switching to a windowed app (no console)

Once you trust the bundle, edit `biospark-light.spec`:

```python
exe = EXE(
    ...
    console=False,           # was True
)
```

Then rebuild. Users will see only the system tray icon — clean.
