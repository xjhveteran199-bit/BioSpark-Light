# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for BioSpark-Light.

Build:  pyinstaller --noconfirm biospark-light.spec
Output: dist/BioSpark-Light/BioSpark-Light.exe  (one-folder bundle)

We use one-folder mode (not --onefile) because:
  - torch's .dll/.pyd files don't survive --onefile cleanly on Windows
  - Cold start is much faster (no temp-extract on every launch)
  - Easier to inspect / patch a broken bundle

Output is ~600 MB-1 GB depending on torch build (CPU-only is leaner).
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# ---- Heavy frameworks: pull in everything (binaries, data, hidden imports) ----
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all("sklearn")
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all("scipy")
matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all("matplotlib")

# ---- App data: ship the entire frontend folder next to the exe ----
# Path inside bundle: BioSpark-Light/_internal/frontend/
app_datas = [
    ("frontend", "frontend"),
]

# ---- Hidden imports that PyInstaller can't infer ----
extra_hidden = [
    # FastAPI / uvicorn underbelly
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # SQLAlchemy async dialect
    "aiosqlite",
    "sqlalchemy.dialects.sqlite",
    "sqlalchemy.ext.asyncio",
    # Pydantic v2 lazily resolves these
    "pydantic.deprecated.decorator",
    # Our own modules — explicit so PyInstaller chases them
    "backend.main",
    "backend.config",
    "backend.database",
    "backend.routers.prep",
    "backend.routers.training",
    "backend.routers.figures",
    "backend.routers.model_history",
    "backend.services.trainer",
    "backend.services.preprocess",
    "backend.services.data_preparator",
    "backend.services.dataset_loader",
    "backend.services.dataset_cache",
    "backend.services.auto_optimizer",
    "backend.services.publication_figures",
    "backend.models.training_history",
]

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=torch_binaries + sklearn_binaries + scipy_binaries + matplotlib_binaries,
    datas=(
        app_datas
        + torch_datas
        + sklearn_datas
        + scipy_datas
        + matplotlib_datas
    ),
    hiddenimports=(
        extra_hidden
        + torch_hiddenimports
        + sklearn_hiddenimports
        + scipy_hiddenimports
        + matplotlib_hiddenimports
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Strip things we know we don't use to shave size
    excludes=[
        "tkinter",
        "PyQt5", "PyQt6", "PySide2", "PySide6",
        "IPython", "jupyter", "notebook",
        "pytest",
        "torchvision",  # we only use torch core
        "torchaudio",
        "onnxruntime",  # not in BioSpark-Light's surface
        "mne",
        "pyedflib",
        "neurokit2",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BioSpark-Light",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX often breaks numpy/torch DLLs on Windows
    console=True,        # Set False for a windowed app once UI is verified
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="frontend/assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BioSpark-Light",
)
