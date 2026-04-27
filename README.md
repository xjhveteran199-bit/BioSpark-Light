# BioSpark-Light

> **本地桌面版生物信号实验台 / Offline desktop biosignal lab**
> 数据整理 · 模型训练 · 我的模型 — 全程在你的电脑上跑，不联网，不上传。

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)]()

BioSpark-Light is the **standalone desktop spin-off** of [BioSpark](https://github.com/) — same training engine, same publication-quality figures, but stripped of the cloud-only bits (auth, inference API, real-time streaming) and packaged to run as `python launcher.py` on your laptop.

> **Why?** Researchers don't always want to upload medical data to a server. Light runs the same CNN training pipeline on your local CPU and writes everything to a per-user data folder. Train a model, export the `.pt`, share the `.pt` — your raw data never leaves the machine.

---

## What's included

| Tab | What it does |
|------|--------------|
| **Data Prep** | Turn raw recordings (long CSVs, ZIPs of recordings) into a training-ready CSV via three modes (folder-per-class, CSV+intervals, generic ZIP). |
| **Train** | 1D-CNN training with auto-optimization, early stopping, optional LR range search, and warm-start from your prior best checkpoint. |
| **My Models** | Versioned per-user checkpoint chain — every training run is saved; activate any version as the next warm-start source. |

**Outputs**: confusion matrix, t-SNE, training curves, per-class metrics, model architecture diagram, full HTML report — Nature/IEEE/Science style. PNG (300 DPI) + SVG. Bulk ZIP download.

**What's deliberately omitted** vs. the web version: the inference API on pre-trained models (ECG/EEG/EMG demos), real-time streaming monitor, Grad-CAM heatmap, multi-user auth, and PostgreSQL support. Light is for *training your own models on your own data*.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/BioSpark-Light.git
cd BioSpark-Light
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only torch
```

### 2. Run

```bash
python launcher.py
```

This:
1. Starts an embedded server on `http://127.0.0.1:8765` (loopback only, never exposed)
2. Opens your default browser to the UI
3. Drops a **system tray icon** so you can quit cleanly later

### 3. Use

- Drop a ZIP / CSV into **Data Prep**, click **Generate Training CSV**, then **Use for Training**
- In **Train**, pick a preset (Quick Test / Smart Auto / Publication Ready), click **Start Training**
- After training, browse exports in **T6. Publication-Quality Figures** or jump to **My Models** to see your version chain

---

## Where your data lives

Light writes everything to your OS's standard per-user data directory — never next to the executable, so reinstalling the app doesn't wipe your work:

| OS | Path |
|----|------|
| **Windows** | `%LOCALAPPDATA%\BioSpark-Light\` |
| **macOS** | `~/Library/Application Support/BioSpark-Light/` |
| **Linux** | `~/.local/share/BioSpark-Light/` |

Inside that folder:
```
biospark.db              # SQLite — training history + checkpoint metadata
uploads/                 # Raw datasets you upload (auto-cleaned by OS, not by app)
checkpoints/0/v1.pt      # Your trained models (per-version chain)
              v2.pt
              ...
```

The system-tray menu has a **"Data folder…"** shortcut that opens this directory in your file browser.

---

## CLI options

```
python launcher.py            # normal: open browser + tray
python launcher.py --no-tray  # headless mode (Ctrl+C to quit)
python launcher.py --no-open  # don't auto-open browser
python launcher.py --port 9000
```

---

## Architecture

```
launcher.py                   # entry point (uvicorn thread + browser + tray)
backend/
  config.py                   # platformdirs paths
  database.py                 # SQLAlchemy + SQLite (per-user)
  main.py                     # FastAPI app + static frontend
  routers/
    prep.py                   # /api/prep/*  — segmentation
    training.py               # /api/train/* — training jobs + WS metrics
    figures.py                # /api/figures/* — matplotlib publication figures
    model_history.py          # /api/models/* — version chain
  services/
    trainer.py                # Signal1DCNN + training loop
    data_preparator.py        # Mode A/B/C segmenters
    auto_optimizer.py         # LR range test, early stop, class weights
    dataset_loader.py         # CSV/ZIP parsing
    preprocess.py             # bandpass / window / normalize
  models/training_history.py  # ORM (TrainingRun, ModelCheckpoint)
frontend/                     # Vanilla HTML/CSS/JS — no build step
  index.html
  css/style.css
  js/{app,prep,trainer,figures,my_models}.js
```

No build step. No bundler. Edit JS, refresh browser.

---

## Development

```bash
# Run server only (no tray, no browser open) — useful when iterating on JS
python launcher.py --no-tray --no-open

# Or directly via uvicorn for hot reload:
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8765
```

API docs at **http://127.0.0.1:8765/docs**.

To force a different data dir for testing:
```bash
DATABASE_URL="sqlite+aiosqlite:///./test.db" python launcher.py
```

---

## Roadmap

- [x] Data Prep with three input modes
- [x] CNN training with warm-start chain
- [x] Publication-quality figure export
- [x] System tray + auto-open browser
- [ ] PyInstaller one-file bundle (no Python install required)
- [ ] Auto-update channel
- [ ] Optional license-gated features

---

## License

MIT — go ahead and fork.

---

<sub>Built with FastAPI + PyTorch + Plotly + Matplotlib · Spun off from [BioSpark](https://github.com/)</sub>
