# BioSpark-Light User Manual

> **Document version**: aligned with BioSpark-Light v0.1.1
> **Audience**: anyone picking up BioSpark-Light for the first time who wants to walk the entire path from raw recordings to a trained model.
> **中文版本**: see [USER-MANUAL.md](USER-MANUAL.md).

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [Install and launch](#2-install-and-launch)
3. [The end-to-end workflow](#3-the-end-to-end-workflow)
4. [Data Prep tab](#4-data-prep-tab)
5. [Train tab](#5-train-tab)
6. [Reading results (T5)](#6-reading-results-t5)
7. [My Models tab](#7-my-models-tab)
8. [Worked example: 5-gesture OpenBCI EMG](#8-worked-example-5-gesture-openbci-emg)
9. [Core concepts cheat sheet](#9-core-concepts-cheat-sheet)
10. [Data-collection guidance (single-session → publishable)](#10-data-collection-guidance)
11. [FAQ and troubleshooting](#11-faq-and-troubleshooting)
12. [Doc map](#12-doc-map)

---

## 1. What this is

**BioSpark-Light** is the **standalone desktop spin-off** of [BioSpark](https://github.com/...) —
an **offline, no-signup, no-network** biosignal training lab.

**What it does**:

- ✅ Turns raw long recordings (CSV / TXT / OpenBCI export) into trainable samples
- ✅ Trains a 1D-CNN classifier (ECG / EEG / EMG / any time-series signal)
- ✅ Auto-renders the confusion matrix, t-SNE, training curves, publication-quality figures
- ✅ **Actively flags data leakage** (group-aware split + yellow warning banner)
- ✅ Warm-start: each new training run resumes from your previous best checkpoint

**What it deliberately does NOT do**:

- ❌ Online inference service (use the full BioSpark for that)
- ❌ Real-time streaming monitor (same)
- ❌ XAI features such as Grad-CAM / SHAP (stripped from this build)
- ❌ Multi-user collaboration (single-machine, single-user by design)
- ❌ Decide your data-collection protocol for you (§10 gives guidance)

---

## 2. Install and launch

### 2.1 End user — double-click the `.exe` (recommended)

1. Download `BioSpark-Light-vX.X.X.zip` from [GitHub Releases](https://github.com/xjhveteran199-bit/BioSpark-Light/releases)
2. Unzip somewhere (**not on Desktop** — Windows AV may flag it)
3. Double-click `BioSpark-Light.exe`
4. Console shows `[biospark.startup] BioSpark-Light ready`, browser auto-opens at <http://127.0.0.1:8765>
5. A system-tray icon appears; right-click → Quit to shut down cleanly

### 2.2 Developer — run from source

```bash
git clone https://github.com/xjhveteran199-bit/BioSpark-Light.git
cd BioSpark-Light
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python launcher.py
```

See [BUILD.md](BUILD.md) for full build instructions.

### 2.3 Where your data lives

| OS | Path |
|----|------|
| Windows | `%LOCALAPPDATA%\BioSpark-Light\` |
| macOS | `~/Library/Application Support/BioSpark-Light/` |
| Linux | `~/.local/share/BioSpark-Light/` |

Inside that folder:

```
biospark.db              # SQLite — training history & checkpoint metadata
uploads/                 # Raw datasets you upload
checkpoints/0/v1.pt      # Your trained models (versioned chain)
              v2.pt
```

**Reinstalling the app or moving the `.exe` does not wipe your data** — everything is stored per-user, independently of the executable's location.

---

## 3. The end-to-end workflow

```
┌───────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  Raw data         │     │                   │     │  Training-ready CSV│
│  (CSV/TXT/ZIP)    │ ──► │  📦 Data Prep     │ ──► │  (with __group__)  │
│  long / multi-file│     │  Mode A / B / C   │     │                    │
└───────────────────┘     └───────────────────┘     └────────┬───────────┘
                                                              │
                                                              ▼
                          ┌───────────────────┐     ┌────────────────────┐
                          │  📊 T5 results    │     │  🎯 Train          │
                          │  Confusion matrix │ ◄── │  Smart Auto / etc. │
                          │  t-SNE, leakage   │     │  WebSocket live    │
                          │  banner           │     │                    │
                          └───────────────────┘     └────────┬───────────┘
                                                              │
                                                              ▼
                                                    ┌────────────────────┐
                                                    │  🗂 My Models     │
                                                    │  Version chain     │
                                                    │  + warm-start      │
                                                    └────────────────────┘
```

**Three iron rules**:

1. **Always go through Data Prep** — never feed a pre-segmented CSV directly into Train (§4 explains why)
2. **Don't ignore the yellow banner on the results page** — it's an honest signal, see §6.3
3. **Report the test-set accuracy, never the validation-set number** — `Evaluation set: held-out test set` is the real figure

---

## 4. Data Prep tab

Cuts long recordings into training samples. **All data must pass through here first.**

### 4.1 Which Mode to pick

| Mode | Right input | Example |
|------|-------------|---------|
| **A — Folder-per-class ZIP** | Folder names ARE the class labels; each file is one independent recording | `gestures.zip` containing `CLENCH/r1.csv`, `CLENCH/r2.csv`, `FIVE/r1.csv` ... |
| **B — Single CSV/TXT + intervals** | One continuous recording; you mark time intervals + labels manually | A 30-min session: "0–600 s = CLENCH, 600–1200 s = FIVE, ..." |
| **C — Generic ZIP + per-file label** | Multiple flat files, you map each filename to a class | `data.zip` with `LIU-CLENCH.txt`, `LIU-FIVE.txt`, ... (this manual's example) |

> **Not sure which?** If your data is already organized by class folders → A. One long CSV needing manual segmentation → B. Anything else → C.

### 4.2 Segmentation parameters (shared by all modes)

| Field | Purpose | Typical value |
|-------|---------|---------------|
| **Sampling Rate (Hz)** | Signal sample rate | 250 (OpenBCI Cyton) |
| **Segment Length (sec)** | Duration of one training sample | 1.0 (EMG), 2.0 (EEG epoch), as needed |
| **Overlap Ratio** | Overlap fraction between adjacent segments — **ignored when stride is set** | Default 0 |
| **Signal Column Index** | Single-channel column number (0-based) | OpenBCI usually overrides this with the multi-channel field |
| **Multi-channel Columns** | Multi-channel column range; accepts `1-5` or `1,2,3,4,5` | OpenBCI Cyton: `1-8` (all 8) or `1-5` (first 5) |
| **Stride (sec, optional)** | Distance between segment starts. Empty = use overlap_ratio | "1 s task every 5 s": `segment=1, stride=5` |
| **Group-aware split unit** | Granularity for the train/val/test split: `recording` (default) or `trial` | See decision matrix below |

#### Choosing `Group-aware split unit`

> **Determines what counts as "one group" for downstream train/val/test splitting. Group-aware splitting is what prevents data leakage.**

| Your scenario | Pick | Why |
|---------------|------|-----|
| Each class has **multiple independent recordings** (multiple sessions / re-electrode) | **`recording`** (default) | One file = one group. All windows from one recording stay in the same split — no "half in train, half in test" |
| Each class has **only 1 long recording** containing repeated trials separated by rest | **`trial`** | Each emitted segment becomes its own group. Picking `recording` here would dump an entire class into one split, leaving test with only 1 class |
| Pre-segmented CSV without a `__group__` column | Either, Prep can't help here | Re-prepare from the original long recording |

**Quick decision tree**: do you have **≥ 4 files per class**? → `recording`. Otherwise → `trial`.

### 4.3 OpenBCI auto-detection

When you upload a `.txt`, the inspector scans the first 4 KB for the OpenBCI metadata header. **If detected**:

- ✅ `Sample Rate = 250.0 Hz` → auto-filled into the SR field
- ✅ Sample-index column auto-skipped (0–255 wrap-around integer detection)
- ✅ Saturated columns auto-skipped (one value covers ≥ 95 % of rows — disconnected EXG at the ADC rail)
- ✅ Zero-variance columns auto-skipped (unmoving accelerometer axes)
- ✅ Multi-channel column field auto-filled with `1-N` (the surviving real signal channels)

→ You only have to fill in segment / stride / group_by yourself.

### 4.4 File → label mapping (Mode B / C)

A table appears with one row per file. **An empty label cell skips that file.** Mode C requires at least one label, otherwise it errors.

Class labels can be **any string**: `CLENCH`, `gesture_01`, even Chinese — they pass straight through to training.

### 4.5 Output CSV schema

After `Generate Training CSV`, single-channel output looks like:

```
s1, s2, s3, ..., sN, label, __group__
0.12, 0.34, ..., 0.56, CLENCH, l/LIU-CLENCH.txt#trial_0
0.13, 0.35, ..., 0.57, CLENCH, l/LIU-CLENCH.txt#trial_1
...
```

Multi-channel output uses `ch1_1, ch1_2, ..., ch1_S, ch2_1, ..., chC_S` column names; the trainer auto-detects this and reshapes to `(N, C, L)` for the 1-D CNN.

**The `__group__` column never goes into the model** — it's used only for train/val/test splitting.

---

## 5. Train tab

### 5.1 Upload, or inherit from Prep

If you just clicked **Use for Training** in Prep, the dataset is already loaded.
Otherwise you can upload a pre-prepared training CSV directly (one with `__group__` is best).

### 5.2 Training-mode preset

| Preset | When to use | Time | Notes |
|--------|-------------|------|-------|
| **🚀 Smart Auto** | Default, beginners use this | 3–10 min CPU | Auto-architecture + class weights + early stopping |
| **⚡ Quick Test** | Sanity-check the data quickly | 30 s – 2 min | 20 epochs, no LR search |
| **🏆 Publication Ready** | Final pre-submission run | 15–45 min | 100 epochs + LR search |
| **⚙️ Custom** | Full manual | Your call | When you want full control |

### 5.3 Two pre-flight banners (both yellow warnings)

| Banner | Meaning | What to do |
|--------|---------|------------|
| ⚠️ This dataset has no `__group__` column | You uploaded a pre-segmented CSV, skipping Prep | Go back to Prep |
| (visible when Overlap > 0) | Overlap_ratio is set above zero | Set to 0 unless you have a reason |

### 5.4 Training config (shown only in Custom)

You usually don't need to touch these, but for Custom:

| Field | Default | When to change |
|-------|---------|----------------|
| Epochs | 30–50 | Reduce on very small datasets |
| Learning rate | 0.001 | Smart Auto finds this for you |
| Batch size | 64 | Halve when GPU memory tight |
| Validation split | 0.2 | Bump to 0.3 on small datasets |
| **Channels** | 0 (auto) | Auto-detected via `ch{N}_M` prefix |
| Early-stop patience | 10 | Lower to 5–8 on small datasets |
| Auto class weights | ✓ | Enable for severe class imbalance |
| Search optimal LR | ✗ | Only for Publication Ready |
| **Continue from previous model** (warm-start) | ✗ | See §7 |

### 5.5 Start training → check the first log line

Click **Start Training**. The moment the WebSocket connects, the first epoch-log line displays the **split mode**:

```
✅ Green: Split: group-aware · 305 groups · train=183 val=61 test=61
⚠️ Yellow: Split: per-row random (no __group__ column) · train=180 val=60 test=60
```

If you see yellow, **stop the run immediately** and go back to Prep (or accept that this dataset can't do group-split).

### 5.6 How to read the curves

| Pattern | Train loss | Val loss | Verdict |
|---------|-----------|----------|---------|
| Healthy | Steady downward | Tracks down then flattens | ✅ Just let it finish |
| Overfit | → 0 | Drops early then bounces back up | ⚠️ Early stopping should fire; if not, stop manually |
| Stuck | Jittering, no trend | Same | ⚠️ Wrong channel count / weird scaling |
| Leakage | Hits 99 %+ in a few epochs | Tracks at 99 %+ | ⚠️ See §6.3 banner after training |

---

## 6. Reading results (T5)

After training, the page auto-scrolls to **T5. Training Result Visualization**, with two panels:

### 6.1 Confusion matrix

**Small text above the matrix**:

```
Evaluation set: held-out test set
```

✅ This line **must say `test`**. If it says `validation set (no held-out test)`, your dataset was too small and the test split was abandoned — the number is not trustworthy.

**Axes**: rows = true class (`True`), columns = predicted class (`Predicted`). Diagonal = correct, off-diagonal = misclassifications.

**Display toggle**: the `#→` / `%→` button in the top-left swaps between counts and percentages.

### 6.2 t-SNE feature visualization

Projects the model's penultimate layer (128-dim features) into 2-D via t-SNE. **What healthy results look like**:

- 5 clusters roughly separable
- **Some overlap between clusters that the confusion matrix says are confused**
- Each cluster is tight but not collapsed to a single point

**Red flag**: 5 clusters separated **suspiciously cleanly** (no overlap at all) + confusion matrix all 100 % → **data leakage** (see §9).

### 6.3 ⚠️ Yellow "suspected data leakage" banner

**Trigger conditions (all three must hold)**:

1. Overall accuracy ≥ 99 %
2. Any class's test support < 30
3. Either no `__group__` column **or** no held-out test set

**When triggered**, the banner's `reason` field tells you **exactly which leg fired**:

| `reason` keyword | What you should do |
|------|-----|
| `does not carry recording-level group ids` | You skipped Prep. Redo it |
| `there is no held-out test set` | Test split was abandoned (too small). Need more data |
| `Test accuracy is ≥99% on only N samples` | Test set so small statistical noise dominates. Number not trustworthy |

### 6.4 Per-class metrics table

Per-class precision / recall / F1 / support. **Pay close attention to the `support` column** — it's how many test samples each class has. **Any class with support < 30 should not be taken seriously.**

---

## 7. My Models tab

### 7.1 Version chain

Every successful training run writes a `TrainingRun` row + a `ModelCheckpoint` (`v1.pt`, `v2.pt`, ...) to the database. Here you can see:

- Training history (timestamp, best val_acc, preset, warm-start parent)
- The currently active checkpoint (next training run defaults to warm-starting from this)
- Each version's input shape (`n_channels` / `n_classes` / `signal_length`)

### 7.2 Warm-start (Self-Improving)

In the Train tab, tick **Continue from previous model** to make the trainer:

1. Find the latest compatible checkpoint (matching channel count)
2. Load its feature-extractor weights (`features.*`)
3. Re-init the classifier head (`classifier.*`, since class count may differ)
4. Fine-tune on your new data

**Critical caveat**: warm-start **propagates leakage** — if v10 was trained with a leaky split (99 %+), v11 inherits the false confidence and after a few fine-tune epochs hits 99 % again.

> **Why does it sometimes say "Compatible vN found (validation accuracy 0.0%)"?**
> That's a previous run that aborted or crashed. **Definitely do NOT enable warm-start in this case** — it would poison the new run.

---

## 8. Worked example: 5-gesture OpenBCI EMG

> This section ties §1–§7 together using **real data**. All numbers reproduce on your machine.

### 8.1 Protocol background

| Item | Value |
|------|-------|
| Subject | LIU (single subject) |
| Gestures | CLENCH / FIVE / OK / ROCK / TWO |
| Device | OpenBCI Cyton 8-channel; only first 5 connected |
| Sample rate | 250 Hz |
| Per-gesture duration | 5-min continuous recording |
| Task structure | "5-second epochs, **first 1 s = active task, last 4 s = rest**" × 60 epochs |
| File format | OpenBCI GUI export `.txt` (`%`-prefixed header + 13 numeric columns) |

Data archive layout:

```
l.zip
└── l/
    ├── LIU-CLENCH.txt   ← 9.5 MB, ~75 000 rows
    ├── LIU-FIVE.txt
    ├── LIU-OK.txt
    ├── LIU-ROCK.txt
    └── LIU-TWO.txt
```

5 files **flat** under one folder, **not folder-per-class** → must use **Mode C**.

### 8.2 Step 1 — Launch

Double-click `BioSpark-Light.exe`, browser opens.

### 8.3 Step 2 — Data Prep

#### (a) Upload

1. Switch to **Data Prep**
2. Pick **Mode C — generic ZIP**
3. Drag `l.zip` into the upload zone

Inspect runs, OpenBCI header is detected, fields auto-filled:

| Field | Auto-filled |
|-------|-------------|
| Sampling Rate (Hz) | 250 |
| Multi-channel Columns | **`1-5`** ← skips SampleIndex / saturated / Accel automatically |

#### (b) Manually set 3 fields

| Field | Set to | Why |
|-------|--------|-----|
| Segment Length (sec) | **`1`** | First 1 s of each epoch is the active window |
| Stride (sec, optional) | **`5`** | One epoch every 5 s, skipping the 4-s rest |
| **Group-aware split unit** | **`Trial`** | One recording per class with 60 trials inside — must use trial-level grouping |

(Leave Overlap Ratio and Signal Column Index alone; defaults are correct.)

#### (c) File → label mapping

| File | Class label |
|------|-------------|
| `l/LIU-CLENCH.txt` | `CLENCH` |
| `l/LIU-FIVE.txt`   | `FIVE` |
| `l/LIU-OK.txt`     | `OK` |
| `l/LIU-ROCK.txt`   | `ROCK` |
| `l/LIU-TWO.txt`    | `TWO` |

#### (d) Click **Generate Training CSV**

Expected output:

```
Generated 305 samples across 5 classes.
```

| Dimension | Value |
|-----------|-------|
| Total samples | 305 (61 per class — recording is slightly > 5 min: `(75 767 − 250) / 1250 ≈ 60.4 → 61`) |
| Signal length | 250 (= 1 s × 250 Hz) |
| Channels | 5 (auto-detected) |
| `__group__` | 305 unique trial ids |
| Total columns | 1 252 (`ch1_1..ch5_250` + `label` + `__group__`) |

The first 10 preview rows all showing `CLENCH` is **expected** — the CSV is concatenated in file order, the first 61 rows are CLENCH. The trainer shuffles internally.

Click **Use for Training** to jump to the Train tab.

### 8.4 Step 3 — Train

#### (a) Pre-flight

The data preview should show **no yellow banner**. If it does → go back to Prep.

#### (b) Configuration

| Item | Pick / Fill |
|------|-------------|
| Preset | **🚀 Smart Auto** |
| Channels | Leave empty, or `5` |
| Auto-optimization | ✓ |
| Auto class weights | ✓ |
| Search optimal LR | ✗ |
| **Continue from previous model** | **✗ (critical!)** |

#### (c) Click **Start Training**

The first epoch-log line should be **green**:

```
✅ Split: group-aware · 305 groups · train=183 val=61 test=61
```

3–10 minutes to finish.

### 8.5 Step 4 — Read results

#### Confusion matrix (actual numbers from this run)

|  | CLENCH | FIVE | OK | ROCK | TWO |
|---|---|---|---|---|---|
| **CLENCH** | **91.7 %** | 0 | 8.3 % | 0 | 0 |
| **FIVE** | 0 | **100 %** | 0 | 0 | 0 |
| **OK** | 11.1 % | 0 | **88.9 %** | 0 | 0 |
| **ROCK** | 0 | 0 | 0 | **100 %** | 0 |
| **TWO** | 0 | 0 | 0 | 0 | **100 %** |

Overall accuracy ≈ **96.6 %** (57/59), and the leakage banner does **not** fire.

#### Why is this run healthy?

| Previous (leaky) | This run (honest) |
|---|---|
| All 5 classes at 100 % (no off-diagonal) | CLENCH ↔ OK confused ~10 % |
| t-SNE clusters separated by 4–10 units, zero overlap | Clusters show realistic overlap |
| Yellow leakage banner triggered | Banner not triggered (< 99 %) |

The CLENCH ↔ OK confusion is **biomechanically plausible** — both gestures involve thumb + index-finger flexion, recruiting overlapping superficial forearm flexors (which the EXG channels 1–5 roughly cover). The model is learning real motor-pattern differences, not time-series memorization.

### 8.6 Can the 96.6 % be reported in a paper?

**No.** Reasons:

| Dimension | Current | Publication-grade |
|---|---|---|
| Subjects | 1 | ≥ 5 |
| Sessions per subject | 1 / class | ≥ 4 / subject |
| Test-set size | ~12 / class | ≥ 30 / class |
| Split strategy | Trial-level random | Subject-level LeaveOneOut |

**Things you can say**: "BioSpark-Light pipeline reaches 96.6 % test accuracy on subject LIU's single-session data."
**Things you cannot say**: "Our method classifies 5 gestures at 96.6 % accuracy" (this implies generalization).

How to turn this into publishable accuracy → §10.

---

## 9. Core concepts cheat sheet

### 9.1 Data leakage

> **Windows from the same recording end up in BOTH train and val/test** → the model memorizes neighbors instead of learning the task → inflated accuracy.

Defense = group-aware split:

```python
sklearn.model_selection.GroupShuffleSplit
```

**Enabled by default in BioSpark-Light**, provided your data has the `__group__` column (Prep emits it automatically).

### 9.2 The `__group__` column

The "recording / trial id" of each sample. Used for splitting, **not** as model input.

| Prep mode | What `__group__` contains | Affected by `group_by`? |
|---|---|---|
| Mode A | File path | ✓ |
| Mode B | Interval index `interval_0`, `interval_1`, ... | ✓ |
| Mode C | File name | ✓ |

When `group_by=trial`, the value gets `<original>#trial_<i>` appended — every segment becomes independent.

### 9.3 Which `group_by` to pick

| Data shape | Pick |
|---|---|
| Multiple independent recordings per class | `recording` (default) |
| One long recording per class with multiple trials | `trial` |
| Pre-segmented CSV (already has `__group__`) | Either; depends on what's in the column |

### 9.4 Warm-start

Loads the previous checkpoint's feature-extractor weights and fine-tunes a new classifier head. **Use case**: data accumulates over time, classes occasionally change. **Pitfall**: if the previous model was trained with leakage, the warm-start carries the false confidence forward.

### 9.5 Class weighting

When classes are severely imbalanced (e.g. `100:5:100:5:100`), the loss is automatically weighted by inverse frequency to prevent the model from collapsing onto the majority class. Doesn't matter on a balanced 305-sample dataset.

### 9.6 Auto-optimization (what Smart Auto does internally)

Three things:

1. **Architecture selection**: kernel sizes and channel widths picked from sample count / channel count / signal length
2. **Class weighting** (if enabled)
3. **Early stopping**: when val_loss fails to improve for N consecutive epochs, training stops

---

## 10. Data-collection guidance

### 10.1 The single-session ceiling

No matter how perfectly you walk the Prep / Train flow, **single-subject + single-session** numbers **cannot serve as evidence of generalization** — the model may simply have memorized the impedance pattern of that one electrode application.

### 10.2 Recommended protocol: multi-session, multi-subject

| Dimension | Quantity |
|-----------|----------|
| **Subjects** | ≥ 5 |
| **Sessions per subject** | ≥ 4 (different days, re-electrode each time) |
| **Length per session** | 5 min is fine (60 trials per class) |

### 10.3 File-naming convention

```
data.zip
├── LIU-DAY1-CLENCH.txt
├── LIU-DAY1-FIVE.txt
├── LIU-DAY1-OK.txt
├── LIU-DAY1-ROCK.txt
├── LIU-DAY1-TWO.txt
├── LIU-DAY2-CLENCH.txt   ← different day, re-electrode
├── ...
├── WANG-DAY1-CLENCH.txt  ← different subject
├── ...
```

In Mode C, map every file to the same class:

```
LIU-DAY1-CLENCH.txt → CLENCH
LIU-DAY2-CLENCH.txt → CLENCH
LIU-DAY3-CLENCH.txt → CLENCH
WANG-DAY1-CLENCH.txt → CLENCH
...
```

### 10.4 Switch `group_by` back to `recording`

With multi-session data, **set `group_by` back to `recording` (the default)** — now each recording IS a genuinely independent trial, and GroupShuffleSplit will partition by session. The test set will land on "entirely unseen sessions". **That number can go in a paper.**

### 10.5 Subject-level / session-level Leave-One-Out

The strictest evaluation: LeaveOneSubjectOut (hold out an entire subject's data for test). The current build doesn't natively support this, but you can **manually run training multiple times, rotating which subject is the test subject**, and report mean ± std. Native support is on the v0.2 roadmap.

---

## 11. FAQ and troubleshooting

### Q1: My confusion matrix is all 100 %

**99 % of the time this is data leakage.** Check, in order:

1. Above the matrix, does it say **`Evaluation set: held-out test set`**? If not, you're on the leaky path
2. Did the yellow banner fire?
3. Did you upload a pre-segmented CSV (skipping Prep)?
4. Single-subject single-session data — even with Prep, t-SNE clusters separated cleanly mean "well-separable within this session" only; cross-session accuracy will likely crash to ~60 %

For the full diagnostic story, see [R&D-REPORT-2026-04-27.md appendix A](R&D-REPORT-2026-04-27.md).

### Q2: Curves stuck (loss not decreasing)

Most likely causes, by frequency:

1. **Channel count mismatch**: column names don't follow `ch{N}_M` but you set channels > 1 → reshape failed
2. **Pathological signal scaling**: BioSpark-Light per-sample-normalizes, but if your signal is in μV ≈ 1e5, numeric instability is possible
3. **Mistyped class label**: one of your 5 files has a typo in the label, the model can't learn that class
4. **Stride too large**: only 1–2 segments per class, not enough data

### Q3: Progress bar frozen / WebSocket disconnects

Possible causes:

- **PyInstaller build**: AV is intercepting; check the system tray for AV warnings
- **Source build**: look for tracebacks in the terminal
- **Either**: open the latest log file under `%LOCALAPPDATA%\BioSpark-Light\logs\`

### Q4: No banners appear in the browser

Hard-refresh (Ctrl-F5). If you've been editing source code, `frontend/js/trainer.js` is probably cached.

### Q5: OpenBCI `.txt` upload errors with "Failed to read"

Check the actual error. Common ones:

- File is actually `.gz`-compressed (OpenBCI GUI sometimes does this) → decompress and re-upload
- File was re-saved through Excel → encoding changed (GBK / UTF-8 BOM); export as UTF-8 plain
- File **only has the `%` header, no data rows** → device wasn't actually connected during recording

### Q6: I want to train on a different dataset

Just go back to Data Prep and upload a new ZIP. Old training records remain in My Models — **they don't get lost**. New training won't auto-warm-start from the old model unless you explicitly tick the box.

### Q7: Segment / stride values that aren't 1 s / 5 s

Adjust Segment Length and Stride to match your protocol. Example: "every 3 s, first 2 s is active" → `segment=2, stride=3`. **Constraint**: `stride ≥ segment` (use `overlap_ratio` for overlapping segments).

### Q8: My signal is continuous, no rest periods

Leave Stride empty and use Overlap Ratio (0–0.9). `group_by` should still typically be `recording` (one file = one group).

### Q9: Different OpenBCI channel counts

Adjust the Multi-channel Columns field directly:

| Device | Value |
|--------|-------|
| Cyton 4-channel | `1-4` |
| Cyton 8-channel | `1-8` |
| Cyton + Daisy 16-channel | `1-16` |
| Non-contiguous (only odd channels) | `1,3,5,7` |

### Q10: Can I use non-English class labels?

Yes. Chinese / Japanese / Korean labels work and display correctly in the confusion matrix and t-SNE plots.

---

## 12. Doc map

| Document | Purpose |
|----------|---------|
| [README.md](../README.md) | Project overview, install snapshot |
| [USER-MANUAL.md](USER-MANUAL.md) | Complete user manual (Chinese) |
| [USER-MANUAL.en.md](USER-MANUAL.en.md) | **This document** — full user manual (English) |
| [USAGE-OPENBCI.md](USAGE-OPENBCI.md) | OpenBCI-specific deep dive (extended version of §8) |
| [BUILD.md](BUILD.md) | PyInstaller packaging, for developers |
| [R&D-REPORT-2026-04-27.md](R&D-REPORT-2026-04-27.md) | R&D report — architecture decisions, data-leakage fix history |
| [PROJECT-STATUS-2026-04-27.md](PROJECT-STATUS-2026-04-27.md) | Project status snapshot |

### Where to look for...

| Topic | Doc |
|-------|-----|
| How to package the `.exe` | [BUILD.md](BUILD.md) |
| The full story of the data-leakage fix | [R&D-REPORT appendix A](R&D-REPORT-2026-04-27.md) |
| Valid range of a specific field | This manual §4 / §5 field tables |
| Filling in my OpenBCI data step-by-step | This manual §8 + [USAGE-OPENBCI.md](USAGE-OPENBCI.md) |
| Whether my training results are honest | This manual §6 + §10 |

---

*BioSpark-Light · v0.1.1 · User Manual · Compiled 2026-04-30*
*Co-authored by xjhveteran199-bit and Claude Opus 4.7. The real-data example (§8) uses subject LIU's 5-gesture OpenBCI Cyton recording.*
