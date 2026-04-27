"""
Publication-quality figure generation for scientific papers.

Renders figures using matplotlib with journal-specific styles
(Nature, IEEE, Science).  All functions return raw ``bytes``
(PNG at 300 DPI or SVG) suitable for direct download.
"""

import io
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Style presets (journal-specific)
# ---------------------------------------------------------------------------

STYLES: dict[str, dict[str, Any]] = {
    "nature": {
        "font_family": "Arial",
        "font_size": 8,
        "title_size": 10,
        "line_width": 0.75,
        "marker_size": 3,
        "dpi": 300,
        "colormap": "Blues",
        "fig_width_single": 3.5,    # inches (Nature single column ~89 mm)
        "fig_width_double": 7.0,    # inches (Nature double column ~183 mm)
        "colors": {
            "train": "#2563eb",      # blue
            "val": "#dc2626",        # red
            "accent": "#059669",     # green
            "grid": "#e5e7eb",
            "text": "#1f2937",
        },
    },
    "ieee": {
        "font_family": "Times New Roman",
        "font_size": 9,
        "title_size": 10,
        "line_width": 0.75,
        "marker_size": 3,
        "dpi": 300,
        "colormap": "YlOrRd",
        "fig_width_single": 3.5,    # inches (IEEE single column ~3.5 in)
        "fig_width_double": 7.16,   # inches (IEEE double column ~7.16 in)
        "colors": {
            "train": "#000000",
            "val": "#666666",
            "accent": "#333333",
            "grid": "#d1d5db",
            "text": "#111827",
        },
    },
    "science": {
        "font_family": "Helvetica",
        "font_size": 7,
        "title_size": 9,
        "line_width": 0.6,
        "marker_size": 2.5,
        "dpi": 300,
        "colormap": "RdBu_r",
        "fig_width_single": 2.3,    # inches (Science single column ~55 mm)
        "fig_width_double": 4.8,    # inches (Science double column ~120 mm)
        "colors": {
            "train": "#1a1a1a",
            "val": "#b91c1c",
            "accent": "#1d4ed8",
            "grid": "#e5e7eb",
            "text": "#1f2937",
        },
    },
}

# Category-10 palette for scatter / bar plots
CATEGORY_COLORS = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#c026d3", "#65a30d", "#ea580c", "#0f766e",
    "#4f46e5", "#e11d48", "#15803d", "#b45309", "#6d28d9",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_style(name: str) -> dict:
    return STYLES.get(name, STYLES["nature"])


def _apply_style(ax: plt.Axes, style: dict) -> None:
    """Apply publication style to an axes object."""
    for spine in ax.spines.values():
        spine.set_linewidth(style["line_width"])
        spine.set_color(style["colors"]["text"])

    ax.tick_params(
        width=style["line_width"],
        labelsize=style["font_size"],
        colors=style["colors"]["text"],
        direction="out",
        length=3,
    )
    ax.xaxis.label.set_fontsize(style["font_size"])
    ax.yaxis.label.set_fontsize(style["font_size"])
    ax.title.set_fontsize(style["title_size"])


def _setup_rc(style: dict) -> None:
    """Set global matplotlib rcParams from style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [style["font_family"], "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": style["font_size"],
        "axes.linewidth": style["line_width"],
        "pdf.fonttype": 42,   # TrueType (editable in Illustrator)
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # Keep text as text in SVG
        "figure.dpi": style["dpi"],
    })


def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    """Serialize a matplotlib Figure to PNG or SVG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Figure 1: Training Curves
# ---------------------------------------------------------------------------

def render_training_curves(
    history: list[dict],
    style_name: str = "nature",
    fmt: str = "png",
) -> bytes:
    """Loss + Accuracy dual-panel training curves.

    Two side-by-side subplots.  Annotates the best val_acc epoch with a
    vertical dashed marker.
    """
    style = _get_style(style_name)
    _setup_rc(style)
    colors = style["colors"]

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(style["fig_width_double"], style["fig_width_single"] * 0.55),
    )

    # Loss
    ax1.plot(epochs, train_loss, color=colors["train"], linewidth=style["line_width"],
             marker="o", markersize=style["marker_size"], label="Train Loss")
    ax1.plot(epochs, val_loss, color=colors["val"], linewidth=style["line_width"],
             linestyle="--", marker="s", markersize=style["marker_size"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("(a) Training & Validation Loss", fontsize=style["title_size"])
    ax1.legend(fontsize=style["font_size"] - 1, frameon=True, edgecolor="#d1d5db")
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    _apply_style(ax1, style)

    # Accuracy
    ax2.plot(epochs, train_acc, color=colors["train"], linewidth=style["line_width"],
             marker="o", markersize=style["marker_size"], label="Train Acc")
    ax2.plot(epochs, val_acc, color=colors["val"], linewidth=style["line_width"],
             linestyle="--", marker="s", markersize=style["marker_size"], label="Val Acc")

    # Mark best epoch
    best_idx = int(np.argmax(val_acc))
    best_ep = epochs[best_idx]
    best_val = val_acc[best_idx]
    ax2.axvline(x=best_ep, color=colors["accent"], linestyle=":", linewidth=0.8, alpha=0.7)
    ax2.annotate(
        f"Best: {best_val:.3f}",
        xy=(best_ep, best_val),
        xytext=(best_ep + max(1, len(epochs) * 0.05), best_val - 0.02),
        fontsize=style["font_size"] - 1,
        arrowprops=dict(arrowstyle="->", color=colors["accent"], lw=0.8),
        color=colors["accent"],
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("(b) Training & Validation Accuracy", fontsize=style["title_size"])
    ax2.legend(fontsize=style["font_size"] - 1, frameon=True, edgecolor="#d1d5db")
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    _apply_style(ax2, style)

    fig.tight_layout()
    return fig_to_bytes(fig, fmt, style["dpi"])


# ---------------------------------------------------------------------------
# Figure 2: Confusion Matrix
# ---------------------------------------------------------------------------

def render_confusion_matrix(
    cm_data: dict,
    mode: str = "both",
    style_name: str = "nature",
    fmt: str = "png",
) -> bytes:
    """Confusion matrix heatmap(s).

    *mode* controls the layout:
        - ``"count"``: single heatmap with raw counts
        - ``"normalized"``: single heatmap with row-normalized percentages
        - ``"both"``: side-by-side count and normalized matrices
    """
    style = _get_style(style_name)
    _setup_rc(style)

    matrix = np.array(cm_data["matrix"])
    class_names = cm_data["class_names"]
    n = len(class_names)

    # Truncate long class names for readability
    short_names = [c[:12] + ".." if len(c) > 14 else c for c in class_names]

    # Row-normalized matrix
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(matrix, row_sums, where=row_sums > 0, out=np.zeros_like(matrix, dtype=float))

    if mode == "both":
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(style["fig_width_double"], style["fig_width_single"] * 0.9),
        )
        _draw_cm_heatmap(ax1, matrix, short_names, style, is_normalized=False, title="(a) Count")
        _draw_cm_heatmap(ax2, norm_matrix, short_names, style, is_normalized=True, title="(b) Normalized (%)")
    elif mode == "normalized":
        fig, ax = plt.subplots(figsize=(style["fig_width_single"], style["fig_width_single"] * 0.9))
        _draw_cm_heatmap(ax, norm_matrix, short_names, style, is_normalized=True, title="Confusion Matrix (Normalized)")
    else:
        fig, ax = plt.subplots(figsize=(style["fig_width_single"], style["fig_width_single"] * 0.9))
        _draw_cm_heatmap(ax, matrix, short_names, style, is_normalized=False, title="Confusion Matrix")

    fig.tight_layout()
    return fig_to_bytes(fig, fmt, style["dpi"])


def _draw_cm_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    class_names: list[str],
    style: dict,
    is_normalized: bool = False,
    title: str = "",
) -> None:
    """Draw a single confusion matrix heatmap on the given axes."""
    n = len(class_names)
    cmap = plt.get_cmap(style["colormap"])

    im = ax.imshow(matrix, cmap=cmap, aspect="equal")

    # Annotate cells
    thresh = matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if is_normalized:
                text = f"{val:.1%}" if val > 0 else "0"
            else:
                text = str(int(val))
            color = "white" if val > thresh else style["colors"]["text"]
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=max(5, style["font_size"] - max(0, n - 6)),
                    color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right",
                       fontsize=max(5, style["font_size"] - max(0, n - 6)))
    ax.set_yticklabels(class_names,
                       fontsize=max(5, style["font_size"] - max(0, n - 6)))
    ax.set_xlabel("Predicted", fontsize=style["font_size"])
    ax.set_ylabel("True", fontsize=style["font_size"])
    ax.set_title(title, fontsize=style["title_size"])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=style["font_size"] - 1)

    _apply_style(ax, style)


# ---------------------------------------------------------------------------
# Figure 3: t-SNE Feature Visualization
# ---------------------------------------------------------------------------

def render_tsne(
    tsne_data: dict,
    style_name: str = "nature",
    fmt: str = "png",
) -> bytes:
    """t-SNE scatter plot, color-coded by class with convex hulls."""
    style = _get_style(style_name)
    _setup_rc(style)

    x = np.array(tsne_data["x"])
    y = np.array(tsne_data["y"])
    labels = tsne_data["labels"]
    class_names = tsne_data["class_names"]

    fig, ax = plt.subplots(figsize=(style["fig_width_single"] * 1.3, style["fig_width_single"] * 1.1))

    for i, cls in enumerate(class_names):
        mask = [l == cls for l in labels]
        cx = x[mask]
        cy = y[mask]
        color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]

        ax.scatter(cx, cy, s=15, alpha=0.7, color=color, label=cls,
                   edgecolors="white", linewidths=0.3, zorder=3)

        # Convex hull (only if enough points)
        if len(cx) >= 3:
            try:
                from scipy.spatial import ConvexHull
                points = np.column_stack([cx, cy])
                hull = ConvexHull(points)
                hull_pts = points[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], color=color,
                        linewidth=0.6, alpha=0.4, linestyle="--")
                ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.05)
            except Exception:
                pass  # degenerate hull (collinear points)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=style["font_size"])
    ax.set_ylabel("t-SNE Dimension 2", fontsize=style["font_size"])
    ax.set_title("t-SNE Feature Space Visualization", fontsize=style["title_size"])

    # Legend outside the plot
    n_cols = min(len(class_names), 4)
    ax.legend(
        fontsize=max(5, style["font_size"] - 1),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=n_cols,
        frameon=True,
        edgecolor="#d1d5db",
        columnspacing=0.8,
        handletextpad=0.3,
    )

    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    _apply_style(ax, style)

    fig.tight_layout()
    return fig_to_bytes(fig, fmt, style["dpi"])


# ---------------------------------------------------------------------------
# Figure 4: Per-Class Metrics Bar Chart
# ---------------------------------------------------------------------------

def render_per_class_metrics(
    cm_data: dict,
    style_name: str = "nature",
    fmt: str = "png",
) -> bytes:
    """Grouped bar chart of Precision, Recall, F1 per class."""
    style = _get_style(style_name)
    _setup_rc(style)

    per_class = cm_data["per_class"]
    class_names = [c["class"] for c in per_class]
    short_names = [c[:10] + ".." if len(c) > 12 else c for c in class_names]
    n = len(class_names)

    precision = [c["precision"] for c in per_class]
    recall = [c["recall"] for c in per_class]
    f1 = [c["f1"] for c in per_class]

    x = np.arange(n)
    bar_w = 0.25

    fig_h = max(style["fig_width_single"] * 0.6, n * 0.25)
    fig, ax = plt.subplots(figsize=(style["fig_width_double"], fig_h))

    bars1 = ax.bar(x - bar_w, precision, bar_w, label="Precision", color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x, recall, bar_w, label="Recall", color="#dc2626", alpha=0.85)
    bars3 = ax.bar(x + bar_w, f1, bar_w, label="F1-Score", color="#059669", alpha=0.85)

    # Value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}",
                    ha="center", va="bottom",
                    fontsize=max(4, style["font_size"] - 2),
                    color=style["colors"]["text"],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right",
                       fontsize=max(5, style["font_size"] - max(0, n - 8)))
    ax.set_ylabel("Score", fontsize=style["font_size"])
    ax.set_title("Per-Class Classification Metrics", fontsize=style["title_size"])
    ax.set_ylim(0, min(1.15, max(max(precision), max(recall), max(f1)) + 0.15))
    ax.legend(fontsize=style["font_size"] - 1, frameon=True, edgecolor="#d1d5db")
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    _apply_style(ax, style)

    # Add overall accuracy line
    acc = cm_data.get("accuracy", 0)
    ax.axhline(y=acc, color=style["colors"]["accent"], linestyle="--",
               linewidth=0.8, alpha=0.7, label=f"Overall Acc: {acc:.3f}")
    ax.legend(fontsize=style["font_size"] - 1, frameon=True, edgecolor="#d1d5db")

    fig.tight_layout()
    return fig_to_bytes(fig, fmt, style["dpi"])


# ---------------------------------------------------------------------------
# Figure 5: Model Architecture Diagram
# ---------------------------------------------------------------------------

def _compute_layer_shapes(model, input_shape: tuple) -> list[dict]:
    """Register forward hooks, run a dummy tensor, collect output shapes."""
    import torch

    shapes = []
    hooks = []

    def _make_hook(name, layer_type):
        def hook(module, inp, out):
            out_shape = tuple(out.shape) if hasattr(out, "shape") else "?"
            shapes.append({
                "name": name,
                "type": layer_type,
                "output_shape": out_shape,
            })
        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        layer_type = module.__class__.__name__
        hooks.append(module.register_forward_hook(_make_hook(name, layer_type)))

    dummy = torch.zeros(1, *input_shape)
    with torch.no_grad():
        model.eval()
        model(dummy)

    for h in hooks:
        h.remove()

    return shapes


def render_architecture_diagram(
    model,
    input_shape: tuple,
    style_name: str = "nature",
    fmt: str = "png",
) -> bytes:
    """Render a CNN architecture diagram using matplotlib patches.

    Each Conv+BN+ReLU+Pool combination is drawn as a single block with
    stacked labels.  Arrows connect blocks left-to-right.
    """
    style = _get_style(style_name)
    _setup_rc(style)

    layer_shapes = _compute_layer_shapes(model, input_shape)

    # Group sequential layers into blocks
    blocks = _group_into_blocks(layer_shapes, input_shape)

    # Color mapping
    TYPE_COLORS = {
        "Conv1d": "#3b82f6",
        "BatchNorm1d": "#9ca3af",
        "ReLU": "#22c55e",
        "MaxPool1d": "#f59e0b",
        "AdaptiveAvgPool1d": "#f59e0b",
        "Dropout": "#d1d5db",
        "Linear": "#8b5cf6",
    }

    n_blocks = len(blocks)
    block_w = 1.2
    block_gap = 0.6
    total_w = n_blocks * block_w + (n_blocks - 1) * block_gap
    fig_w = max(style["fig_width_double"], total_w * 0.85 + 1.5)
    fig_h = 3.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, total_w + 0.5)
    ax.set_ylim(-1.5, 3.0)
    ax.axis("off")
    ax.set_title("Model Architecture", fontsize=style["title_size"], pad=10)

    for i, block in enumerate(blocks):
        x_center = i * (block_w + block_gap) + block_w / 2
        x_left = x_center - block_w / 2

        layers = block["layers"]
        block_h = max(1.8, len(layers) * 0.35 + 0.4)
        y_bottom = (3.0 - block_h) / 2 - 0.2

        # Block background
        main_type = layers[0]["type"] if layers else "Linear"
        bg_color = TYPE_COLORS.get(main_type, "#e5e7eb")

        rect = mpatches.FancyBboxPatch(
            (x_left, y_bottom), block_w, block_h,
            boxstyle="round,pad=0.08",
            facecolor=bg_color,
            edgecolor="#374151",
            linewidth=style["line_width"],
            alpha=0.2,
        )
        ax.add_patch(rect)

        # Border
        rect_border = mpatches.FancyBboxPatch(
            (x_left, y_bottom), block_w, block_h,
            boxstyle="round,pad=0.08",
            facecolor="none",
            edgecolor="#374151",
            linewidth=style["line_width"] * 1.5,
        )
        ax.add_patch(rect_border)

        # Layer labels stacked inside block
        n_layers = len(layers)
        for j, layer in enumerate(layers):
            y_pos = y_bottom + block_h - (j + 1) * (block_h / (n_layers + 1))
            color = TYPE_COLORS.get(layer["type"], "#6b7280")

            # Compose label text
            label = layer["type"]
            if "params" in layer:
                label += f"\n{layer['params']}"

            ax.text(
                x_center, y_pos, label,
                ha="center", va="center",
                fontsize=max(5, style["font_size"] - 1),
                fontweight="bold" if layer["type"] in ("Conv1d", "Linear") else "normal",
                color="#1f2937",
            )

        # Shape annotation below block
        shape_text = block.get("output_shape", "")
        if shape_text:
            ax.text(
                x_center, y_bottom - 0.25, shape_text,
                ha="center", va="top",
                fontsize=style["font_size"] - 1,
                color="#6b7280",
                fontstyle="italic",
            )

        # Block title above
        ax.text(
            x_center, y_bottom + block_h + 0.15, block["title"],
            ha="center", va="bottom",
            fontsize=style["font_size"],
            fontweight="bold",
            color="#374151",
        )

        # Arrow to next block
        if i < n_blocks - 1:
            ax.annotate(
                "",
                xy=(x_left + block_w + block_gap * 0.1, (y_bottom + block_h / 2)),
                xytext=(x_left + block_w + block_gap * 0.9, (y_bottom + block_h / 2)),
                arrowprops=dict(
                    arrowstyle="<-",
                    color="#6b7280",
                    lw=style["line_width"] * 1.5,
                ),
            )

    # Input shape annotation
    ax.text(
        -0.3, (3.0 - 1.8) / 2 + 0.7,
        f"Input\n{input_shape}",
        ha="center", va="center",
        fontsize=style["font_size"],
        color="#6b7280",
        fontstyle="italic",
    )

    fig.tight_layout()
    return fig_to_bytes(fig, fmt, style["dpi"])


def _group_into_blocks(layer_shapes: list[dict], input_shape: tuple) -> list[dict]:
    """Group flat layer list into logical blocks for the architecture diagram."""
    blocks = []
    current_layers = []
    current_title = ""
    block_idx = 0

    # Classify layers into conv blocks vs classifier layers
    conv_block_types = {"Conv1d", "BatchNorm1d", "ReLU", "MaxPool1d", "AdaptiveAvgPool1d"}
    classifier_types = {"Linear", "Dropout", "ReLU"}

    last_shape = None
    in_classifier = False

    for info in layer_shapes:
        ltype = info["type"]
        shape = info["output_shape"]

        # Build param description
        params = ""
        if ltype == "Conv1d":
            # Extract from shape: (B, C_out, L)
            if isinstance(shape, tuple) and len(shape) == 3:
                params = f"→ {shape[1]}ch"
        elif ltype == "Linear":
            if isinstance(shape, tuple) and len(shape) == 2:
                params = f"→ {shape[1]}"
        elif ltype == "MaxPool1d":
            params = "k=2"
        elif ltype == "AdaptiveAvgPool1d":
            params = "→ 1"
        elif ltype == "Dropout":
            params = ""

        layer_entry = {"type": ltype, "params": params}

        # Detect block boundaries
        if ltype == "Conv1d" and current_layers:
            # Start of a new conv block — flush current
            if current_layers:
                shape_str = ""
                if last_shape and isinstance(last_shape, tuple):
                    shape_str = "×".join(str(s) for s in last_shape[1:])
                blocks.append({
                    "title": current_title or f"Block {block_idx}",
                    "layers": current_layers,
                    "output_shape": shape_str,
                })
                block_idx += 1
                current_layers = []

        if ltype == "Conv1d":
            current_title = f"Conv Block {block_idx + 1}"
        elif ltype == "Linear" and not in_classifier:
            # First Linear — flush conv block, start classifier
            if current_layers:
                shape_str = ""
                if last_shape and isinstance(last_shape, tuple):
                    shape_str = "×".join(str(s) for s in last_shape[1:])
                blocks.append({
                    "title": current_title or f"Block {block_idx}",
                    "layers": current_layers,
                    "output_shape": shape_str,
                })
                block_idx += 1
                current_layers = []
            current_title = "Classifier"
            in_classifier = True

        current_layers.append(layer_entry)
        last_shape = shape

    # Flush remaining
    if current_layers:
        shape_str = ""
        if last_shape and isinstance(last_shape, tuple):
            shape_str = "×".join(str(s) for s in last_shape[1:])
        blocks.append({
            "title": current_title or f"Block {block_idx}",
            "layers": current_layers,
            "output_shape": shape_str,
        })

    return blocks
