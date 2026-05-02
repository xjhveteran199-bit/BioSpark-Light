"""Unit tests for the Signal1DCNNTransformer hybrid model."""

import torch

from backend.services.trainer import Signal1DCNN, Signal1DCNNTransformer


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_forward_shape_singlechannel():
    model = Signal1DCNNTransformer(n_classes=5, in_channels=1)
    x = torch.randn(4, 1, 512)
    out = model(x)
    assert out.shape == (4, 5)


def test_forward_shape_multichannel():
    model = Signal1DCNNTransformer(n_classes=3, in_channels=8)
    x = torch.randn(2, 8, 1024)
    out = model(x)
    assert out.shape == (2, 3)


def test_param_count_in_target_range():
    """Hybrid should be heavier than CNN baseline but still 'lightweight'."""
    cnn = Signal1DCNN(n_classes=5, in_channels=1)
    hybrid = Signal1DCNNTransformer(n_classes=5, in_channels=1)
    cnn_params = _count_params(cnn)
    hybrid_params = _count_params(hybrid)
    assert hybrid_params > cnn_params, "hybrid should have more capacity than CNN baseline"
    assert 50_000 <= hybrid_params <= 500_000, (
        f"hybrid param count {hybrid_params} outside target 50K-500K range"
    )


def test_extract_features_dimension():
    model = Signal1DCNNTransformer(n_classes=4, in_channels=2)
    x = torch.randn(3, 2, 256)
    feats = model.extract_features(x)
    assert feats.ndim == 2
    assert feats.shape[0] == 3
    assert feats.shape[1] == model._feat_dim


def test_backward_pass_no_error():
    model = Signal1DCNNTransformer(n_classes=5, in_channels=4)
    x = torch.randn(8, 4, 512)
    y = torch.randint(0, 5, (8,))
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_arch_config_overrides_defaults():
    cfg = {
        "kernel_sizes": [5, 3],
        "channels": [16, 32],
        "d_model": 64,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 128,
        "transformer_dropout": 0.0,
        "dropout1": 0.0,
        "dropout2": 0.0,
        "fc_hidden": 32,
        "max_tokens": 256,
    }
    model = Signal1DCNNTransformer(n_classes=2, in_channels=1, arch_config=cfg)
    x = torch.randn(2, 1, 256)
    out = model(x)
    assert out.shape == (2, 2)
    # smaller config should produce fewer params than default
    default = Signal1DCNNTransformer(n_classes=2, in_channels=1)
    assert _count_params(model) < _count_params(default)


def test_long_sequence_truncates_to_max_tokens():
    """Sequence longer than max_tokens after stem should be truncated, not crash."""
    cfg = dict(max_tokens=32)  # very small token budget
    model = Signal1DCNNTransformer(n_classes=3, in_channels=1, arch_config=cfg)
    # 1024 / 4 (two MaxPools) = 256 tokens > max_tokens=32 → truncated
    x = torch.randn(1, 1, 1024)
    out = model(x)
    assert out.shape == (1, 3)
