"""Unit tests for recommend_architecture() decision rules."""

import numpy as np

from backend.services.auto_optimizer import (
    DataQualityAssessor,
    recommend_architecture,
)


def test_few_samples_picks_cnn():
    rec = recommend_architecture(n_samples=200, n_channels=4, signal_length=512, n_groups=10)
    assert rec["recommended"] == "cnn"
    assert "样本" in rec["reason_zh"]


def test_few_groups_picks_cnn():
    # 800 samples but only 3 subjects → still cnn
    rec = recommend_architecture(n_samples=800, n_channels=4, signal_length=512, n_groups=3)
    assert rec["recommended"] == "cnn"


def test_short_sequence_picks_cnn():
    rec = recommend_architecture(n_samples=2000, n_channels=8, signal_length=64, n_groups=20)
    assert rec["recommended"] == "cnn"
    assert rec["reason_zh"].find("长") >= 0 or "短" in rec["reason_zh"]


def test_multichannel_long_sequence_picks_hybrid():
    rec = recommend_architecture(n_samples=1500, n_channels=8, signal_length=1024, n_groups=12)
    assert rec["recommended"] == "hybrid"
    assert rec["confidence"] >= 0.7


def test_very_many_samples_picks_hybrid_even_single_channel():
    # 3000 samples with 1 channel + 256 length → falls into n_samples >= 2000 branch
    rec = recommend_architecture(n_samples=3000, n_channels=1, signal_length=256, n_groups=20)
    assert rec["recommended"] == "hybrid"


def test_middle_range_falls_back_to_cnn():
    # 700 samples, 1 channel, 256 length, 10 groups — neither extreme
    rec = recommend_architecture(n_samples=700, n_channels=1, signal_length=256, n_groups=10)
    assert rec["recommended"] == "cnn"


def test_recommendation_contains_required_fields():
    rec = recommend_architecture(n_samples=1000, n_channels=4, signal_length=512, n_groups=8)
    for k in ("recommended", "confidence", "reason_zh", "reason_en", "data_profile", "alternatives"):
        assert k in rec
    assert rec["data_profile"]["n_samples"] == 1000
    assert len(rec["alternatives"]) == 2


def test_data_quality_assessor_includes_arch_recommendation():
    rng = np.random.default_rng(0)
    n_samples, total = 600, 8 * 256
    X = rng.standard_normal((n_samples, total)).astype(np.float32)
    y = rng.integers(0, 5, size=n_samples).astype(np.int64)
    groups = np.array([f"sub_{i % 12}" for i in range(n_samples)], dtype=object)

    result = DataQualityAssessor().assess(
        X, y, class_names=[f"c{i}" for i in range(5)],
        n_channels=8, groups=groups,
    )
    assert "architecture_recommendation" in result
    assert "data_profile" in result
    profile = result["data_profile"]
    assert profile["n_channels"] == 8
    assert profile["signal_length"] == 256
    assert profile["n_groups"] == 12
    assert result["architecture_recommendation"]["recommended"] in {"cnn", "hybrid"}
