"""Signal preprocessing pipeline for ECG, EEG, and EMG.

Uses NeuroKit2 for ECG/EMG and basic scipy for EEG.
Falls back to scipy-only processing if NeuroKit2/MNE are unavailable.
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample
from typing import Optional


def preprocess(data: np.ndarray, signal_type: str, sampling_rate: float,
               target_sr: Optional[float] = None, channel_idx: int = 0,
               model_id: Optional[str] = None) -> dict:
    """Preprocess biosignal data for model inference.

    Args:
        data: shape (n_channels, n_samples)
        signal_type: 'ecg', 'eeg', or 'emg'
        sampling_rate: original sampling rate in Hz
        target_sr: target sampling rate for resampling (None = no resampling)
        channel_idx: which channel to process (default: first)
        model_id: optional model ID to determine multi-channel requirements

    Returns:
        dict with:
            - segments: list of np.ndarray, each ready for model input
            - info: dict with preprocessing metadata
    """
    # Check if model requires multi-channel input
    from backend.config import MODEL_REGISTRY
    in_channels = 1
    if model_id and model_id in MODEL_REGISTRY:
        in_channels = MODEL_REGISTRY[model_id].get("in_channels", 1)

    if signal_type == "emg" and in_channels > 1:
        # Multi-channel EMG: use all available channels
        return _preprocess_emg_multichannel(data, sampling_rate, target_sr, in_channels)

    signal = data[channel_idx]

    if signal_type == "ecg":
        return _preprocess_ecg(signal, sampling_rate, target_sr)
    elif signal_type == "eeg":
        return _preprocess_eeg(signal, sampling_rate, target_sr)
    elif signal_type == "emg":
        return _preprocess_emg(signal, sampling_rate, target_sr)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def _bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                     fs: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def _resample_signal(signal: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    """Resample signal to target sampling rate."""
    if abs(orig_sr - target_sr) < 0.1:
        return signal
    n_target = int(len(signal) * target_sr / orig_sr)
    return resample(signal, n_target)


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    std = np.std(signal)
    if std < 1e-10:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def _segment(signal: np.ndarray, segment_length: int, overlap: float = 0.0) -> list[np.ndarray]:
    """Split signal into fixed-length segments.

    Args:
        signal: 1D array
        segment_length: number of samples per segment
        overlap: fraction of overlap between segments (0.0 - 0.9)
    """
    step = max(1, int(segment_length * (1 - overlap)))
    segments = []
    for start in range(0, len(signal) - segment_length + 1, step):
        seg = signal[start:start + segment_length]
        segments.append(seg)

    # If no complete segments, pad the signal
    if not segments and len(signal) > 0:
        padded = np.zeros(segment_length)
        padded[:len(signal)] = signal
        segments.append(padded)

    return segments


def _preprocess_ecg(signal: np.ndarray, sr: float, target_sr: Optional[float]) -> dict:
    """ECG preprocessing: filter → resample → normalize → segment into heartbeats."""
    # Bandpass filter 0.5-40 Hz
    filtered = _bandpass_filter(signal, 0.5, 40.0, sr)

    # Resample if needed
    effective_sr = sr
    if target_sr:
        filtered = _resample_signal(filtered, sr, target_sr)
        effective_sr = target_sr

    # Normalize
    normalized = _normalize(filtered)

    # Try R-peak detection with NeuroKit2, fall back to fixed segmentation
    try:
        import neurokit2 as nk
        _, rpeaks_info = nk.ecg_peaks(normalized, sampling_rate=effective_sr)
        rpeaks = rpeaks_info["ECG_R_Peaks"]

        # Extract heartbeat segments around R-peaks (187 samples centered on R-peak for MIT-BIH)
        seg_len = 187
        half = seg_len // 2
        segments = []
        for rpeak in rpeaks:
            start = rpeak - half
            end = start + seg_len
            if start >= 0 and end <= len(normalized):
                segments.append(normalized[start:end])

        if not segments:
            segments = _segment(normalized, seg_len)

        return {
            "segments": segments,
            "info": {
                "preprocessing": "bandpass(0.5-40Hz) → resample → normalize → R-peak segmentation",
                "n_rpeaks": len(rpeaks),
                "n_segments": len(segments),
                "segment_length": seg_len,
                "effective_sr": effective_sr,
            }
        }
    except (ImportError, Exception):
        # Fallback: fixed-length segmentation
        seg_len = 187
        segments = _segment(normalized, seg_len, overlap=0.5)
        return {
            "segments": segments,
            "info": {
                "preprocessing": "bandpass(0.5-40Hz) → resample → normalize → fixed segmentation",
                "n_segments": len(segments),
                "segment_length": seg_len,
                "effective_sr": effective_sr,
            }
        }


def _preprocess_eeg(signal: np.ndarray, sr: float, target_sr: Optional[float]) -> dict:
    """EEG preprocessing: filter → resample → normalize → epoch extraction."""
    # Bandpass filter 0.5-45 Hz
    filtered = _bandpass_filter(signal, 0.5, 45.0, sr)

    # Resample to 100 Hz (standard for sleep staging)
    effective_sr = target_sr or 100.0
    filtered = _resample_signal(filtered, sr, effective_sr)

    # Normalize
    normalized = _normalize(filtered)

    # 30-second epochs (standard for sleep staging)
    epoch_samples = int(30 * effective_sr)  # 3000 samples at 100 Hz
    segments = _segment(normalized, epoch_samples)

    return {
        "segments": segments,
        "info": {
            "preprocessing": "bandpass(0.5-45Hz) → resample → normalize → 30s epochs",
            "n_segments": len(segments),
            "segment_length": epoch_samples,
            "effective_sr": effective_sr,
        }
    }


def _preprocess_emg(signal: np.ndarray, sr: float, target_sr: Optional[float]) -> dict:
    """EMG preprocessing (single-channel): filter → rectify → normalize → segment."""
    # Bandpass filter 20-450 Hz
    highcut = min(450.0, sr * 0.49)  # Cannot exceed Nyquist
    filtered = _bandpass_filter(signal, 20.0, highcut, sr)

    # Full-wave rectification
    rectified = np.abs(filtered)

    # Resample if needed
    effective_sr = sr
    if target_sr:
        rectified = _resample_signal(rectified, sr, target_sr)
        effective_sr = target_sr

    # Normalize
    normalized = _normalize(rectified)

    # 400ms segments (standard for gesture recognition)
    seg_len = int(0.4 * effective_sr)
    segments = _segment(normalized, seg_len, overlap=0.5)

    return {
        "segments": segments,
        "info": {
            "preprocessing": "bandpass(20-450Hz) → rectify → resample → normalize → 400ms segments",
            "n_segments": len(segments),
            "segment_length": seg_len,
            "effective_sr": effective_sr,
        }
    }


def _preprocess_emg_multichannel(data: np.ndarray, sr: float,
                                  target_sr: Optional[float],
                                  required_channels: int) -> dict:
    """EMG preprocessing (multi-channel): process all channels, return (channels, time) segments.

    For the EMG gesture model which takes 16-channel input.
    """
    n_available = data.shape[0]
    effective_sr = target_sr or sr
    highcut = min(450.0, sr * 0.49)

    # Process each channel
    processed_channels = []
    for ch in range(min(n_available, required_channels)):
        signal = data[ch]
        # Bandpass filter
        filtered = _bandpass_filter(signal, 20.0, highcut, sr)
        # Rectify
        rectified = np.abs(filtered)
        # Resample
        if target_sr and abs(sr - target_sr) > 0.1:
            rectified = _resample_signal(rectified, sr, target_sr)
        # Normalize per-channel
        normalized = _normalize(rectified)
        processed_channels.append(normalized)

    # Pad with zeros if fewer channels than required
    if n_available < required_channels:
        min_len = len(processed_channels[0]) if processed_channels else 0
        for _ in range(required_channels - n_available):
            processed_channels.append(np.zeros(min_len, dtype=np.float32))

    # Stack: (n_channels, n_samples)
    multichannel = np.array(processed_channels, dtype=np.float32)

    # Segment into windows: each segment is (n_channels, seg_len)
    seg_len = int(0.4 * effective_sr)  # 400ms
    step = max(1, int(seg_len * 0.5))  # 50% overlap
    n_samples = multichannel.shape[1]

    segments = []
    for start in range(0, n_samples - seg_len + 1, step):
        seg = multichannel[:, start:start + seg_len]  # (channels, time)
        segments.append(seg)

    # Fallback: pad if no complete segments
    if not segments and n_samples > 0:
        padded = np.zeros((required_channels, seg_len), dtype=np.float32)
        padded[:, :min(n_samples, seg_len)] = multichannel[:, :min(n_samples, seg_len)]
        segments.append(padded)

    return {
        "segments": segments,
        "info": {
            "preprocessing": f"bandpass(20-{highcut:.0f}Hz) → rectify → resample → normalize → 400ms multi-ch segments",
            "n_segments": len(segments),
            "segment_length": seg_len,
            "n_channels": required_channels,
            "effective_sr": effective_sr,
            "multichannel": True,
        }
    }
