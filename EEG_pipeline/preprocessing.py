"""
preprocessing.py
Signal-processing steps applied to raw EEG chunks before feature extraction.

Steps
-----
1. Notch filter     – remove power-line interference (50 or 60 Hz)
2. Band-pass filter – keep only the EEG-relevant frequency range
3. Artifact rejection – zero-out / flag epochs with extreme amplitude
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt
from config import (
    SAMPLE_RATE, NOTCH_FREQ, NOTCH_Q,
    BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    ARTIFACT_THRESH,
)


# ---------------------------------------------------------------------------
# Filter builders (cached as module-level constants for efficiency)
# ---------------------------------------------------------------------------

def _build_notch(freq: float, q: float, fs: int):
    b, a = iirnotch(freq, q, fs)
    # Convert ba → sos for numerical stability
    from scipy.signal import tf2sos
    return tf2sos(b, a)


def _build_bandpass(low: float, high: float, order: int, fs: int):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


_NOTCH_SOS    = _build_notch(NOTCH_FREQ, NOTCH_Q, SAMPLE_RATE)
_BANDPASS_SOS = _build_bandpass(BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def notch_filter(data: np.ndarray) -> np.ndarray:
    """
    Apply a notch filter to remove power-line noise.

    Args:
        data: shape (n_channels, n_samples)

    Returns:
        Filtered array, same shape.
    """
    # sosfiltfilt needs at least 3 × padlen samples; skip if chunk too short
    min_len = 3 * (2 * len(_NOTCH_SOS) + 1)
    if data.shape[1] < min_len:
        return data
    return sosfiltfilt(_NOTCH_SOS, data, axis=1)


def bandpass_filter(data: np.ndarray) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter.

    Args:
        data: shape (n_channels, n_samples)

    Returns:
        Filtered array, same shape.
    """
    min_len = 3 * (2 * len(_BANDPASS_SOS) + 1)
    if data.shape[1] < min_len:
        return data
    return sosfiltfilt(_BANDPASS_SOS, data, axis=1)


def reject_artifacts(data: np.ndarray, threshold: float = ARTIFACT_THRESH):
    """
    Simple peak-to-peak artifact rejection.
    Channels whose signal exceeds `threshold` µV in the chunk are marked bad.

    Args:
        data      : shape (n_channels, n_samples)
        threshold : amplitude limit in µV

    Returns:
        clean_data : same shape, bad channels replaced with zeros
        bad_mask   : bool array of shape (n_channels,), True = bad
    """
    peak_to_peak = data.max(axis=1) - data.min(axis=1)
    bad_mask = peak_to_peak > threshold
    clean_data = data.copy()
    clean_data[bad_mask] = 0.0
    return clean_data, bad_mask


def preprocess(raw_chunk: dict) -> dict:
    """
    Full preprocessing pipeline applied to one chunk dict.

    Args:
        raw_chunk: dict produced by FakeMuseGenerator.get_chunk()
                   Keys: 'data', 'timestamps', 'channels', 'state'

    Returns:
        Enriched dict with added keys:
            'data'       : preprocessed EEG (n_channels, n_samples)
            'bad_channels': list of channel names flagged as artifacts
    """
    data = raw_chunk["data"].copy()          # (n_channels, n_samples)

    # 1. Notch filter
    data = notch_filter(data)

    # 2. Band-pass filter
    data = bandpass_filter(data)

    # 3. Artifact rejection
    data, bad_mask = reject_artifacts(data)

    bad_channels = [ch for ch, bad in zip(raw_chunk["channels"], bad_mask) if bad]

    return {
        **raw_chunk,
        "data": data,
        "bad_channels": bad_channels,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import FakeMuseGenerator

    gen = FakeMuseGenerator(state="relaxed")
    raw = gen.get_chunk()
    processed = preprocess(raw)

    print("Raw      mean abs:", np.abs(raw["data"]).mean().round(3), "µV")
    print("Processed mean abs:", np.abs(processed["data"]).mean().round(3), "µV")
    print("Bad channels:", processed["bad_channels"] or "none")