"""
EEG Synthetic Data Generator
===========================

Generates multi-channel EEG-like signals with controllable brain states.
Designed for validating signal processing pipelines:
- Filtering
- Windowing
- PSD / band power extraction

Author: You (but now done properly)
"""

import numpy as np


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2"]

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


STATE_PROFILES = {
    "focused": {
        "alpha": 0.6,
        "beta": 1.2,
        "theta": 0.3,
        "delta": 0.2,
    },
    "drowsy": {
        "alpha": 0.5,
        "beta": 0.3,
        "theta": 1.2,
        "delta": 0.8,
    },
    "neutral": {
        "alpha": 0.8,
        "beta": 0.7,
        "theta": 0.6,
        "delta": 0.4,
    }
}


# ─────────────────────────────────────────────
# CORE SIGNAL GENERATION
# ─────────────────────────────────────────────

def _band_signal(freq_range, t, amplitude):
    """Generate signal within a frequency band."""
    low, high = freq_range
    freq = np.random.uniform(low, high)
    phase = np.random.uniform(0, 2 * np.pi)
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def generate_eeg_channel(state, t):
    """Generate one EEG channel signal."""
    profile = STATE_PROFILES[state]

    signal = np.zeros_like(t)

    for band, amp in profile.items():
        signal += _band_signal(BANDS[band], t, amp)

    # Add realistic noise
    noise = np.random.normal(0, 0.5, len(t))

    # Add slow drift (baseline wander)
    drift = 0.2 * np.sin(2 * np.pi * 0.2 * t)

    return signal + noise + drift


def generate_multichannel_eeg(
    duration_sec=60,
    sfreq=160,
    channels=None,
    state="neutral",
    seed=42
):
    """
    Generate multi-channel EEG data.

    Returns:
        dict:
            "data": np.array (n_channels, n_samples)
            "channels": list
            "sfreq": int
            "state": str
    """

    np.random.seed(seed)

    channels = channels or DEFAULT_CHANNELS
    n_samples = duration_sec * sfreq

    t = np.linspace(0, duration_sec, n_samples)

    data = []

    for ch in channels:
        ch_signal = generate_eeg_channel(state, t)

        # Slight channel variability
        ch_signal *= np.random.uniform(0.9, 1.1)

        data.append(ch_signal)

    data = np.array(data)

    return {
        "data": data,
        "channels": channels,
        "sfreq": sfreq,
        "state": state
    }


# ─────────────────────────────────────────────
# SEGMENTED / STATE-CHANGING DATA
# ─────────────────────────────────────────────

def generate_state_sequence(
    sequence,
    segment_duration=10,
    sfreq=160,
    channels=None
):
    """
    Generate EEG data with changing mental states.

    Example:
        sequence = ["neutral", "focused", "drowsy"]
    """

    all_data = []
    labels = []

    for state in sequence:
        segment = generate_multichannel_eeg(
            duration_sec=segment_duration,
            sfreq=sfreq,
            channels=channels,
            state=state,
            seed=np.random.randint(0, 10000)
        )

        all_data.append(segment["data"])
        labels.extend([state] * (segment_duration * sfreq))

    combined = np.concatenate(all_data, axis=1)

    return {
        "data": combined,
        "channels": channels or DEFAULT_CHANNELS,
        "sfreq": sfreq,
        "labels": labels
    }


# ─────────────────────────────────────────────
# OPTIONAL: MNE COMPATIBILITY
# ─────────────────────────────────────────────

def to_mne_raw(eeg_dict):
    """
    Convert synthetic EEG to MNE Raw object.
    Requires mne installed.
    """

    import mne

    info = mne.create_info(
        ch_names=eeg_dict["channels"],
        sfreq=eeg_dict["sfreq"],
        ch_types="eeg"
    )

    raw = mne.io.RawArray(eeg_dict["data"], info)

    return raw


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Generating synthetic EEG...")

    eeg = generate_state_sequence(
        sequence=["neutral", "focused", "drowsy"],
        segment_duration=20
    )

    print(f"Shape: {eeg['data'].shape}")
    print(f"Channels: {eeg['channels']}")
    print(f"Sampling rate: {eeg['sfreq']} Hz")

    try:
        raw = to_mne_raw(eeg)
        print(raw)
    except ImportError:
        print("MNE not installed, skipping Raw conversion.")