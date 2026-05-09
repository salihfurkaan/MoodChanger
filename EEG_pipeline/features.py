import numpy as np
from scipy.signal import welch
from config import BANDS, SFREQ, EPSILON


def compute_band_powers(window):
    """
    window: (channels, samples)
    returns: dict per channel
    """
    features = []

    for ch_data in window:
        freqs, psd = welch(ch_data, fs=SFREQ, nperseg=256)

        band_powers = {}

        total_power = np.sum(psd)

        for band, (low, high) in BANDS.items():
            idx = (freqs >= low) & (freqs <= high)
            power = np.sum(psd[idx])
            band_powers[band] = power

        # Relative powers
        for band in band_powers:
            band_powers[f"rel_{band}"] = band_powers[band] / (total_power + EPSILON)

        features.append(band_powers)

    return features


def compute_focus_score(band_dict):
    """
    Beta / (Alpha + Theta)
    """
    return band_dict["beta"] / (band_dict["alpha"] + band_dict["theta"] + EPSILON)