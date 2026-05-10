"""
features.py
Extract EEG features from a preprocessed chunk.

Features computed
-----------------
- Absolute band power  (delta, theta, alpha, beta, gamma)  per channel
- Relative band power  (normalised so all bands sum to 1)  per channel
- Channel-average band powers
- Key neuroscientific ratios:
    theta/alpha  (mental fatigue)
    alpha/beta   (relaxation index)
    (alpha+theta)/beta  (engagement index, inverted)
- Spectral entropy (Shannon entropy of the PSD)
"""

import numpy as np
from scipy.signal import welch
from config import SAMPLE_RATE, BANDS, CHANNELS, WELCH_NPERSEG, SMOOTHING_ALPHA


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Integrate PSD between [low, high] Hz using the trapezoidal rule."""
    idx = np.where((freqs >= low) & (freqs <= high))[0]
    if len(idx) == 0:
        return 0.0
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(trapz(psd[idx], freqs[idx]))


def _spectral_entropy(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Normalised Shannon entropy of the power spectrum."""
    psd_norm = psd / (psd.sum() + 1e-12)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    max_entropy = np.log2(len(psd_norm))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# EMA smoother (stateful, call once per pipeline run)
# ---------------------------------------------------------------------------

class BandPowerSmoother:
    """Exponential moving average over consecutive chunk band-power estimates."""

    def __init__(self, alpha: float = SMOOTHING_ALPHA):
        self.alpha = alpha
        self._state = None  # dict band → smoothed_value, initialised on first call

    def update(self, raw_powers: dict) -> dict:
        if self._state is None:
            self._state = dict(raw_powers)
            return dict(self._state)
        for band in raw_powers:
            self._state[band] = (
                self.alpha * raw_powers[band]
                + (1 - self.alpha) * self._state[band]
            )
        return dict(self._state)

    def reset(self):
        self._state = None


# Module-level smoother instance shared by the pipeline
_smoother = BandPowerSmoother()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_band_powers(data: np.ndarray) -> dict:
    """
    Compute per-channel and average band powers.

    Args:
        data: shape (n_channels, n_samples)

    Returns:
        dict with keys:
            'absolute'   : {band: {channel: power, ..., 'mean': power}}
            'relative'   : {band: {channel: rel_power, ..., 'mean': rel_power}}
            'entropy'    : {channel: entropy_value}
    """
    n_channels, n_samples = data.shape
    nperseg = min(WELCH_NPERSEG, n_samples)

    abs_powers  = {band: {} for band in BANDS}
    entropies   = {}

    for i, ch in enumerate(CHANNELS[:n_channels]):
        freqs, psd = welch(data[i], fs=SAMPLE_RATE, nperseg=nperseg)

        for band, (low, high) in BANDS.items():
            abs_powers[band][ch] = _band_power(freqs, psd, low, high)

        entropies[ch] = _spectral_entropy(freqs, psd)

    # Channel means
    for band in BANDS:
        vals = list(abs_powers[band].values())
        abs_powers[band]["mean"] = float(np.mean(vals)) if vals else 0.0

    # Relative powers (across all bands, per channel + mean)
    rel_powers = {band: {} for band in BANDS}
    for ch in CHANNELS[:n_channels]:
        total = sum(abs_powers[b][ch] for b in BANDS) + 1e-12
        for band in BANDS:
            rel_powers[band][ch] = abs_powers[band][ch] / total
    for band in BANDS:
        vals = [rel_powers[band][ch] for ch in CHANNELS[:n_channels]]
        rel_powers[band]["mean"] = float(np.mean(vals)) if vals else 0.0

    return {
        "absolute": abs_powers,
        "relative": rel_powers,
        "entropy":  entropies,
    }


def extract_ratios(rel_powers: dict) -> dict:
    """
    Compute neuroscientific band-ratio indices from relative powers.

    Args:
        rel_powers: the 'relative' sub-dict from extract_band_powers()

    Returns:
        dict with ratio names → float values (channel-mean based)
    """
    a = rel_powers["alpha"]["mean"]
    b = rel_powers["beta"]["mean"]
    t = rel_powers["theta"]["mean"]
    g = rel_powers["gamma"]["mean"]
    d = rel_powers["delta"]["mean"]

    return {
        "theta_alpha":        t / (a + 1e-12),      # fatigue / drowsiness
        "alpha_beta":         a / (b + 1e-12),      # relaxation index
        "engagement":         b / (a + t + 1e-12),  # cognitive engagement
        "theta_beta":         t / (b + 1e-12),      # ADHD-related ratio
        "alpha_dominance":    a,                     # raw alpha (relaxation)
        "beta_dominance":     b,                     # raw beta  (alertness/anxiety)
        "gamma_dominance":    g,                     # raw gamma (anxiety spike)
        "delta_dominance":    d,                     # raw delta (sleep pressure)
    }


def extract_features(preprocessed_chunk: dict, smooth: bool = True) -> dict:
    """
    Main feature extraction entry point.

    Args:
        preprocessed_chunk: dict from preprocessing.preprocess()
        smooth: apply EMA smoothing to band powers

    Returns:
        Feature dict containing band_powers, ratios, entropy, and
        a flat 'summary' dict of the most important scalars.
    """
    data = preprocessed_chunk["data"]
    bp   = extract_band_powers(data)

    if smooth:
        raw_means = {band: bp["relative"][band]["mean"] for band in BANDS}
        smoothed  = _smoother.update(raw_means)
        for band in BANDS:
            bp["relative"][band]["smoothed_mean"] = smoothed[band]
    else:
        for band in BANDS:
            bp["relative"][band]["smoothed_mean"] = bp["relative"][band]["mean"]

    ratios = extract_ratios(bp["relative"])

    # Flat summary for easy downstream consumption
    summary = {f"{band}_power": bp["relative"][band]["smoothed_mean"] for band in BANDS}
    summary.update(ratios)
    summary["spectral_entropy"] = float(np.mean(list(bp["entropy"].values())))

    return {
        "band_powers": bp,
        "ratios":      ratios,
        "summary":     summary,
        "channels":    preprocessed_chunk["channels"],
        "timestamps":  preprocessed_chunk["timestamps"],
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import FakeMuseGenerator
    from preprocessing  import preprocess

    gen = FakeMuseGenerator(state="relaxed")
    chunk = preprocess(gen.get_chunk())
    feats = extract_features(chunk)

    print("=== Band Powers (relative, smoothed mean) ===")
    for band in BANDS:
        val = feats["band_powers"]["relative"][band]["smoothed_mean"]
        print(f"  {band:6s}: {val:.4f}")

    print("\n=== Ratios ===")
    for k, v in feats["ratios"].items():
        print(f"  {k:25s}: {v:.4f}")

    print(f"\nSpectral entropy (mean): {feats['summary']['spectral_entropy']:.4f}")