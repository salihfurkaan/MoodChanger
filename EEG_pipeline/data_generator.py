"""
data_generator.py
Generates synthetic EEG data mimicking a Muse headband output.
Channels: TP9, AF7, AF8, TP10 (Muse standard)
Sampling rate: 256 Hz (Muse standard)
"""

import numpy as np
import time
from config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE


class FakeMuseGenerator:
    """
    Simulates a Muse headband EEG stream.
    Generates realistic EEG-like signals by summing band-specific sinusoids
    plus pink noise, with optional state-based modulation.
    """

    BAND_RANGES = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 45),
    }

    # State presets: (band, amplitude_multiplier)
    STATE_PROFILES = {
        "relaxed":   {"alpha": 3.0, "theta": 0.8, "beta": 0.6, "delta": 0.4, "gamma": 0.2},
        "focused":   {"beta": 2.5,  "alpha": 0.8, "theta": 0.5, "delta": 0.4, "gamma": 0.8},
        "anxious":   {"beta": 3.5,  "gamma": 2.0, "alpha": 0.3, "theta": 0.6, "delta": 0.2},
        "drowsy":    {"theta": 3.0, "delta": 2.0, "alpha": 1.2, "beta": 0.3, "gamma": 0.1},
        "meditative":{"alpha": 3.0, "theta": 2.5, "delta": 0.5, "beta": 0.3, "gamma": 0.2},
    }

    def __init__(self, state: str = "relaxed", noise_level: float = 0.3):
        """
        Args:
            state: One of the STATE_PROFILES keys.
            noise_level: Amplitude of pink noise added to the signal (µV).
        """
        if state not in self.STATE_PROFILES:
            raise ValueError(f"Unknown state '{state}'. Choose from: {list(self.STATE_PROFILES)}")
        self.state = state
        self.noise_level = noise_level
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self._t = 0  # global sample counter (keeps phase continuous across chunks)

    def _pink_noise(self, n_samples: int) -> np.ndarray:
        """Approximate pink (1/f) noise via spectral shaping."""
        white = np.random.randn(n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-6  # avoid div-by-zero at DC
        pink_filter = 1.0 / np.sqrt(freqs)
        fft *= pink_filter
        return np.fft.irfft(fft, n=n_samples)

    def _generate_channel(self, n_samples: int) -> np.ndarray:
        """Generate one channel of EEG-like signal."""
        t = (np.arange(n_samples) + self._t) / self.sample_rate
        profile = self.STATE_PROFILES[self.state]
        signal = np.zeros(n_samples)

        for band, (f_low, f_high) in self.BAND_RANGES.items():
            amp = profile.get(band, 1.0)
            # Pick a random centre frequency within the band
            centre_freq = np.random.uniform(f_low, f_high)
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * centre_freq * t + phase)

        # Add pink noise
        signal += self.noise_level * self._pink_noise(n_samples)
        # Scale to realistic µV range (~10–100 µV)
        signal *= 10
        return signal

    def get_chunk(self, n_samples: int = CHUNK_SIZE) -> dict:
        """
        Returns a chunk of multi-channel EEG data.

        Returns:
            dict with keys:
                'data'      : np.ndarray of shape (n_channels, n_samples)
                'timestamps': np.ndarray of shape (n_samples,) in seconds
                'channels'  : list of channel names
                'state'     : ground-truth state label
        """
        data = np.array([self._generate_channel(n_samples) for _ in self.channels])
        timestamps = (np.arange(n_samples) + self._t) / self.sample_rate
        self._t += n_samples
        return {
            "data": data,
            "timestamps": timestamps,
            "channels": self.channels,
            "state": self.state,
        }

    def stream(self, duration_sec: float = None, realtime: bool = True):
        """
        Generator that yields chunks indefinitely (or for `duration_sec`).

        Args:
            duration_sec: Stop after this many seconds. None = infinite.
            realtime: Sleep between chunks to simulate real-time streaming.
        """
        chunk_duration = CHUNK_SIZE / self.sample_rate
        elapsed = 0.0
        while True:
            if duration_sec is not None and elapsed >= duration_sec:
                break
            chunk = self.get_chunk()
            yield chunk
            elapsed += chunk_duration
            if realtime:
                time.sleep(chunk_duration)

    def set_state(self, state: str):
        """Dynamically change the simulated mental state."""
        if state not in self.STATE_PROFILES:
            raise ValueError(f"Unknown state '{state}'.")
        self.state = state
        print(f"[Generator] State changed to '{state}'")


if __name__ == "__main__":
    gen = FakeMuseGenerator(state="relaxed")
    print("Streaming fake EEG for 2 seconds...")
    for chunk in gen.stream(duration_sec=2, realtime=False):
        print(f"  Chunk shape: {chunk['data'].shape}, "
              f"timestamps: {chunk['timestamps'][0]:.3f}s – {chunk['timestamps'][-1]:.3f}s")