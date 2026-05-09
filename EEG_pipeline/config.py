"""
Global configuration for EEG pipeline
"""

SFREQ = 160

CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2"]

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

WINDOW_SIZE = 2.0   # seconds
STEP_SIZE = 1.0     # seconds

EPSILON = 1e-10