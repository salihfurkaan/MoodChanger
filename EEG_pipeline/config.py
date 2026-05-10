"""
config.py
Central configuration for the EEG pipeline.
All magic numbers live here — import from this module everywhere else.
"""

# ---------------------------------------------------------------------------
# Hardware / stream settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 256          # Hz — standard Muse sample rate
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]  # Muse electrode names
CHUNK_SIZE = 64            # samples per processing chunk (~250 ms)

# ---------------------------------------------------------------------------
# EEG frequency bands  (Hz)
# ---------------------------------------------------------------------------
BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
NOTCH_FREQ      = 50.0     # Hz — power-line noise (60 for US, 50 for EU/ET)
NOTCH_Q         = 30.0     # Quality factor for notch filter
BANDPASS_LOW    = 0.5      # Hz — high-pass cut-off
BANDPASS_HIGH   = 45.0     # Hz — low-pass cut-off
BANDPASS_ORDER  = 4        # Butterworth filter order
ARTIFACT_THRESH = 150.0    # µV — amplitude threshold for artifact rejection

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
WELCH_NPERSEG   = 128      # samples per Welch segment
SMOOTHING_ALPHA = 0.3      # EMA smoothing factor for band powers (0 = no update)

# ---------------------------------------------------------------------------
# State classification
# ---------------------------------------------------------------------------
# Thresholds used by the rule-based classifier (relative band powers, 0-1)
STATE_THRESHOLDS = {
    "alpha_high":  0.35,   # alpha dominance → relaxed/meditative
    "beta_high":   0.35,   # beta dominance  → focused/anxious
    "theta_high":  0.35,   # theta dominance → drowsy/meditative
    "gamma_high":  0.20,   # gamma spike     → anxious/alert
    "delta_high":  0.40,   # delta dominance → deep sleep / drowsy
}

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
VIZ_WINDOW_SEC  = 5.0      # seconds of raw EEG shown in time-series plot
VIZ_UPDATE_MS   = 200      # dashboard refresh interval in milliseconds
PLOT_COLORS = {
    "delta": "#6366f1",
    "theta": "#22d3ee",
    "alpha": "#4ade80",
    "beta":  "#facc15",
    "gamma": "#f87171",
}

# ---------------------------------------------------------------------------
# Feedback / output
# ---------------------------------------------------------------------------
FEEDBACK_HISTORY = 10      # number of recent state predictions to keep