import pandas as pd

from preprocessing import bandpass_filter, window_data
from features import compute_band_powers, compute_focus_score
from states import classify_state
from config import WINDOW_SIZE, STEP_SIZE


def run_pipeline(data, channels):
    """
    data: (channels, samples)
    """

    # 1. Filter
    data = bandpass_filter(data)

    # 2. Windowing
    windows = window_data(data, WINDOW_SIZE, STEP_SIZE)

    results = []

    for window in windows:
        band_features = compute_band_powers(window)

        for ch_name, feats in zip(channels, band_features):
            focus = compute_focus_score(feats)
            state = classify_state(feats)

            row = {
                "channel": ch_name,
                **feats,
                "focus_score": focus,
                "state": state
            }

            results.append(row)

    return pd.DataFrame(results)