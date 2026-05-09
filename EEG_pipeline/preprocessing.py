import numpy as np
from scipy.signal import butter, filtfilt
from config import SFREQ


def bandpass_filter(data, low=0.5, high=40.0, order=4):
    nyq = 0.5 * SFREQ
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def window_data(data, window_size, step_size):
    """
    data: (channels, samples)
    returns: list of windows (channels, window_samples)
    """
    win_samples = int(window_size * SFREQ)
    step_samples = int(step_size * SFREQ)

    windows = []

    for start in range(0, data.shape[1] - win_samples, step_samples):
        window = data[:, start:start + win_samples]
        windows.append(window)

    return windows