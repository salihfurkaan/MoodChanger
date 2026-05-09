"""
Main entry point for EEG pipeline
"""

import argparse

from pipeline import run_pipeline


def run_synthetic():
    from data_generator import generate_state_sequence

    eeg = generate_state_sequence(
        sequence=["neutral", "focused", "drowsy"],
        segment_duration=10
    )

    df = run_pipeline(eeg["data"], eeg["channels"])
    print("\nSynthetic Data Results:")
    print(df.head())


def run_edf(filepath):
    import mne

    raw = mne.io.read_raw_edf(filepath, preload=True)

    data = raw.get_data()
    channels = raw.ch_names

    df = run_pipeline(data, channels)

    print("\nEDF Data Results:")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "edf"],
        default="synthetic"
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path to EDF file (required if mode=edf)"
    )

    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic()

    elif args.mode == "edf":
        if not args.file:
            raise ValueError("Provide --file for EDF mode")
        run_edf(args.file)