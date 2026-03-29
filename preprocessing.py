"""
TASK 2: Data Pre-Processing Pipeline
=====================================
- Remove missing / duplicate timestamps
- Interpolate HR gaps
- Normalize acceleration signals
- Flag and handle outliers
"""

import pandas as pd
import numpy as np
from data_architecture import simulate_wearable_data

def load_and_validate(df):
    """Step 1: Load, sort, remove duplicates."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    # Inject synthetic gaps for demo (5% of rows)
    drop_idx = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[drop_idx, "HR_bpm"] = np.nan
    df.loc[drop_idx, "HRV_ms"] = np.nan
    return df

def remove_missing_timestamps(df):
    """Step 2: Ensure continuous 10-min timeline; fill gaps."""
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="10min")
    df = df.set_index("timestamp").reindex(full_range).rename_axis("timestamp").reset_index()
    return df

def interpolate_signals(df):
    """Step 3: Interpolate HR & HRV gaps (time-linear)."""
    df["HR_bpm"]  = df["HR_bpm"].interpolate(method="linear", limit_direction="both")
    df["HRV_ms"]  = df["HRV_ms"].interpolate(method="linear", limit_direction="both")
    df["skin_temp_C"] = df["skin_temp_C"].interpolate(method="linear", limit_direction="both")
    df["SpO2_pct"]    = df["SpO2_pct"].interpolate(method="linear", limit_direction="both")
    # Forward-fill categorical columns
    for col in ["date", "day_type", "phase", "accel_label"]:
        df[col] = df[col].ffill().bfill()
    return df

def remove_outliers(df):
    """Step 4: Clip physiological impossibilities."""
    df["HR_bpm"]      = df["HR_bpm"].clip(30, 210)
    df["HRV_ms"]      = df["HRV_ms"].clip(5, 120)
    df["skin_temp_C"] = df["skin_temp_C"].clip(32, 38)
    df["SpO2_pct"]    = df["SpO2_pct"].clip(85, 100)
    df["accel_g"]     = df["accel_g"].clip(0, 3)
    return df

def normalize_signals(df):
    """Step 5: Min-max normalize continuous sensor columns."""
    cols_to_norm = ["HR_bpm", "HRV_ms", "accel_g", "skin_temp_C", "SpO2_pct"]
    for col in cols_to_norm:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_norm"] = ((df[col] - col_min) / (col_max - col_min)).round(4)
    return df

def add_rolling_features(df):
    """Step 6: Add 1-hour rolling mean/std for HR and HRV."""
    df = df.set_index("timestamp")
    df["HR_roll1h_mean"] = df["HR_bpm"].rolling("60min").mean().round(2)
    df["HRV_roll1h_mean"] = df["HRV_ms"].rolling("60min").mean().round(2)
    df = df.reset_index()
    return df

def preprocess_pipeline(df=None):
    if df is None:
        df = simulate_wearable_data()
    print(f"Raw records: {len(df):,}")
    df = load_and_validate(df)
    df = remove_missing_timestamps(df)
    df = interpolate_signals(df)
    df = remove_outliers(df)
    df = normalize_signals(df)
    df = add_rolling_features(df)
    df = df.dropna(subset=["HR_bpm"]).reset_index(drop=True)
    print(f"Clean records: {len(df):,} | Missing HR: {df['HR_bpm'].isna().sum()}")
    return df

if __name__ == "__main__":
    import os
    clean_df = preprocess_pipeline()
    output_dir = os.path.dirname(__file__)
    clean_df.to_csv(os.path.join(output_dir, "clean_wearable_data.csv"), index=False)
    print("Saved clean_wearable_data.csv")
