"""
TASKS 4–7: Feature Extraction, Derived States, Recovery, Training Load & Injury Risk
======================================================================================
Task 4: Extract signal features from daily data
Task 5: Derived State Model : Readiness Score
Task 6: Recovery State Classification
Task 7: Training Load Balance (Acute:Chronic ratio) + Injury Risk Model
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from synthetic_data import generate_daily_records

# ════════════════════════════════════════════════════════════════════
# TASK 4: FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════
def extract_features(df):
    """Compute intermediate features from raw daily signals."""
    df = df.copy()

    # 4.1 Resting HR deviation from personal baseline (52 bpm)
    df["resting_hr_delta"] = (df["resting_HR"] - 52).round(2)

    # 4.2 HRV ratio vs rolling 7-day mean (>1 = above baseline = good)
    df["hrv_7d_mean"] = df["HRV_ms"].rolling(7, min_periods=1).mean().round(2)
    df["hrv_ratio"]   = (df["HRV_ms"] / df["hrv_7d_mean"]).round(3)

    # 4.3 Activity intensity proxy : training_load_AU
    df["load_norm"] = ((df["training_load_AU"] - df["training_load_AU"].min()) /
                       (df["training_load_AU"].max() - df["training_load_AU"].min())).round(4)

    # 4.4 Sleep quality score (0–1): >8h=1.0, <5h=0.0, linear between
    df["sleep_quality"] = ((df["sleep_duration_h"] - 5) / 3).clip(0, 1).round(4)

    # 4.5 Temperature deviation absolute (proxy for inflammation)
    df["temp_dev_abs"] = df["temp_deviation_C"].abs().round(3)

    return df

# ════════════════════════════════════════════════════════════════════
# TASK 5: READINESS SCORE
# ════════════════════════════════════════════════════════════════════
def compute_readiness(df):
    """
    readiness_score = 0.4 * normalized_hrv
                    + 0.3 * sleep_quality
                    + 0.3 * (1 - fatigue_load)
    Scaled to 0–100.
    """
    df = df.copy()

    # Normalize HRV to 0–1 (15–100 ms range)
    df["hrv_norm"]     = ((df["HRV_ms"] - 15) / (100 - 15)).clip(0, 1).round(4)

    # Fatigue load: normalized training load from last 3 days rolling average
    df["fatigue_load"] = df["training_load_AU"].rolling(3, min_periods=1).mean()
    df["fatigue_load"] = ((df["fatigue_load"] - df["fatigue_load"].min()) /
                          (df["fatigue_load"].max() - df["fatigue_load"].min())).clip(0, 1).round(4)

    df["readiness_score"] = (
        0.4 * df["hrv_norm"] +
        0.3 * df["sleep_quality"] +
        0.3 * (1 - df["fatigue_load"])
    ).round(4) * 100

    df["readiness_score"] = df["readiness_score"].round(1)

    # Readiness label
    def label_readiness(s):
        if s >= 75: return "high"
        if s >= 50: return "moderate"
        return "low"

    df["readiness_label"] = df["readiness_score"].apply(label_readiness)
    return df

# ════════════════════════════════════════════════════════════════════
# TASK 6: RECOVERY STATE
# ════════════════════════════════════════════════════════════════════
def compute_recovery(df):
    """
    recovery_state based on:
      - HRV trend (3-day slope)
      - Sleep duration
      - Skin temperature deviation
    """
    df = df.copy()

    # HRV 3-day trend slope
    hrv_trends = []
    for i in range(len(df)):
        window = df["HRV_ms"].iloc[max(0, i-2):i+1].values
        if len(window) >= 2:
            slope = np.polyfit(range(len(window)), window, 1)[0]
        else:
            slope = 0
        hrv_trends.append(round(slope, 3))
    df["hrv_trend"] = hrv_trends

    def classify_recovery(row):
        score = 0
        # HRV trending up = good (+1), down = bad (-1)
        if row["hrv_trend"] > 0.5:    score += 2
        elif row["hrv_trend"] > 0:    score += 1
        elif row["hrv_trend"] < -1:   score -= 2
        else:                          score -= 1
        # Sleep
        if row["sleep_duration_h"] >= 7.5: score += 2
        elif row["sleep_duration_h"] >= 6: score += 1
        else:                              score -= 1
        # Temp deviation
        if row["temp_dev_abs"] < 0.15:  score += 1
        elif row["temp_dev_abs"] > 0.4: score -= 2

        if score >= 4:  return "optimal"
        if score >= 1:  return "partial"
        return "poor"

    df["recovery_state"] = df.apply(classify_recovery, axis=1)
    return df

# ════════════════════════════════════════════════════════════════════
# TASK 7: TRAINING LOAD BALANCE + INJURY RISK MODEL
# ════════════════════════════════════════════════════════════════════
def compute_load_balance(df):
    """Acute:Chronic Workload Ratio (ACWR)."""
    df = df.copy()
    df["acute_load"]   = df["training_load_AU"].rolling(7,  min_periods=1).mean().round(3)
    df["chronic_load"] = df["training_load_AU"].rolling(28, min_periods=1).mean().round(3)
    df["acwr"] = (df["acute_load"] / df["chronic_load"].replace(0, np.nan)).round(3)

    def load_label(r):
        if pd.isna(r): return "unknown"
        if r < 0.8:   return "undertraining"
        if r <= 1.3:  return "optimal"
        if r <= 1.5:  return "high"
        return "overreaching"

    df["load_balance_label"] = df["acwr"].apply(load_label)
    return df

def build_injury_model(df):
    """Train logistic regression to predict injury_event."""
    features = ["acwr", "readiness_score", "soreness_score",
                "hrv_ratio", "sleep_quality", "temp_dev_abs"]

    model_df = df[features + ["injury_event"]].dropna()
    X = model_df[features]
    y = model_df["injury_event"]

    if y.sum() < 3:
        print("⚠️  Too few injury events for model : using rule-based fallback")
        df["injury_risk_score"] = (
            (df["acwr"].fillna(1) > 1.4).astype(int) * 30 +
            (df["readiness_score"] < 45).astype(int) * 30 +
            (df["soreness_score"] > 7).astype(int) * 20 +
            (df["temp_dev_abs"] > 0.4).astype(int) * 20
        )
        df["injury_risk_label"] = df["injury_risk_score"].apply(
            lambda s: "high" if s >= 60 else ("moderate" if s >= 30 else "low"))
        return df, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    print("\nInjury Risk Model : Test Performance:")
    print(classification_report(y_te, model.predict(X_te), zero_division=0))

    full_X = scaler.transform(df[features].fillna(df[features].median()))
    df["injury_risk_score"] = (model.predict_proba(full_X)[:, 1] * 100).round(1)
    df["injury_risk_label"] = df["injury_risk_score"].apply(
        lambda s: "high" if s >= 60 else ("moderate" if s >= 30 else "low"))
    return df, model, scaler

# ════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ════════════════════════════════════════════════════════════════════
def run_full_pipeline():
    print("="*60)
    print("RUNNING FULL ANALYTICS PIPELINE (Tasks 3–7)")
    print("="*60)

    df = generate_daily_records()
    print(f"\n[Task 3] Generated {len(df)} synthetic daily records")

    df = extract_features(df)
    print("[Task 4] Features extracted")

    df = compute_readiness(df)
    print("[Task 5] Readiness scores computed")

    df = compute_recovery(df)
    print("[Task 6] Recovery states classified")

    df = compute_load_balance(df)
    print("[Task 7] Training load balance computed")

    df, model, scaler = build_injury_model(df)
    print("[Task 7] Injury risk model built")

    output_cols = [
        "date", "day_index", "day_category",
        "resting_HR", "HRV_ms", "sleep_duration_h",
        "training_load_AU", "soreness_score", "injury_event",
        "sleep_quality", "hrv_norm", "hrv_ratio",
        "readiness_score", "readiness_label",
        "hrv_trend", "recovery_state",
        "acute_load", "chronic_load", "acwr", "load_balance_label",
        "injury_risk_score", "injury_risk_label",
    ]

    out_df = df[[c for c in output_cols if c in df.columns]]
    out_df.to_csv("/mnt/user-data/outputs/analytics_pipeline_output.csv", index=False)
    print(f"\n Pipeline complete : {len(out_df)} rows saved")
    print("\nSample output:")
    print(out_df[["date","readiness_score","recovery_state","acwr","injury_risk_label"]].head(10).to_string())
    return out_df

if __name__ == "__main__":
    run_full_pipeline()
