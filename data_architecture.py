"""
TASK 1: Data Architecture & Simulated Wearable Data Generator
=============================================================
Generates 45 days of realistic wearable sensor data covering:
- Resting periods  |  - Sleep periods
- Training sessions  |  - Recovery days

Schema: timestamp | HR | HRV | accel | temp | phase | day_type
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

START_DATE = datetime(2024, 1, 1)
DAYS = 45
FREQ_MINUTES = 10

DAY_TYPES = ["training", "training", "training", "recovery", "rest"]

PHASES = [
    (0,  6,  "sleep"),
    (6,  7,  "wake_rest"),
    (7,  9,  "morning_training"),
    (9,  12, "active"),
    (12, 13, "lunch_rest"),
    (13, 17, "active"),
    (17, 19, "evening_training"),
    (19, 22, "rest"),
    (22, 24, "sleep"),
]

def get_phase(hour, day_type):
    for start, end, label in PHASES:
        if start <= hour < end:
            if day_type in ("recovery", "rest") and "training" in label:
                return "rest"
            return label
    return "rest"

def generate_hr(phase, fatigue_factor=1.0):
    base = {
        "sleep": (48, 58), "wake_rest": (58, 68),
        "morning_training": (130, 165), "active": (85, 110),
        "lunch_rest": (65, 75), "evening_training": (125, 160), "rest": (62, 72),
    }
    lo, hi = base.get(phase, (65, 75))
    return round(np.clip(np.random.uniform(lo, hi) * fatigue_factor + np.random.normal(0, 2), 40, 200), 1)

def generate_hrv(phase, fatigue_factor=1.0):
    base = {
        "sleep": (55, 75), "wake_rest": (45, 60),
        "morning_training": (30, 45), "active": (35, 55),
        "lunch_rest": (45, 60), "evening_training": (28, 42), "rest": (48, 65),
    }
    lo, hi = base.get(phase, (40, 60))
    return round(np.clip(np.random.uniform(lo, hi) / fatigue_factor + np.random.normal(0, 1.5), 15, 100), 1)

def generate_accel(phase):
    accel_map = {
        "sleep": (0.0, 0.05), "wake_rest": (0.05, 0.2),
        "morning_training": (0.6, 1.5), "active": (0.2, 0.6),
        "lunch_rest": (0.05, 0.15), "evening_training": (0.6, 1.4), "rest": (0.05, 0.2),
    }
    lo, hi = accel_map.get(phase, (0.05, 0.2))
    return round(np.random.uniform(lo, hi), 3)

def accel_label(val):
    if val < 0.1: return "very_low"
    if val < 0.3: return "low"
    if val < 0.7: return "moderate"
    return "high"

def generate_temp(phase, fatigue_factor=1.0):
    base = {
        "sleep": (33.5, 34.2), "wake_rest": (33.8, 34.5),
        "morning_training": (34.5, 35.8), "active": (34.2, 35.2),
        "lunch_rest": (33.9, 34.6), "evening_training": (34.4, 35.7), "rest": (33.8, 34.5),
    }
    lo, hi = base.get(phase, (34.0, 34.5))
    return round(np.clip(np.random.uniform(lo, hi) + (fatigue_factor - 1) * 0.3 + np.random.normal(0, 0.05), 32.0, 37.5), 2)

def generate_spo2(phase):
    if phase == "sleep":
        return round(np.random.uniform(95.0, 98.5), 1)
    return round(np.random.uniform(97.0, 99.5), 1)

def build_fatigue_profile(days):
    fatigue, f = [], 1.0
    day_cycle = [DAY_TYPES[i % len(DAY_TYPES)] for i in range(days)]
    for dt in day_cycle:
        if dt == "training":   f = min(f + np.random.uniform(0.02, 0.08), 1.35)
        elif dt == "recovery": f = max(f - np.random.uniform(0.03, 0.07), 1.0)
        else:                  f = max(f - np.random.uniform(0.05, 0.10), 1.0)
        fatigue.append(f)
    return day_cycle, fatigue

def simulate_wearable_data():
    records = []
    day_cycle, fatigue_profile = build_fatigue_profile(DAYS)
    for day_idx in range(DAYS):
        day_type = day_cycle[day_idx]
        fatigue_factor = fatigue_profile[day_idx]
        current_day = START_DATE + timedelta(days=day_idx)
        for minute in range(0, 1440, FREQ_MINUTES):
            ts = current_day + timedelta(minutes=minute)
            hour = minute / 60.0
            phase = get_phase(hour, day_type)
            accel = generate_accel(phase)
            records.append({
                "timestamp":      ts,
                "date":           current_day.strftime("%Y-%m-%d"),
                "day_index":      day_idx + 1,
                "day_type":       day_type,
                "phase":          phase,
                "HR_bpm":         generate_hr(phase, fatigue_factor),
                "HRV_ms":         generate_hrv(phase, fatigue_factor),
                "accel_g":        accel,
                "accel_label":    accel_label(accel),
                "skin_temp_C":    generate_temp(phase, fatigue_factor),
                "SpO2_pct":       generate_spo2(phase),
                "fatigue_factor": round(fatigue_factor, 3),
            })
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = simulate_wearable_data()
    df.to_csv("/mnt/user-data/outputs/raw_wearable_data.csv", index=False)
    print(f"Generated {len(df):,} records | {df['date'].nunique()} days")
