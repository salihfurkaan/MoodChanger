"""
Dynamic Data Generation for Patients
======================================
Generates synthetic wearable and clinical data tailored to each patient's baselines.
Integrates with the existing pipeline to create complete analytics datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

def generate_patient_data(patient_profile, days=45, output_dir=None, seed=7):
    """
    Generate complete synthetic dataset for a patient.
    
    Parameters:
    -----------
    patient_profile : dict
        Patient data with keys: id, name, age, sex, sport, 
        resting_hr_baseline, hrv_baseline, sleep_baseline_h
    days : int
        Number of days to simulate (default 45)
    output_dir : str
        Directory to save CSV files (default current directory)
    seed : int
        Random seed for reproducibility (default 7)
    
    Returns:
    --------
    tuple : (daily_df, wearable_df)
        DataFrames with daily records and wearable sensor data
    """
    
    np.random.seed(seed)
    output_dir = output_dir or os.path.dirname(__file__)
    
    START_DATE = datetime(2024, 1, 1)
    
    # Extract patient baselines
    resting_hr_base = patient_profile.get("resting_hr_baseline", 52)
    hrv_base = patient_profile.get("hrv_baseline", 68)
    sleep_base = patient_profile.get("sleep_baseline_h", 7.5)
    patient_id = patient_profile.get("id", "SYN-2024-001")
    
    # ───────────────────────────────────────────────────────────────────
    # DAILY SYNTHETIC DATA
    # ───────────────────────────────────────────────────────────────────
    
    DAY_CATEGORY_CYCLE = (
        ["normal"] * 3 + ["fatigue"] +
        ["normal"] * 2 + ["poor_sleep"] +
        ["overload"] * 3 + ["fatigue"] + ["normal"] +
        ["injury_risk"] * 2 + ["normal"] * 2
    )
    
    # Scale ranges based on patient baselines
    PHYSIONET_RANGES = {
        "normal": {
            "resting_hr":    (resting_hr_base - 2, resting_hr_base + 10),
            "hrv":           (hrv_base - 10, hrv_base + 10),
            "sleep_h":       (sleep_base - 0.5, sleep_base + 1.0),
            "session_h":     (1.0, 1.5),
            "intensity":     (5, 7),
            "temp_dev":      (-0.1, 0.1),
            "soreness":      (1, 3),
            "injury_p":      0.02,
        },
        "fatigue": {
            "resting_hr":    (resting_hr_base + 11, resting_hr_base + 20),
            "hrv":           (max(20, hrv_base - 30), max(20, hrv_base - 16)),
            "sleep_h":       (max(4, sleep_base - 1.5), max(4, sleep_base - 0.3)),
            "session_h":     (0.8, 1.2),
            "intensity":     (4, 6),
            "temp_dev":      (0.1, 0.4),
            "soreness":      (5, 8),
            "injury_p":      0.10,
        },
        "poor_sleep": {
            "resting_hr":    (resting_hr_base + 8, resting_hr_base + 18),
            "hrv":           (max(20, hrv_base - 33), max(20, hrv_base - 18)),
            "sleep_h":       (max(4, sleep_base - 3.5), max(4, sleep_base - 2)),
            "session_h":     (0.7, 1.0),
            "intensity":     (3, 5),
            "temp_dev":      (0.0, 0.3),
            "soreness":      (4, 7),
            "injury_p":      0.08,
        },
        "overload": {
            "resting_hr":    (resting_hr_base + 16, resting_hr_base + 28),
            "hrv":           (max(20, hrv_base - 43), max(20, hrv_base - 28)),
            "sleep_h":       (max(4, sleep_base - 2), max(4, sleep_base - 0.5)),
            "session_h":     (1.8, 2.5),
            "intensity":     (8, 10),
            "temp_dev":      (0.3, 0.7),
            "soreness":      (7, 10),
            "injury_p":      0.20,
        },
        "injury_risk": {
            "resting_hr":    (resting_hr_base + 18, resting_hr_base + 33),
            "hrv":           (max(20, hrv_base - 48), max(20, hrv_base - 33)),
            "sleep_h":       (max(4, sleep_base - 3), max(4, sleep_base - 1.5)),
            "session_h":     (1.5, 2.2),
            "intensity":     (7, 10),
            "temp_dev":      (0.4, 0.9),
            "soreness":      (8, 10),
            "injury_p":      0.40,
        },
    }
    
    def sample(lo, hi):
        return round(np.random.uniform(lo, hi), 2)
    
    records = []
    category_cycle = [DAY_CATEGORY_CYCLE[i % len(DAY_CATEGORY_CYCLE)] for i in range(days)]
    
    for day_idx in range(days):
        date = (START_DATE + timedelta(days=day_idx)).strftime("%Y-%m-%d")
        cat = category_cycle[day_idx]
        r = PHYSIONET_RANGES[cat]
        
        resting_hr = sample(*r["resting_hr"])
        hrv = sample(*r["hrv"])
        sleep_h = sample(*r["sleep_h"])
        session_h = sample(*r["session_h"])
        intensity = round(np.random.uniform(*r["intensity"]))
        temp_dev = sample(*r["temp_dev"])
        soreness = round(np.random.uniform(*r["soreness"]))
        injury_event = 1 if np.random.random() < r["injury_p"] else 0
        
        training_load = round(session_h * intensity, 2)
        
        records.append({
            "date": date,
            "day_index": day_idx + 1,
            "patient_id": patient_id,
            "day_category": cat,
            "resting_HR": resting_hr,
            "HRV_ms": hrv,
            "sleep_duration_h": sleep_h,
            "session_duration_h": session_h,
            "training_intensity": intensity,
            "training_load_AU": training_load,
            "temp_deviation_C": temp_dev,
            "soreness_score": soreness,
            "injury_event": injury_event,
        })
    
    daily_df = pd.DataFrame(records)
    
    # Add patient baseline metadata for feature extraction in pipeline
    daily_df["patient_resting_hr_baseline"] = resting_hr_base
    daily_df["patient_hrv_baseline"] = hrv_base
    daily_df["patient_sleep_baseline_h"] = sleep_base
    
    # ───────────────────────────────────────────────────────────────────
    # WEARABLE SENSOR DATA (10-min intervals)
    # ───────────────────────────────────────────────────────────────────
    
    PHASES = [
        (0, 6, "sleep"),
        (6, 8, "wake_rest"),
        (8, 10, "morning_training"),
        (10, 12, "active"),
        (12, 13, "lunch_rest"),
        (13, 17, "evening_training"),
        (17, 22, "rest"),
        (22, 24, "sleep"),
    ]
    
    DAY_TYPES = ["training", "recovery", "rest"]
    
    def get_phase(hour, day_type):
        for start, end, label in PHASES:
            if start <= hour < end:
                if day_type in ("recovery", "rest") and "training" in label:
                    return "rest"
                return label
        return "rest"
    
    def generate_hr(phase, fatigue_factor=1.0):
        base = {
            "sleep": (resting_hr_base - 4, resting_hr_base + 6),
            "wake_rest": (resting_hr_base + 6, resting_hr_base + 16),
            "morning_training": (resting_hr_base + 78, resting_hr_base + 113),
            "active": (resting_hr_base + 33, resting_hr_base + 58),
            "lunch_rest": (resting_hr_base + 13, resting_hr_base + 23),
            "evening_training": (resting_hr_base + 73, resting_hr_base + 108),
            "rest": (resting_hr_base + 10, resting_hr_base + 20),
        }
        lo, hi = base.get(phase, (resting_hr_base + 13, resting_hr_base + 23))
        return round(np.clip(
            np.random.uniform(lo, hi) * fatigue_factor + np.random.normal(0, 2),
            40, 200
        ), 1)
    
    def generate_hrv(phase, fatigue_factor=1.0):
        base = {
            "sleep": (hrv_base - 13, hrv_base + 7),
            "wake_rest": (hrv_base - 23, hrv_base - 8),
            "morning_training": (hrv_base - 38, hrv_base - 23),
            "active": (hrv_base - 33, hrv_base - 13),
            "lunch_rest": (hrv_base - 23, hrv_base - 8),
            "evening_training": (hrv_base - 40, hrv_base - 26),
            "rest": (hrv_base - 20, hrv_base - 3),
        }
        lo, hi = base.get(phase, (hrv_base - 28, hrv_base))
        return round(np.clip(
            np.random.uniform(lo, hi) / fatigue_factor + np.random.normal(0, 1.5),
            15, 100
        ), 1)
    
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
        return round(np.clip(
            np.random.uniform(lo, hi) + (fatigue_factor - 1) * 0.3 + np.random.normal(0, 0.05),
            32.0, 37.5
        ), 2)
    
    def generate_spo2(phase):
        if phase == "sleep":
            return round(np.random.uniform(95.0, 98.5), 1)
        return round(np.random.uniform(97.0, 99.5), 1)
    
    def build_fatigue_profile(num_days):
        fatigue, f = [], 1.0
        day_cycle = [DAY_TYPES[i % len(DAY_TYPES)] for i in range(num_days)]
        for dt in day_cycle:
            if dt == "training":   f = min(f + np.random.uniform(0.02, 0.08), 1.35)
            elif dt == "recovery": f = max(f - np.random.uniform(0.03, 0.07), 1.0)
            else:                  f = max(f - np.random.uniform(0.05, 0.10), 1.0)
            fatigue.append(f)
        return day_cycle, fatigue
    
    # Generate wearable data
    FREQ_MINUTES = 10
    wearable_records = []
    day_cycle, fatigue_profile = build_fatigue_profile(days)
    
    for day_idx in range(days):
        day_type = day_cycle[day_idx]
        fatigue_factor = fatigue_profile[day_idx]
        current_day = START_DATE + timedelta(days=day_idx)
        
        for minute in range(0, 1440, FREQ_MINUTES):
            ts = current_day + timedelta(minutes=minute)
            hour = minute / 60.0
            phase = get_phase(hour, day_type)
            accel = generate_accel(phase)
            
            wearable_records.append({
                "timestamp": ts,
                "date": current_day.strftime("%Y-%m-%d"),
                "day_index": day_idx + 1,
                "day_type": day_type,
                "phase": phase,
                "HR_bpm": generate_hr(phase, fatigue_factor),
                "HRV_ms": generate_hrv(phase, fatigue_factor),
                "accel_g": accel,
                "accel_label": accel_label(accel),
                "skin_temp_C": generate_temp(phase, fatigue_factor),
                "SpO2_pct": generate_spo2(phase),
                "fatigue_factor": round(fatigue_factor, 3),
            })
    
    wearable_df = pd.DataFrame(wearable_records)
    
    # Save to CSV
    daily_path = os.path.join(output_dir, "synthetic_daily_data.csv")
    wearable_path = os.path.join(output_dir, "raw_wearable_data.csv")
    
    daily_df.to_csv(daily_path, index=False)
    wearable_df.to_csv(wearable_path, index=False)
    
    return daily_df, wearable_df


if __name__ == "__main__":
    # Test with demo patient
    from patients_config import get_demo_patient
    
    demo = get_demo_patient()
    print(f"Generating data for {demo['name']} ({demo['id']})...")
    
    daily_df, wearable_df = generate_patient_data(demo)
    
    print(f"✓ Generated {len(daily_df)} daily records")
    print(f"✓ Generated {len(wearable_df)} wearable readings")
    print(f"\nDaily summary:")
    print(daily_df[["date", "day_category", "resting_HR", "HRV_ms", "sleep_duration_h"]].head(10))
