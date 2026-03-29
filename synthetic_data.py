"""
TASK 3: Synthetic Patient / Clinical Data
==========================================
3.1 Synthea-inspired: daily clinical snapshot per patient
3.2 PhysioNet-inspired: realistic signal patterns by day category
     - normal | fatigue | poor_sleep | overload | injury_risk
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(7)

START_DATE = datetime(2024, 1, 1)
DAYS = 45

# ─── 3.1 SYNTHEA-INSPIRED PATIENT PROFILE ───────────────────────────────────
PATIENT = {
    "id":           "SYN-2024-001",
    "name":         "Alex Mora",
    "age":          28,
    "sex":          "M",
    "sport":        "triathlon",
    "resting_hr_baseline": 52,
    "hrv_baseline":        68,
    "sleep_baseline_h":    7.5,
}

# ─── 3.2 PHYSIONET-INSPIRED DAY CATEGORIES ───────────────────────────────────
# Ranges derived from PhysioNet MIMIC, WFDB, and sleep-edf reference distributions
PHYSIONET_RANGES = {
    "normal": {
        "resting_hr":    (50, 62),  "hrv":      (58, 78),
        "sleep_h":       (7.0, 8.5),"session_h":(1.0, 1.5),
        "intensity":     (5, 7),    "temp_dev": (-0.1, 0.1),
        "soreness":      (1, 3),    "injury_p": 0.02,
    },
    "fatigue": {
        "resting_hr":    (63, 72),  "hrv":      (38, 52),
        "sleep_h":       (6.0, 7.2),"session_h":(0.8, 1.2),
        "intensity":     (4, 6),    "temp_dev": (0.1, 0.4),
        "soreness":      (5, 8),    "injury_p": 0.10,
    },
    "poor_sleep": {
        "resting_hr":    (60, 70),  "hrv":      (35, 50),
        "sleep_h":       (4.0, 5.5),"session_h":(0.7, 1.0),
        "intensity":     (3, 5),    "temp_dev": (0.0, 0.3),
        "soreness":      (4, 7),    "injury_p": 0.08,
    },
    "overload": {
        "resting_hr":    (68, 80),  "hrv":      (25, 40),
        "sleep_h":       (5.5, 7.0),"session_h":(1.8, 2.5),
        "intensity":     (8, 10),   "temp_dev": (0.3, 0.7),
        "soreness":      (7, 10),   "injury_p": 0.20,
    },
    "injury_risk": {
        "resting_hr":    (70, 85),  "hrv":      (20, 35),
        "sleep_h":       (4.5, 6.0),"session_h":(1.5, 2.2),
        "intensity":     (7, 10),   "temp_dev": (0.4, 0.9),
        "soreness":      (8, 10),   "injury_p": 0.40,
    },
}

# Scripted day-category sequence (realistic periodization block)
DAY_CATEGORY_CYCLE = (
    ["normal"] * 3 + ["fatigue"] +
    ["normal"] * 2 + ["poor_sleep"] +
    ["overload"] * 3 + ["fatigue"] + ["normal"] +
    ["injury_risk"] * 2 + ["normal"] * 2
)

def sample(lo, hi):
    return round(np.random.uniform(lo, hi), 2)

def generate_daily_records():
    records = []
    category_cycle = [DAY_CATEGORY_CYCLE[i % len(DAY_CATEGORY_CYCLE)] for i in range(DAYS)]

    for day_idx in range(DAYS):
        date = (START_DATE + timedelta(days=day_idx)).strftime("%Y-%m-%d")
        cat  = category_cycle[day_idx]
        r    = PHYSIONET_RANGES[cat]

        resting_hr   = sample(*r["resting_hr"])
        hrv          = sample(*r["hrv"])
        sleep_h      = sample(*r["sleep_h"])
        session_h    = sample(*r["session_h"])
        intensity    = round(np.random.uniform(*r["intensity"]))
        temp_dev     = sample(*r["temp_dev"])
        soreness     = round(np.random.uniform(*r["soreness"]))
        injury_event = 1 if np.random.random() < r["injury_p"] else 0

        # Training load = session_h * intensity (AU - arbitrary units)
        training_load = round(session_h * intensity, 2)

        records.append({
            "date":              date,
            "day_index":         day_idx + 1,
            "patient_id":        PATIENT["id"],
            "day_category":      cat,
            "resting_HR":        resting_hr,
            "HRV_ms":            hrv,
            "sleep_duration_h":  sleep_h,
            "session_duration_h":session_h,
            "training_intensity":intensity,
            "training_load_AU":  training_load,
            "temp_deviation_C":  temp_dev,
            "soreness_score":    soreness,
            "injury_event":      injury_event,
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    import os
    df = generate_daily_records()
    output_dir = os.path.dirname(__file__)
    df.to_csv(os.path.join(output_dir, "synthetic_daily_data.csv"), index=False)
    print(f"Generated {len(df)} daily records")
    print(df[["date","day_category","resting_HR","HRV_ms","sleep_duration_h","injury_event"]].head(10).to_string())
    print(f"\nInjury events: {df['injury_event'].sum()} / {len(df)} days")
    print(f"Day category distribution:\n{df['day_category'].value_counts()}")
