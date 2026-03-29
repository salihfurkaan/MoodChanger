"""
Patient Profiles Configuration
================================
Manages multiple patient profiles for the wellness analytics dashboard.
Users can select from predefined profiles or add custom ones.
"""

PATIENTS = {
    "demo": {
        "id":           "SYN-2024-001",
        "name":         "Alex Mora",
        "age":          28,
        "sex":          "M",
        "sport":        "Triathlon",
        "resting_hr_baseline": 52,
        "hrv_baseline":        68,
        "sleep_baseline_h":    7.5,
        "description":  "Demo athlete (triathlon)"
    },
    "SYN-2024-002": {
        "id":           "SYN-2024-002",
        "name":         "Jordan Chen",
        "age":          26,
        "sex":          "M",
        "sport":        "Marathon Running",
        "resting_hr_baseline": 48,
        "hrv_baseline":        72,
        "sleep_baseline_h":    8.0,
        "description":  "Distance runner"
    },
    "SYN-2024-003": {
        "id":           "SYN-2024-003",
        "name":         "Emma Wilson",
        "age":          30,
        "sex":          "F",
        "sport":        "Swimming",
        "resting_hr_baseline": 50,
        "hrv_baseline":        65,
        "sleep_baseline_h":    7.5,
        "description":  "Competitive swimmer"
    },
    "SYN-2024-004": {
        "id":           "SYN-2024-004",
        "name":         "Marcus Johnson",
        "age":          24,
        "sex":          "M",
        "sport":        "CrossFit",
        "resting_hr_baseline": 55,
        "hrv_baseline":        60,
        "sleep_baseline_h":    7.0,
        "description":  "Functional fitness athlete"
    },
}

def get_patient(patient_id):
    """Retrieve patient profile by ID."""
    return PATIENTS.get(patient_id, PATIENTS.get("demo"))

def get_patient_list():
    """Return list of available patient IDs."""
    return list(PATIENTS.keys())

def get_patient_labels():
    """Return mapping of patient IDs to display labels."""
    return {pid: f"{p['name']} ({p['sport']}) - {p['description']}" 
            for pid, p in PATIENTS.items()}

def get_default_patient():
    """Return default patient ID."""
    return "demo"

def create_patient(patient_id, name, age, sex, sport, resting_hr_baseline, hrv_baseline, sleep_baseline_h):
    """Create a new patient profile with validation."""
    # Validate inputs
    if not patient_id or not name or not sport:
        raise ValueError("Patient ID, name, and sport are required")
    
    if not (20 <= age <= 80):
        raise ValueError("Age must be between 20 and 80")
    
    if sex not in ["M", "F"]:
        raise ValueError("Sex must be 'M' or 'F'")
    
    if not (30 <= resting_hr_baseline <= 120):
        raise ValueError("Resting HR must be between 30 and 120 bpm")
    
    if not (20 <= hrv_baseline <= 200):
        raise ValueError("HRV baseline must be between 20 and 200 ms")
    
    if not (4 <= sleep_baseline_h <= 12):
        raise ValueError("Sleep baseline must be between 4 and 12 hours")
    
    return {
        "id": patient_id,
        "name": name,
        "age": age,
        "sex": sex,
        "sport": sport,
        "resting_hr_baseline": resting_hr_baseline,
        "hrv_baseline": hrv_baseline,
        "sleep_baseline_h": sleep_baseline_h,
        "description": f"{name} - {sport}"
    }

def get_demo_patient():
    """Return the hardcoded demo patient for testing."""
    return PATIENTS.get("demo")
