# 🧠 EEG Analysis Pipeline

A brain-wave analysis pipeline that reads EEG signals, identifies frequency bands, and interprets your mental state in real time. Currently uses fake Muse-like data — real Muse headband integration is one step away.

---

## What it does

1. **Generates** fake EEG signal (4 channels, 256 Hz — same format as a Muse headband)
2. **Cleans** the signal — removes power-line noise and filters to brain-wave range
3. **Extracts** band powers — how much delta, theta, alpha, beta, and gamma is present
4. **Classifies** your mental state based on the band pattern
5. **Reports** a human-readable result with tips and a session summary

### Mental states detected

| State | Brain signature |
|---|---|
| 😌 Relaxed | High alpha, low beta |
| 🧠 Focused | High beta, moderate alpha, low gamma |
| 😰 Anxious | High beta + elevated gamma, suppressed alpha |
| 😴 Drowsy | High theta + delta |
| 🧘 Meditative | High alpha + theta together, very low beta |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run it

```bash
python main.py
```

Then pick a state from the menu and a duration. Or use command-line flags:

```bash
python main.py --state anxious --duration 10
python main.py --tour              # cycles through all 5 states
python main.py --state focused --quiet   # minimal output
```

---

## File structure

```
eeg_pipeline/
├── main.py            # Entry point — interactive menu + CLI
├── pipeline.py        # Orchestrates all steps end-to-end
├── config.py          # All settings (sample rate, thresholds, bands)
├── data_generator.py  # Fake Muse EEG signal generator
├── preprocessing.py   # Notch filter, bandpass, artifact rejection
├── features.py        # Band power extraction, ratios, spectral entropy
├── states.py          # Mental state classifier
├── feedback.py        # User-facing output + session summary
└── requirements.txt
```

---

## Connecting a real Muse headband

When you're ready to use real hardware, only `data_generator.py` needs to change. Replace `FakeMuseGenerator.get_chunk()` with a pylsl stream from muselsl — everything else stays the same.

Install the extra packages:
```bash
pip install muselsl pylsl bleak
```

Then stream from your Muse:
```bash
muselsl stream   # in one terminal
python main.py   # in another
```

---

## EEG frequency bands reference

| Band | Range | Associated with |
|---|---|---|
| Delta | 0.5–4 Hz | Deep sleep, unconscious processes |
| Theta | 4–8 Hz | Drowsiness, meditation, memory |
| Alpha | 8–13 Hz | Relaxed wakefulness, calm focus |
| Beta | 13–30 Hz | Active thinking, focus, anxiety |
| Gamma | 30–45 Hz | High-level cognition, stress |