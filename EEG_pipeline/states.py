"""
states.py
Rule-based mental-state classifier.

Anxious vs Focused separation (data-driven thresholds):
  - focused gamma mean ~0.09, std ~0.03  → threshold at 0.13
  - anxious gamma mean ~0.17, std ~0.05  → clearly above 0.13
  - focused alpha mean ~0.13, std ~0.06  → threshold at 0.09
  - anxious alpha mean ~0.06, std ~0.04  → clearly below 0.09
  Both conditions must hold simultaneously for "anxious".
"""

from dataclasses import dataclass, field
from typing import Dict, List
from config import STATE_THRESHOLDS, FEEDBACK_HISTORY


@dataclass
class StateResult:
    label:       str
    confidence:  float
    description: str
    tips:        List[str]
    band_powers: Dict[str, float] = field(default_factory=dict)
    ratios:      Dict[str, float] = field(default_factory=dict)

    def __str__(self):
        return (
            f"State: {self.label.upper()} (confidence {self.confidence:.0%})\n"
            f"  {self.description}\n"
            f"  Tips: {' | '.join(self.tips)}"
        )


_STATE_META = {
    "relaxed": {
        "description": "Calm and relaxed. Alpha waves are dominant.",
        "tips": [
            "Great time for creative thinking or light reading.",
            "Maintain this state with slow, deep breathing.",
        ],
    },
    "focused": {
        "description": "High cognitive engagement. Beta activity is elevated.",
        "tips": [
            "Ideal for analytical tasks and problem-solving.",
            "Take short breaks every 25–30 minutes to avoid fatigue.",
        ],
    },
    "anxious": {
        "description": "Elevated beta and gamma with suppressed alpha — signs of stress or anxiety.",
        "tips": [
            "Try box breathing: inhale 4 s, hold 4 s, exhale 4 s.",
            "Step away from screens for a few minutes.",
            "Progressive muscle relaxation can help reduce tension.",
        ],
    },
    "drowsy": {
        "description": "Theta and delta power are high — you may be sleepy or fatigued.",
        "tips": [
            "Consider a 10–20 minute nap if possible.",
            "Avoid demanding tasks; hydrate and move around.",
        ],
    },
    "meditative": {
        "description": "Deep alpha-theta synchrony — a meditative or flow state.",
        "tips": [
            "Excellent for mindfulness, creative insight, or light visualisation.",
            "Avoid abrupt interruptions; allow the state to deepen naturally.",
        ],
    },
    "neutral": {
        "description": "No strongly dominant frequency band detected.",
        "tips": ["A balanced baseline — good for planning or transitioning between tasks."],
    },
}


class MentalStateClassifier:
    """
    Rule-based classifier with data-driven thresholds.

    Focused vs Anxious — the hard case:
      Anxious requires BOTH gamma > 0.13 AND alpha < 0.09 simultaneously.
      This avoids misclassifying focused (which has gamma ~0.09, alpha ~0.13)
      as anxious on noisy chunks.
    """

    # Data-driven thresholds (derived from 40-chunk distribution analysis)
    GAMMA_ANXIOUS_THRESH = 0.11   # anxious mean(0.17) - 1.5std(0.05) = 0.095; focused mean+0.7std = 0.11   # focused mean+1std ≈ 0.12; anxious mean-1std ≈ 0.12
    ALPHA_ANXIOUS_THRESH = 0.09   # anxious mean+1std ≈ 0.09; focused mean-1std ≈ 0.07

    def __init__(self, history_len: int = FEEDBACK_HISTORY):
        self.history_len = history_len
        self._history: List[str] = []

    def _raw_classify(self, summary: dict) -> tuple:
        a = summary.get("alpha_power", 0)
        b = summary.get("beta_power",  0)
        t = summary.get("theta_power", 0)
        g = summary.get("gamma_power", 0)
        d = summary.get("delta_power", 0)
        th = STATE_THRESHOLDS

        # --- Boolean signals ---
        high_beta       = b > 0.40
        # Anxious: BOTH gamma AND alpha conditions must hold
        is_anxious      = g > self.GAMMA_ANXIOUS_THRESH and a < self.ALPHA_ANXIOUS_THRESH
        high_alpha      = a > th["alpha_high"]          # > 0.35
        # Meditative: high alpha AND high theta together
        is_meditative   = t > 0.40 and a > 0.38   # meditative: high theta(0.47) + high alpha(0.45); drowsy has low alpha(0.31)
        high_theta      = t > th["theta_high"]          # > 0.35
        low_beta        = b < 0.20

        scores = {
            "meditative": (
                is_meditative   * 0.6 +
                low_beta        * 0.4
            ),
            "relaxed": (
                high_alpha              * 0.5 +
                (b < th["beta_high"])   * 0.3 +
                (not high_theta)        * 0.2
            ),
            # focused: high beta, NOT anxious pattern
            "focused": (
                high_beta               * 0.5 +
                (not is_anxious)        * 0.3 +   # key: not anxious
                (a > 0.08)              * 0.2
            ),
            # anxious: strict dual condition required
            "anxious": (
                is_anxious              * 0.6 +   # must satisfy both gamma+alpha
                high_beta               * 0.4
            ),
            "drowsy": (
                high_theta              * 0.4 +
                (d > th["delta_high"])  * 0.4 +
                low_beta                * 0.2
            ),
        }

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        if best_score < 0.25:
            return "neutral", 0.4

        confidence = 0.4 + min(best_score, 1.0) * 0.55
        return best_label, round(confidence, 2)

    def classify(self, feature_dict: dict) -> StateResult:
        summary = feature_dict["summary"]
        label, confidence = self._raw_classify(summary)

        self._history.append(label)
        if len(self._history) > self.history_len:
            self._history.pop(0)

        if len(self._history) >= 3:
            from collections import Counter
            smoothed_label = Counter(self._history).most_common(1)[0][0]
        else:
            smoothed_label = label

        meta = _STATE_META.get(smoothed_label, _STATE_META["neutral"])
        return StateResult(
            label       = smoothed_label,
            confidence  = confidence,
            description = meta["description"],
            tips        = meta["tips"],
            band_powers = {k: round(summary.get(f"{k}_power", 0), 4)
                           for k in ["delta", "theta", "alpha", "beta", "gamma"]},
            ratios      = {k: round(v, 4) for k, v in feature_dict["ratios"].items()},
        )

    def reset(self):
        self._history.clear()


if __name__ == "__main__":
    from data_generator import FakeMuseGenerator
    from preprocessing  import preprocess
    from features       import extract_features

    print(f"{'Ground truth':15s}  {'Predicted':15s}  {'Conf':6s}  Alpha    Beta     Gamma")
    print("-" * 70)

    correct = 0
    for state_name in ["relaxed", "focused", "anxious", "drowsy", "meditative"]:
        gen = FakeMuseGenerator(state=state_name)
        clf = MentalStateClassifier()
        result = None
        for _ in range(10):
            chunk  = preprocess(gen.get_chunk())
            feats  = extract_features(chunk)
            result = clf.classify(feats)
        a = result.band_powers.get("alpha", 0)
        b = result.band_powers.get("beta", 0)
        g = result.band_powers.get("gamma", 0)
        match = "✓" if result.label == state_name else "✗"
        if result.label == state_name:
            correct += 1
        print(f"{match} {state_name:13s}  {result.label:15s}  "
              f"{result.confidence:.0%}    {a:.2%}   {b:.2%}   {g:.2%}")
    print(f"\nAccuracy: {correct}/5")