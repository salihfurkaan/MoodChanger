"""
feedback.py
Translates StateResult objects into rich, user-facing text feedback.

Provides:
- Short one-liner summary
- Detailed paragraph explanation referencing the actual band powers
- Personalised tips
- Session-level trend analysis (if history is provided)
"""

from typing import List
from states import StateResult


# ---------------------------------------------------------------------------
# Emoji map for terminal / web rendering
# ---------------------------------------------------------------------------
_STATE_EMOJI = {
    "relaxed":    "😌",
    "focused":    "🧠",
    "anxious":    "😰",
    "drowsy":     "😴",
    "meditative": "🧘",
    "neutral":    "😐",
}

_BAND_LABELS = {
    "delta": "Delta (0.5–4 Hz)",
    "theta": "Theta (4–8 Hz)",
    "alpha": "Alpha (8–13 Hz)",
    "beta":  "Beta (13–30 Hz)",
    "gamma": "Gamma (30–45 Hz)",
}


# ---------------------------------------------------------------------------
# Per-result feedback
# ---------------------------------------------------------------------------

def format_result(result: StateResult, verbose: bool = True) -> str:
    """
    Format a StateResult as a human-readable string.

    Args:
        result : StateResult from MentalStateClassifier.classify()
        verbose: include band-power breakdown and tips

    Returns:
        Formatted string suitable for terminal output.
    """
    emoji = _STATE_EMOJI.get(result.label, "🔬")
    lines = [
        f"{emoji}  Mental State: {result.label.upper()}  "
        f"(confidence: {result.confidence:.0%})",
        f"   {result.description}",
    ]

    if verbose:
        lines.append("\n📊  Band Power Breakdown:")
        for band, val in result.band_powers.items():
            bar = "█" * int(val * 30)
            lines.append(f"   {_BAND_LABELS[band]:22s}  {bar:<30s}  {val:.2%}")

        lines.append("\n⚡  Key Ratios:")
        interesting = ["alpha_beta", "theta_alpha", "engagement", "gamma_dominance"]
        for k in interesting:
            v = result.ratios.get(k, 0)
            lines.append(f"   {k:25s}: {v:.3f}")

        lines.append("\n💡  Tips:")
        for tip in result.tips:
            lines.append(f"   • {tip}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session-level trend analysis
# ---------------------------------------------------------------------------

def session_summary(history: List[StateResult]) -> str:
    """
    Summarise a list of StateResult objects from a session.

    Args:
        history: list of StateResult, ordered chronologically.

    Returns:
        Human-readable session summary string.
    """
    if not history:
        return "No session data available."

    from collections import Counter
    counts   = Counter(r.label for r in history)
    dominant = counts.most_common(1)[0][0]
    total    = len(history)

    # Average band powers across session
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    avg_powers = {}
    for b in bands:
        vals = [r.band_powers.get(b, 0) for r in history]
        avg_powers[b] = sum(vals) / len(vals) if vals else 0.0

    # Trend: compare first and last thirds
    third = max(1, total // 3)
    early = history[:third]
    late  = history[-third:]

    def mean_alpha(subset):
        return sum(r.band_powers.get("alpha", 0) for r in subset) / len(subset)

    alpha_trend = mean_alpha(late) - mean_alpha(early)
    trend_str = (
        "↑ increasing relaxation" if alpha_trend >  0.03 else
        "↓ decreasing relaxation" if alpha_trend < -0.03 else
        "→ stable"
    )

    lines = [
        "=" * 50,
        "📋  SESSION SUMMARY",
        "=" * 50,
        f"   Duration     : {total} chunks analysed",
        f"   Dominant state: {dominant.upper()} ({counts[dominant]/total:.0%} of session)",
        "",
        "   State distribution:",
    ]
    for label, cnt in counts.most_common():
        bar = "█" * int((cnt / total) * 20)
        lines.append(f"   {label:12s} {bar:<20s}  {cnt/total:.0%}")

    lines += [
        "",
        "   Average band powers:",
    ]
    for b in bands:
        lines.append(f"   {_BAND_LABELS[b]:22s}: {avg_powers[b]:.2%}")

    lines += [
        "",
        f"   Alpha trend     : {trend_str}",
        "=" * 50,
    ]

    # Personalised session recommendation
    if dominant in ("anxious",):
        lines.append("🔴  High stress detected throughout session. Prioritise rest and stress management.")
    elif dominant in ("relaxed", "meditative"):
        lines.append("🟢  Great session! Your brain was in a restful, restorative state.")
    elif dominant in ("focused",):
        lines.append("🟡  Sustained focus detected. Remember to schedule recovery breaks.")
    elif dominant in ("drowsy",):
        lines.append("🟠  Significant drowsiness detected. Sleep quality may need attention.")
    else:
        lines.append("⚪  Mixed session — review individual state epochs for more detail.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import FakeMuseGenerator
    from preprocessing  import preprocess
    from features       import extract_features
    from states         import MentalStateClassifier

    classifier = MentalStateClassifier()
    history    = []

    print("Running 10-chunk session simulation (state: anxious)...\n")
    gen = FakeMuseGenerator(state="anxious")
    for _ in range(10):
        chunk  = preprocess(gen.get_chunk())
        feats  = extract_features(chunk)
        result = classifier.classify(feats)
        history.append(result)

    print(format_result(history[-1]))
    print()
    print(session_summary(history))