def generate_feedback(state, focus_score):
    if state == "Focused":
        return "High engagement detected. Maintain current activity."
    elif state == "Drowsy":
        return "Low alertness. Consider a break or stimulation."
    else:
        return "Stable state. No strong cognitive bias detected."