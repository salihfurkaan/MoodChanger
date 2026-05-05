def compute_relative_powers(alpha, beta, theta):
    """
    Compute normalized (relative) band powers.
    Returns tuple: (rel_alpha, rel_beta, rel_theta)
    """
    total = alpha + beta + theta

    if total == 0:
        return 0, 0, 0

    rel_alpha = alpha / total
    rel_beta = beta / total
    rel_theta = theta / total

    return rel_alpha, rel_beta, rel_theta


def compute_focus_score(alpha, beta, theta):
    """
    Compute focus score using beta vs (alpha + theta).
    More stable than beta/alpha.
    """
    denom = alpha + theta

    if denom == 0:
        return 0

    return beta / denom


def classify_state(alpha, beta, theta):
    """
    Classify mental state based on relative band powers.
    This is a heuristic (not clinical).
    """

    rel_alpha, rel_beta, rel_theta = compute_relative_powers(alpha, beta, theta)

    # Relaxed: alpha dominates
    if rel_alpha > 0.45:
        return "Relaxed"

    # Highly active: beta strongly dominates
    elif rel_beta > 0.6:
        return "Highly Active"

    # Theta-heavy (could indicate drowsiness)
    elif rel_theta > 0.4:
        return "Drowsy"

    # Otherwise balanced state
    else:
        return "Neutral"