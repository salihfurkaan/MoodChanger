def classify_state(features):
    """
    features: dict with band powers
    """

    beta = features["rel_beta"]
    alpha = features["rel_alpha"]
    theta = features["rel_theta"]

    if beta > alpha and beta > theta:
        return "Focused"
    elif theta > beta:
        return "Drowsy"
    else:
        return "Neutral"