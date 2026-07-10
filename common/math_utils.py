import numpy as np


def boltzmann(scores, beta):
    """
    Softmax choice probabilities under Boltzmann rationality.
    Same function as BoltzmannHumanOperator's kernel, kept standalone here
    so this module has no dependency on the rest of the operator/belief code.
    """
    scores = np.array(scores, dtype=float)
    probs = np.exp(beta * scores)
    return probs / probs.sum()


def shift_scores(nominal_scores, advised_idx, alpha):
    """
    Apply the additive advice bias: nominal scores with `alpha` added to the
    advised action's score. If advised_idx is None (DEFER — no advice issued),
    returns the nominal scores unchanged.
    """
    scores = list(nominal_scores)
    if advised_idx is not None:
        scores[advised_idx] = scores[advised_idx] + alpha
    return scores


def normalize_dict(vec):
    """ Normalize vector rep. by dict """
    total = sum(vec.values())
    if total == 0:
        raise ValueError("All probabilities collapsed to zero.")
    return {k: v/total for k, v in vec.items()}


def normalize_list(vec):
    total = sum(vec)
    if total == 0:
        raise ValueError("All probabilities collapsed to zero.")
    return [v/total for v in vec]


def normalize_distribution(vec):
    """ Normalize vector """
    if isinstance(vec, dict):
        return normalize_dict(vec)
    else:
        normalize_list(vec)

