import numpy as np


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

