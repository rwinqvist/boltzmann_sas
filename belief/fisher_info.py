import numpy as np
from common.math_utils import boltzmann, shift_scores

def fisher_info_beta(scores, beta):
    """
    I(beta) = Var_p(s), where p = softmax(beta * scores).
 
    This is the NAIVE (single-parameter) Fisher information for beta: it
    assumes alpha (already baked into `scores` via shift_scores, if this is
    an ADVICE observation) is known. Under DEFER, `scores` has no alpha
    dependence at all, so this is also the correct marginal information.
    Under ADVICE, use fisher_info_joint() below for the honest marginal value.
    """
    p = boltzmann(scores, beta)
    scores = np.array(scores, dtype=float)
    Es = np.sum(p * scores)
    Es2 = np.sum(p * scores ** 2)
    return Es2 - Es ** 2


def fisher_info_alpha(scores, advised_idx, beta):
    """
    I(alpha) = beta^2 * p_k * (1 - p_k), where p_k is the EFFECTIVE
    (post-advice-shift) probability of the advised action.
 
    NAIVE (single-parameter): assumes beta is known. Only defined for ADVICE
    observations (advised_idx is not None) — DEFER carries no information
    about alpha at all, by construction of the model.
    """
    if advised_idx is None:
        return 0.0
    p = boltzmann(scores, beta)
    p_k = p[advised_idx]
    return (beta ** 2) * p_k * (1 - p_k)


def fisher_info_cross(scores, advised_idx, beta):
    """
    I(beta, alpha) = beta * p_k * (s_k - E_p[s]).
 
    The cross/off-diagonal term of the joint Fisher information matrix.
    Measures how entangled beta and alpha are in a given observation: large
    when advising an action whose (shifted) score sits far from the mean
    score (e.g. advising an already-strong action), meaning the observed
    outcome is ambiguous between "high beta" and "large alpha effect".
    Exactly zero under DEFER (alpha doesn't appear in the scores at all).
    """
    if advised_idx is None:
        return 0.0
    p = boltzmann(scores, beta)
    scores = np.array(scores, dtype=float)
    Es = np.sum(p * scores)
    return beta * p[advised_idx] * (scores[advised_idx] - Es)


def fisher_info_alpha_ceiling(beta):
    """
    Theoretical BEST-CASE per-step I(alpha), achieved when advice is chosen so
    the advised action's effective probability p_k = 0.5 (maximizing the
    Bernoulli-variance term p_k(1-p_k), which peaks at 0.25).
 
    I(alpha) = beta^2 * p_k(1-p_k) <= beta^2 * 0.25 always.
 
    Used as an optimistic forecast: "even with the best possible advice
    choice from here on, how much alpha-information could each remaining
    step contribute at most". Also note: this NAIVE ceiling upper-bounds the
    true MARGINAL alpha information too (I_alpha_marginal <= I_alpha_naive
    always, since I_alpha_marginal = I_alpha_naive - I_cross^2/I_beta), so
    using it as a forecast errs toward NOT collapsing prematurely -- the
    forecast is always at least as optimistic as reality, so if even this
    optimistic ceiling says further steps won't help, that conclusion is safe.
    """
    return (beta ** 2) * 0.25


def fisher_info_joint(nominal_scores, advised_idx, beta, alpha):
    """
    Compute the full joint (beta, alpha) Fisher information for a single
    observation, and the MARGINAL information for each parameter — i.e. the
    honest information about beta accounting for alpha being unknown too
    (and vice versa), via the standard 2x2 Cramer-Rao inverse.
 
    For DEFER (advised_idx=None): alpha terms are all 0/undefined, and the
    marginal beta information exactly equals the naive value (no entanglement).
 
    Returns a dict with: I_beta_naive, I_alpha_naive, I_cross,
    I_beta_marginal, I_alpha_marginal.
    """
    scores = shift_scores(nominal_scores, advised_idx, alpha)
 
    I_beta = fisher_info_beta(scores, beta)
 
    if advised_idx is None:
        return {
            "I_beta_naive": I_beta,
            "I_alpha_naive": 0.0,
            "I_cross": 0.0,
            "I_beta_marginal": I_beta,   # no entanglement under DEFER
            "I_alpha_marginal": 0.0,     # DEFER carries no alpha information
        }
 
    I_alpha = fisher_info_alpha(scores, advised_idx, beta)
    I_cross = fisher_info_cross(scores, advised_idx, beta)
 
    det = I_beta * I_alpha - I_cross ** 2
    # guard against numerical edge cases (e.g. I_alpha or I_beta ~ 0)
    I_beta_marginal = det / I_alpha if I_alpha > 1e-12 else 0.0
    I_alpha_marginal = det / I_beta if I_beta > 1e-12 else 0.0
 
    return {
        "I_beta_naive": I_beta,
        "I_alpha_naive": I_alpha,
        "I_cross": I_cross,
        "I_beta_marginal": I_beta_marginal,
        "I_alpha_marginal": I_alpha_marginal,
    }