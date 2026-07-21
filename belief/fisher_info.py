import numpy as np
from common.math_utils import boltzmann, shift_scores
from belief.observation import Observation
from boltzmann_sas.globals import DEFER


def fisher_info_defer(nominal_scores, beta):
    """
        I_0(theta)(s;beta), per-observation Fisher information contribution from a single
        DEFER observation. Only the beta-beta entry (A, in Proposition 1) is non-zero, 
        since DEFER carries no information about alpha.

        Returns a 2x2 array [[A_i, 0], [0, 0]], A_i = Var_nu(Phi)
        under nu = softmax(beta*nominal_scores)
    """

    p = boltzmann(nominal_scores, beta)
    s = np.asarray(nominal_scores, dtype="float")
    E_s = np.sum(p*s)
    E_s2 = np.sum(p*s**2)
    A_i = E_s2 - E_s**2

    return np.array([[A_i, 0], [0, 0]])


def fisher_info_advice(nominal_scores, advised_idx, beta, alpha):
    """
    I_1(theta)(s, a_D; beta, alpha). Per-observation Fisher information contribution 
    from a single ADVICE observation (advised_idx = index of advised action). Contributes to 
    all three entries A, B, C since alpha is identifiable (in principle) from ADVICE observations.

    effective_scores = nominal_score + alpha*g, where g is the indicator of the advised action.
    nu_tilde = softmax(beta*effective_scores)

    Returns a 2x2 array [[A_j, C_j], [C_j, B_j]]:
    A_j = Var_nu_tilde(effective_scores)
    B_j = beta^2 * p_k * (1 - p_k),  p_k = nu_tilde(advised action)
    C_j = beta * p_k * (s_k - E_nu_tilde[effective_scores])
    """

    effective_scores = np.asarray(nominal_scores, dtype="float").copy()
    effective_scores[advised_idx] += alpha
    p = boltzmann(effective_scores, beta)
    E_s = np.sum(p * effective_scores)
    E_s2 = np.sum(p * effective_scores ** 2)
    A_j = E_s2 - E_s ** 2

    p_k = p[advised_idx]
    B_j = (beta ** 2) * p_k * (1 - p_k)

    s_k = effective_scores[advised_idx]
    C_j = beta * p_k * (s_k - E_s)

    return np.array([[A_j, C_j], [C_j, B_j]])


def observation_to_scores(obs: Observation, enabled_actions:dict):
    """
    Adapter: extract (nominal_scores, advised_idx) from an Observation,
    in the form fisher_info_defer or form fisher_info_advice expect.

    nominal_scores: list of Phi values, one per enabled action at obs's domain_state
    advised_idx; index into that list of the advised issued action, or None if this was a DEFER observation
    """
    actions = enabled_actions[obs.domain_state]
    nominal_scores = [
        obs.operator.nom_scoring[obs.op_state][(obs.domain_state, a)]
        for a in actions
    ]

    if obs.issued_domain_action == DEFER:
        advised_idx = None 
    else: 
        advised_idx = actions.index(obs.issued_domain_action)

    return nominal_scores, advised_idx


def accumulate_fisher_info(observations, beta_hat, alpha_hat, enabled_actions):
    """
    Sum per-observation Fisher information contributions over a list of observations, evaluated
    at the given point estimate (beta_hat, alpha_hat). Returns the total 2x2 FIM [[A, C], [C, B]]
    """

    total = np.zeros((2, 2))
    for obs in observations: 
        nominal_scores, advised_idx = observation_to_scores(obs, enabled_actions)
        if advised_idx is None:
            total += fisher_info_defer(nominal_scores, beta_hat)
        else:
            total += fisher_info_advice(nominal_scores, advised_idx, beta_hat, alpha_hat)

    return total 


def marginal_variances(fisher_matrix):
    """
    Extract marginal Var(beta) and Var(alpha) from the 
    accumulaeted 2x2 FIM [[A, C], [C, B]] via the standard 2x2 inverse
    Var(beta) = B / (A*B-C^2)
    Var(alpha) = A / (A*B-C^2)

    Returns (var_beta, var_alpha) or (inf, inf) if the matrix is 
    singular or near-singular.
    """
    A, C = fisher_matrix[0, 0], fisher_matrix[0, 1]
    B = fisher_matrix[1, 1]
    det = A*B - C**2
    if det <= 0:
        #print("Det of FIM is: ", det)
        return np.inf, np.inf

    var_beta = B/det 
    var_alpha= A/det 
    return var_beta, var_alpha
