import numpy as np
from domains.uav.enums import DomainAction
from domains.uav.uav_domain import UAVDomain

def generate_scoring_function(domain, scale=1, seed=None):
    """
    Generate scoring function for Boltzmann human operator.
    Each state gets a fresh random permutation of {0, 1, 2} across its 3 actions.
    Scores are distinct per state and independent of state rewards.
    """
    assert domain.num_actions == 3, "Structured scoring currently only supports 3 actions"
    
    rng = np.random.default_rng(seed)
    Phi_nom = {0: {}}
    
    for state in domain.states:
        if state in domain.terminal_states:
            continue
        actions = domain.enabled_actions[state]
        
        # fresh permutation of {0,1,2} per state
        scores = rng.permutation([0.0, 1.0, 2.0])*scale
        #print(f"State {state}: scores={scores}")  # debug
        for k, action in enumerate(actions):
            Phi_nom[0][(state, action)] = float(scores[k])
    
    return Phi_nom


# def generate_progression_biased_uav_scoring_function(domain: UAVDomain, scale=1, seed=None):
#     """
#     Human's nominal preference favors RIGHT and DOWN equally, whenever
#     enabled, over LEFT/UP.

#     RIGHT and DOWN share the SAME top score when both enabled -- not one
#     arbitrarily preferred over the other -- so the human's default,
#     no-advice behavior splits evenly between the two progress-making
#     directions rather than favoring one for no principled reason.

#     NOTE: assumes the goal is at the right bottom corner of the UAV domain. 
#     """

#     rng = np.random.default_rng(seed)
#     Phi_nom = {0:{}}
#     PROGRESS_ACTIONS = {DomainAction.RIGHT, DomainAction.DOWN}

#     for state in domain.states: 
#         if state in domain.terminal_states: 
#             continue 
#         actions = domain.enabled_actions[state]
#         n_actions = len(actions)

#         progressing = [a for a in actions if a in PROGRESS_ACTIONS]
#         other_actions = [a for a in actions if a not in PROGRESS_ACTIONS]

#         top_score = (n_actions - 1)*scale 
#         scores = {a: top_score for a in progressing}

#         if other_actions:
#             other_scores = rng.permutation(np.arange(len(other_actions), dtype=float)) * scale
#             scores.update(zip(other_actions, other_scores))

#         for action in actions: 
#             Phi_nom[0][(state, action)] = float(scores[action])

#     return Phi_nom

def generate_progression_biased_uav_scoring_function(domain: UAVDomain, scale=1, seed=None):
    """
    Human's nominal preference favors making progress toward the goal, but
    not deterministically: at each state, ONE of {RIGHT, DOWN} (whichever
    are enabled) is chosen at random to be the strict top-scoring action.
    The other progress-making direction is NOT given special treatment --
    it's folded into the same randomized ranking as every other action, so
    it's no more likely to score highly than LEFT/UP.

    This keeps the human's default, no-advice behavior progression-biased
    on average (some progress-making action is guaranteed top at every
    state) without forcing RIGHT and DOWN to be jointly optimal, which made
    the nominal policy behave almost greedily toward the goal.

    NOTE: assumes the goal is at the right bottom corner of the UAV domain.
    """

    rng = np.random.default_rng(seed)
    Phi_nom = {0: {}}
    PROGRESS_ACTIONS = {DomainAction.RIGHT, DomainAction.DOWN}

    for state in domain.states:
        if state in domain.terminal_states:
            continue
        actions = domain.enabled_actions[state]
        n_actions = len(actions)

        progressing = [a for a in actions if a in PROGRESS_ACTIONS]

        if progressing:
            # randomly pick ONE progress-making action to be the strict top
            top_action = progressing[rng.integers(len(progressing))]
            remaining = [a for a in actions if a != top_action]

            top_score = (n_actions - 1) * scale
            scores = {top_action: top_score}

            if remaining:
                remaining_scores = rng.permutation(np.arange(len(remaining), dtype=float)) * scale
                scores.update(zip(remaining, remaining_scores))
        else:
            # neither RIGHT nor DOWN enabled here -- just rank everything
            scores = dict(zip(actions, rng.permutation(np.arange(n_actions, dtype=float)) * scale))

        for action in actions:
            Phi_nom[0][(state, action)] = float(scores[action])

    return Phi_nom


def generate_random_uav_scoring_function(domain, scale=1, bias_action=None, seed=None):
    """
        Generate scoring function for Boltzmann human operator in UAV domain. 
        Each state gets a random assignment of scores across its enabled 
        actions (same idea as generate_scoring_function for the layered MDP). 
        If bias_action is given, it is GUARANTEED the highest score at every state.
        The remaining actions get a random permutation of the leftover scores.

        Note that this sets the human's NOMINAL utility, not the effective one. 
        Under advice, any action can thus still be ranked the highest.
    """
    rng = np.random.default_rng(seed)
    Phi_nom = {0: {}}

    for state in domain.states: 
        if state in domain.terminal_states:
            continue 
        actions = domain.enabled_actions[state]
        n_actions = len(actions)

        if bias_action is not None and bias_action in actions: 
            other_actions = [a for a in actions if a != bias_action]
            other_scores = rng.permutation(np.arange(n_actions - 1, dtype=float)) * scale
            scores = {bias_action: (n_actions - 1)*scale}
            scores.update(zip(other_actions, other_scores))
        else:
            base_scores = rng.permutation(np.arange(n_actions, dtype=float)) * scale
            scores = dict(zip(actions, base_scores))

        for action in actions: 
            Phi_nom[0][(state, action)] = float(scores[action])


    return Phi_nom







