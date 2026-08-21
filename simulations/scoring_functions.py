import numpy as np

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



def generate_uav_scoring_function(domain, scale=1, bias_action=None, bias_strength=0.0, seed=None):
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







