import numpy as np

def generate_scoring_function(domain, seed=None):
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
        scores = rng.permutation([0.0, 1.0, 2.0])
        #print(f"State {state}: scores={scores}")  # debug
        for k, action in enumerate(actions):
            Phi_nom[0][(state, action)] = float(scores[k])
    
    return Phi_nom