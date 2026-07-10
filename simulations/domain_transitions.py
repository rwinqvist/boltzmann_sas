def build_auto_domain_transitions(domain, p_success):
    """
    Build domain_transitions for a stochastic-performance AutonomousOperator:
    with probability p_success, the issued action succeeds and moves to the
    intended next state (same outcome domain.T_det would give); with
    probability (1-p_success), the action fails and the domain state stays
    unchanged (no progress this step).
 
    :param domain: needs domain.T_det[0][(state, action)] -- a
        {next_state: probability} dict, same structure used elsewhere in this
        codebase (e.g. OperatorContext(domain_transitions=domain.T_det)).
    :param p_success: probability the action succeeds. Either a single float
        (same reliability everywhere) or a dict keyed by (domain_state, action)
        for state/action-dependent reliability.
 
    Returns a dict shaped {0: {(state, action): {next_state: prob, ...}}} --
    the {0: ...} wrapper matches internal_state=0, since this operator only
    ever has one performance state (num_states=1).
    """
    auto_transitions = {}
    for (state, action), next_state_probs in domain.T_det[0].items():
        intended_next_state = max(next_state_probs, key=next_state_probs.get)  # the state with probability 1.0
        p = p_success[(state, action)] if isinstance(p_success, dict) else p_success

        auto_transitions[(state, action)] = {}
 
        for next_state in next_state_probs.keys():
            if next_state == intended_next_state:
                auto_transitions[(state, action)][next_state] = p
            else:
                auto_transitions[(state, action)][next_state] = (1-p)/(len(next_state_probs)-1)

    return {0: auto_transitions}
 