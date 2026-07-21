from boltzmann_sas.globals import DEFER

def unpack_action(model, action): 
    is_defer, is_advice, is_auto = 0, 0, 0

    opidx, advice = action 
    if not opidx in model.boltzmann_operator_indices.values():
        is_auto = 1         
    else: 
        if advice == DEFER:
            is_defer = 1 
        else: 
            is_advice = 1

    return is_advice, is_defer, is_auto