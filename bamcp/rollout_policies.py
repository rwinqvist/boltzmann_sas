import random
from bamcp.history import History


def random_rollout(state, model):
    if isinstance(state, History):
        return random.choice(model.enabled_actions[state.last_state])
    else:
        return random.choice(model.enabled_actions[state])

