import math

def value_iteration(model, gamma=1, epsilon=0.001, max_iter=1000):
    Q, V, policy = {}, {}, {}
    for state in model.states:
        V[state] = 0
        policy[state] = None 
        for action in model.enabled_actions[state]:
            Q[(state, action)] = 0

    #while delta > epsilon and iter < max_iter:
    for iter in range(max_iter):
        delta = 0

        for state in model.states:
            if model.is_terminal(state) or model.is_goal(state):
                continue

            v = V[state] 
            q_max = -math.inf 
            a_opt = None 

            for action in model.enabled_actions[state]:
                q = 0
                for next_state, p in model.T[(state, action)].items():
                    r = model.R[(state, action, next_state)]
                    q += p * (r + gamma * V[next_state])

                Q[(state, action)] = q 

                if q > q_max:
                    q_max = q 
                    a_opt = action 
            
            policy[state] = a_opt 
            V[state] = q_max 
            delta = max(delta, abs(v-q_max))

        if delta < epsilon:
            break
    
    return Q, V, policy

