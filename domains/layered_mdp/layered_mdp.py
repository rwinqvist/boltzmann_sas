import numpy as np 
import math 
from copy import deepcopy
import matplotlib.pyplot as plt


"""
    MDP instance: 
    S = {s0, s1, s2, s3, s4, s5, s6}
        * s0: init 
        * s4, s5, s6 absorbing states 

    A(s0) = {a1, a2, a3}
    A(s1) = {a4, a5, a6}
    A(s2) = {a7, a8, a9} 
    A(s2) = {a10, a11, a12} 

    R(s0) = 0
    R(s1) = 0.1
    R(s2) = 0 
    R(s3) = 0.2 
    R(s4) = 1
    R(s5) = 0
    R(s6) = epsilon


    Human (decision-maker) policy:  
    Sampled from a Boltzmann distribution: p1 ~ exp(beta*Phi(a1)) / ( exp(beta*Phi(a1)) + exp(beta*Phi(a2)) + exp(beta*Phi(a3)) )

# domain 
epsilon = -1

# human 
beta = 1    # rationality parameter 
alpha = 1   # compliance parameter, under additive advice bias

# states 
s0 = "s0"
s1 = "s1"
s2 = "s2"
s3 = "s3"
s4 = "s4"
s5 = "s5"
s6 = "s6"

# actions 
a1 = "a1"
a2 = "a2"
a3 = "a3"
a4 = "a4"
a5 = "a5"
a6 = "a6"
a7 = "a7"
a8 = "a8"
a9 = "a9"
a10 = "a10"
a11 = "a11"
a12 = "a12"

r0 = 0
r1 = 0.1 
r2 = 0 
r3 = 0.2 
r4 = 1
r5 = 0
r6 = epsilon

R = {s0: r0,
     s1: r1, 
     s2: r2, 
     s3: r3, 
     s4: r4, 
     s5: r5, 
     s6: r6,
    }


# Transitions 
T = {(s0, s1): a1,
     (s0, s2): a2,
     (s0, s3): a3,
     (s1, s4): a4,
     (s1, s5): a5,
     (s1, s6): a6,
     (s2, s4): a7,
     (s2, s5): a8,
     (s2, s6): a9,
     (s3, s4): a10,
     (s3, s5): a11,
     (s3, s6): a12,
    }


# paths 
paths = [(s0, s1, s4),
         (s0, s1, s5),
         (s0, s1, s6),
         (s0, s2, s4),
         (s0, s2, s5),
         (s0, s2, s6),
         (s0, s3, s4),
         (s0, s3, s5),
         (s0, s3, s6)]


# action scores 
# compute or hard code 
Phi_nom = {a1: 3, 
       a2: 0,
       a3: 2, 
       a4: 4, 
       a5: 1.5, 
       a6: 0,
       a7: 1,
       a8: 1,
       a9: 1,
       a10: 0,
       a11: 0.5,
       a12: 3}


# action scores 
# compute or hard code 
Phi_nom = {a1: 1, 
       a2: 3,
       a3: 1, 
       a4: 3, 
       a5: 3, 
       a6: 1}

"""


class LayeredMDP(object):
    """
        Class representing a general MDP instance of different sizes.
    """

    def __init__(self, depth, states, s0, actions, enabled_actions, goal_states, terminal_states, T_structure, T_det, state_rewards=None):
        self.domain_name = "layered_mdp"
        self.depth = depth
        self.states = states 
        self.s0 = s0
        self.actions = actions
        self.num_actions = len(actions)
        self.enabled_actions = enabled_actions
        self.T_structure = T_structure 
        self.T_det = T_det
        self.goal_states = goal_states 
        self.terminal_states = terminal_states 

        self.name_to_action = {f"a{aidx+1}": action for aidx, action in enumerate(self.actions)}
        self.action_to_name = {action: f"a{aidx+1}" for aidx, action in enumerate(self.actions)}
        self.state_to_name = {state: f"s{sidx}" for sidx, state in enumerate(self.states)}
        self.name_to_state = {f"s{sidx}": state for sidx, state in enumerate(self.states)}
        self.action_names = list(self.name_to_action.keys())

        self.nom_cost = 0
        self.state_rewards = state_rewards
        self.R = state_rewards

    @classmethod
    def generate_layered_mdp(cls, depth: int, num_actions: int = 3, state_rewards=None, seed=None):
        """
        Generate a layered MDP matching the structure of the toy example.

        Structure:
            - Layer 0: s0 (initial state)
            - Layer t (1..depth): num_actions states, all reachable from any state in layer t-1
            - Layer depth: terminal states
            - Actions a1..aK reused at every state, action ak always leads to k-th state in next layer
            - Rewards are state-dependent and randomly assigned per layer
            - Scoring function ΔΦ varies per state

        Total states: 1 + depth * num_actions

        Args:
            depth:          number of decision layers
            num_actions:    branching factor / number of states per layer
            seed:           random seed
            delta_phi_std:  std of ΔΦ across states
            reward_range:   (low, high) for uniform state reward sampling
        """

        actions = [f"a{k+1}" for k in range(num_actions)]

                # --- build state space ---
        all_states, leaf_states, layers = cls.build_state_space(depth, num_actions)


        # --- build dynamics ---
        # IF YOU WANT TO GO BACK, GENERATE STATE_REWARDS LIKE BEFORE (SEE DOWN BELOW) AND REMOVE IT FROM THE BUILD DYNAMICS
        T_structure, T_det, enabled_actions, state_rewards = cls.build_dynamics(layers, actions, leaf_states, seed=seed)

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        print(f"Generated layered MDP:")
        print(f"  Depth:              {depth}")
        print(f"  Num actions:        {num_actions}")
        print(f"  Total states:       {len(all_states)}")
        print(f"  Terminal states:    {len(leaf_states)}")

        #print("state rewards: ", state_rewards)
        #if state_rewards is None:
           # state_rewards = cls.generate_state_rewards(all_states, seed=seed)

        return cls(
            depth              = depth, 
            states             = all_states,
            s0                 = "s0",
            actions            = actions,
            enabled_actions    = enabled_actions,
            goal_states        = leaf_states,
            terminal_states    = leaf_states,
            T_structure        = T_structure,
            T_det              = T_det,
            state_rewards      = state_rewards,
        )
    

    def id_tag(self):
        """A short string identifying this domain instance for e.g. result paths."""
        return f"d{self.depth}_a{self.num_actions}"
    
    # @staticmethod
    # def generate_state_rewards(states, seed, discrete=True):
    #     rng = np.random.default_rng(seed)
    #     rewards = {}
    #     for state in states: 
    #         if discrete: 
    #             values = np.arange(-1, 1.1, 0.1)
    #             rewards[state] = rng.choice(values)
    #         else: 
    #             rewards[state] = rng.unform(-1, 1)
        
    #     return rewards

    @staticmethod
    def build_state_space(depth, num_actions):
        """Build states layer by layer."""
        layers = [["s0"]]
        state_counter = 1
        for _ in range(depth):
            layer = [f"s{state_counter + k}" for k in range(num_actions)]
            state_counter += num_actions
            layers.append(layer)

        all_states  = [s for layer in layers for s in layer]
        leaf_states = set(layers[-1])

        return all_states, leaf_states, layers
    

    @staticmethod
    def build_dynamics(layers, actions, leaf_states, seed=None):
        """Build transitions and rewards."""
        num_actions = len(actions)
        T_structure = {}
        T_det = {0: {}}
        enabled_actions    = {}
        state_rewards = {"s0": 0}   # initial state gives no reward

        rng = np.random.default_rng(seed)

        for layer_idx, layer in enumerate(layers[:-1]):
            next_layer = layers[layer_idx + 1]

            # fresh permutation of {-1, 0, 1}
            # rewards: permutation of {-1, 0, 1} — hardcoded for 3 actions
            assert num_actions == 3, "Structured rewards currently only support 3 actions"
            rewards = rng.permutation([-1.0, 0.0, 1.0])

            for sidx, state in enumerate(layer):
                enabled_actions[state] = actions
                T_structure[state] = []
                state_rewards[state] = rewards[sidx]

                for k, action in enumerate(actions):
                    next_state = next_layer[k]
                    # full transition dict with 0 probabilities for non-reached states
                    T_det[0][(state, action)] = {
                        s: (1.0 if s == next_state else 0.0) 
                        for s in next_layer
                    }

                    T_structure[state].append(next_state)
                    #print("State: ", state)
                    #print("Action: ", action)
                    #print("Next state: ", next_state)
                    #print(domain_transitions)
                    #input("Next...")

        # terminal states
        for state in leaf_states:
            enabled_actions[state] = []

        return T_structure, T_det, enabled_actions, state_rewards


    def get_next_states(self, state, action): 
        return self.T_structure[(state, action)]
    

    
    def get_reward(self, state, action, next_state):
        if next_state in self.state_rewards:
            return self.state_rewards[next_state]
        else:
            return 0
    
