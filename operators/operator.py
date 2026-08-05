import numpy as np
import random
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from operators.context import OperatorContext

class Operator(ABC): 
    """
        Class representing Operator. Contains information about performance states, 
        transitions between performance states, nominal performance, performance vfariation etc. 
    """

    # some params used to compute performance variations
    ALPHA, BETA, GAMMA, K= 0.05, 0.05, 0.05, 0.25

    def __init__(self, category, num_states, actions, enabled_actions:dict, domain_transitions: dict, nom_cost=1):
        """
        Docstring for __init__
        
        :param category: type of operator, either (B)HUMAN or AUTO(nomouos)
        :param num_states: the number of performance states
        :param nom_performance: nominal performance (domain action success rate)
        :param nom_resilience_rate: nominal resilience rate (not evolving to a worse performance state when active)
        :param nom_recovery_rate: nominal recovery rate (evolving to a better performance state when dormant) 
        :param degradation_rate: 
        :param nom_cost: nominal operator cost (in optimal performance state)
        :param roc: 
        """
        
        self.category = category
        self.num_states = num_states
        self.actions = set(actions)
        self.states = [i for i in range(self.num_states)] 
        self.x_init = self.states[0]
        self.x_min, self.x_max = self.states[0], self.states[-1]
        self.state = self.x_init
        self.domain_transitions = domain_transitions
        self.enabled_actions = enabled_actions

        self.nom_cost = nom_cost       # nominal operator cost (in best performance state)
        self.cost_reduction_rate = 0.8      # how much cost lowers when performance lowers

        #self.decay_rates = self.compute_decay_rates()
        self.cost_function = self.compute_cost_function()
        

    @classmethod
    @abstractmethod
    def from_context(cls, operator_context: OperatorContext):
        """
            Construct operator from OperatorContext object.
        """

    def copy(self):
        return deepcopy(self)

    def get_action_likelihoods(self, domain_state, internal_state, issued_domain_action, parameters=None):
        enabled_actions = self.enabled_actions[domain_state]
        likelihoods = {}
        for action in enabled_actions:
            if action == issued_domain_action:
                likelihoods[action] = 1 
            else: 
                likelihoods[action] = 0

        return likelihoods


    def resolve_domain_action_choice(self, domain_state, internal_state, issued_domain_action, parameter=None):
        """ Resolve operator's domain action choice based on suggested `issued_domain_action`. Defaults to full compliance and rationality, i.e., issued_domain_action. """
        return issued_domain_action


    def get_domain_transitions(self, domain_state, internal_state, domain_action):
        """ Get domain transition likelihoods """
        #print("Domain transitions: ")
        #print(self.domain_transitions)
        return self.domain_transitions[internal_state][domain_state, domain_action]
    

    def get_operator_state_transitions(self, state, is_active, action=None):
        """
        Compute operator performance state transitions. 

        Args: 
            * state (int): current performance state
            * is_active (bool): Whether the operator is currently active (True) or dormant (False)

        Returns:
            * dict: a mapping of possible next performance states and their likelihoods. 
        """

        if self.num_states == 1:
            next_states = [state]
            likelihoods = [1]

        return dict(zip(next_states, likelihoods))

    

    def get_operator_cost(self, state): 
        """
        Compute the cost of assigning control to an operator in a given performance state.

        Args: 
            * state (int): current performance state

        Returns:
            * float: the computed cost.
        """
        cost = self.cost_function[state]
        return cost 
    

    
    def compute_cost_function(self):
        """
        Computes cost function for operator. 

        The cost decreases linearly as performance state increases, based on the
        formula: cost_rate = 1 - (state / num_states).
        """
        cf = {}
        for s in self.states:
            cost_rate = 1 - s/len(self.states)  # linear rate of change as performance goes down
            cf[s] = self.nom_cost*cost_rate

        return cf
    


    def simulate_step(self, state, is_active, p):
        if is_active: 
            next_states = [state, min(state+1, self.x_max)]
        else: 
            next_states = [max(state-1, self.x_min), state]

        likelihoods = [p, 1-p]
        next_state = random.choices(next_states, likelihoods)[0]

        return next_state
    