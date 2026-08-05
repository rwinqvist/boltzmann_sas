import math 
import random
from typing import Optional
from operators.operator import Operator
from operators.utils import OperatorType, OperatorParams
from operators.context import OperatorContext
from boltzmann_sas.globals import DEFER
from belief.belief import Belief

class HumanOperator(Operator):
    """
    Class representing an Autonomous Operator.
    """

    def __init__(self, num_states, actions, nom_cost=0):
        """
        Initialization of Autonomous Operator instance.
        
        :param num_states (int): number of performance states 
        :param actions (list): available actions 
        :param nom_performance (float): nominal performance rate (domain action success rate)
        :param nom_resilience_rate (float): nominal resilience rate (not evolving to a worse performance state when active)
        :param nom_recovery_rate (float): nominal recovery rate (evolving to a better performance state when dormant)
        :param nom_cost (float): nominal cost for agent when in optimal/best performance state 
        :param degradation_rate (float): the rate at which the performance degrades as the agent performance state evolves
        :param roc (str): rate of change of performance state evolution
        """

        super().__init__(OperatorType.HUMAN, num_states, actions, nom_cost)




class BoltzmannHumanOperator(Operator):
    """
    Class representing an Autonomous Operator.
    """

    def __init__(self, num_states, actions, enabled_actions:dict, domain_transitions:dict, nom_scoring:dict, true_params:OperatorParams, init_belief: Belief, nom_cost=0):
        """
        Initialization of Autonomous Operator instance.
        
        :param num_states (int): number of performance states 
        :param actions (list): available actions 
        :param nom_performance (float): nominal performance rate (domain action success rate)
        :param nom_resilience_rate (float): nominal resilience rate (not evolving to a worse performance state when active)
        :param nom_recovery_rate (float): nominal recovery rate (evolving to a better performance state when dormant)
        :param nom_scoring (mapping): nominal action scoring function
        :param params (OperatorParams): rationality parameter (beta) and compliance parameter (alpha)
        :param nom_cost (float): nominal cost for agent when in optimal/best performance state 
        :param degradation_rate (float): the rate at which the performance degrades as the agent performance state evolves
        :param roc (str):  
        """

        super().__init__(OperatorType.BHUMAN, num_states, actions, enabled_actions, domain_transitions, nom_cost=nom_cost)
        
        self.true_params = true_params
        self.nom_beta = true_params.beta
        self.nom_scoring = nom_scoring 
        self.alpha = true_params.alpha
        self.init_belief = init_belief
        self.planning_params = None

        if self.nom_scoring is None:
            raise ValueError("Scoring function for Boltzmann human operator cannot be None!")


    def __str__(self): 
        obj_str = f"Operator: {self.category} " + \
                    f"\nStates: {self.states}" + \
                    f"\nNominal rationality: {self.nom_beta}" + \
                    f"\nNominal compliance: {self.alpha}" + \
                    f"\nNom performance: {self.nom_performance}" + \
                    f"\nNom resilience: {self.nom_resilience_rate}" + \
                    f"\nNom recovery: {self.nom_recovery_rate}"
        return obj_str

    

    @classmethod
    def from_context(cls, operator_context: OperatorContext):
        """
        Construct operator from an OperatorContext object.
        """
        num_states = operator_context.n
        #nom_performance = {}
        #nom_resilience_rate = {}
        #nom_recovery_rate = {}
        #nom_cost = {}
        actions = operator_context.actions
        enabled_actions = operator_context.enabled_actions
        nom_scoring = operator_context.nom_scoring
        #for action in actions:
            #nom_performance[action] = operator_context.transition_nominals[action][TransitionType.T]
            #nom_resilience_rate[action] = operator_context.transition_nominals[action][TransitionType.TAU]
            #nom_recovery_rate[action] = operator_context.transition_nominals[action][TransitionType.REC]
            #nom_cost[action] = operator_context.cost_nominals[action]

        init_belief = operator_context.init_belief
        params = operator_context.params
        domain_transitions = operator_context.domain_transitions

        return cls(num_states=num_states, actions=actions, enabled_actions=enabled_actions, domain_transitions=domain_transitions, nom_scoring=nom_scoring, true_params=params, init_belief=init_belief)
    


    @property
    def beta(self):
        return self.nom_beta ** self.state
    

    def set_planning_params(self, planning_params):
        self.planning_params = planning_params
    

    def get_action_likelihoods(self, domain_state, internal_state, issued_domain_action, parameters=None):
        enabled_actions = self.enabled_actions[domain_state]
        likelihoods = {}
        params = parameters if parameters is not None else self.true_params

        for action in enabled_actions:
            p = self.boltzmann_kernel(domain_state, internal_state, action, issued_domain_action, params)
            likelihoods[action] = p 
        
        return likelihoods


    def resolve_domain_action_choice(self, domain_state, op_state, issued_domain_action, params: Optional[OperatorParams] = None):
        """ Resolve operator's domain action choice based on suggested `issued_domain_action`. Based on Boltzmann kernel. """

        params = params if params is not None else self.true_params
        
        action_weights = []
        for action in self.enabled_actions[domain_state]: 
            weight = self.boltzmann_kernel(domain_state, op_state, action, issued_domain_action, params)
            action_weights.append(weight)
        
        # sample action choice randomly 
        #selected_action = random.choice(self.enabled_actions[domain_state], p=action_weights)
        selected_action = random.choices(self.enabled_actions[domain_state], weights=action_weights, k=1)[0]
        return selected_action


    def boltzmann_kernel(self, domain_state, op_state, action, advice, params: OperatorParams):
        """
        Docstring for boltzmann_kernel

        """
        
        # PRINT
        #print("domain state: ", domain_state)
        #print("op state: ", op_state)
        #print("action: ", action)
        #print("advice: ", advice)
        #print("params: ", params)

        action_score = self.scoring_function(domain_state, op_state, action, advice, params.alpha)
        num = math.exp(params.beta*action_score)

        denom = 0 
        for alt_action in self.enabled_actions[domain_state]:
            alt_action_score = self.scoring_function(domain_state, op_state, alt_action, advice, params.alpha)
            denom += math.exp(params.beta*alt_action_score)
        
        p = num/denom 

        return p 
    

    def scoring_function(self, domain_state, op_state, action, advice, alpha):
        if advice == DEFER: 
            # use nominal scoring 
            return self.nom_scoring[op_state][(domain_state, action)]
        else:
            # use effective scoring 
            # additive bias for now 
            # PRINT
            #print("in here")
            #print("domain state: ", domain_state)
            return self.nom_scoring[op_state][(domain_state, action)] + alpha*(action == advice)





    