from operators.operator import Operator
from operators.context import OperatorContext
from operators.utils import OperatorType

class AutonomousOperator(Operator):
    """
    Class representing an Autonomous Operator.
    """

    def __init__(self, num_states, actions, enabled_actions:dict, domain_transitions:dict, nom_cost=0):
        """
        Initialization of Autonomous Operator instance.
        
        :param num_states (int): number of performance states 
        :param nom_performance (float): nominal performance rate (domain action success rate)
        :param nom_resilience_rate (float): nominal resilience rate (not evolving to a worse performance state when active)
        :param nom_recovery_rate (float): nominal recovery rate (evolving to a better performance state when dormant)
        :param degradation_rate (float): the rate at which the performance degrades as the agent performance state evolves
        :param nom_cost (float): nominal cost for agent when in optimal/best performance state 
        :param roc (str):  
        """

        super().__init__(OperatorType.AUTO, num_states, actions, enabled_actions=enabled_actions, domain_transitions=domain_transitions, nom_cost=nom_cost)
        

    @classmethod
    def from_context(cls, operator_context: OperatorContext):
        """
        Construct operator from an OperatorContext object.
        """
        num_states = operator_context.n
        #nom_performance = {}
        #nom_resilience_rate = {}
        #nom_recovery_rate = {}
        nom_cost = {}
        actions = operator_context.actions
        enabled_actions = operator_context.enabled_actions
        domain_transitions = operator_context.domain_transitions
        #for action in actions:
            #nom_performance[action] = operator_context.transition_nominals[action][TransitionType.T]
            #nom_resilience_rate[action] = operator_context.transition_nominals[action][TransitionType.TAU]
            #nom_recovery_rate[action] = operator_context.transition_nominals[action][TransitionType.REC]
            #nom_cost[action] = operator_context.cost_nominals[action]

        return cls(num_states, actions, enabled_actions, domain_transitions)
    
    

