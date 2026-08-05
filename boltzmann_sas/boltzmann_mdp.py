import logging 
import itertools
import random
from typing import Optional 
from operators.operator import Operator
from operators.utils import OperatorParams
from operators.human import BoltzmannHumanOperator
from operators.utils import OperatorType
from belief.observation import Observation
from boltzmann_sas.globals import DEFER


class BoltzmannMDP():
    """ Class representing the full SAS model with Boltzmann-rational human operators with known parameters beta and alpha.
    Represented by an MDP. 
    """

    def __init__(self, domain, operators:list[Operator]):
        logging.debug(f"Initializing the BoltzmannSAS MDP...")

        self.domain = domain 
        self.domain_states = domain.states 
        self.operators = operators 
        self.num_operators = len(operators)
        self.boltzmann_operators = [operator for operator in operators if operator.category == OperatorType.BHUMAN]
        self.boltzmann_operator_indices = {boltzmann_op: bidx for (bidx, boltzmann_op) in enumerate(self.boltzmann_operators)}

        self.domain_actions = domain.actions 
        self.domain_action_names = domain.action_names
        self.domain_name_to_action = domain.name_to_action 
        self.domain_action_to_name = domain.action_to_name
        self.domain_terminal_states = domain.terminal_states
        self.domain_goal_states = domain.goal_states
        self.domain_s0 = domain.s0

        # Build joint operator space 
        all_operator_states = [operator.states for operator in operators]
        self.joint_operator_states = list(itertools.product(*all_operator_states))


        # Build SAS state and action space
        self.states = [(domain_state, joint_op_state) for domain_state in self.domain_states for joint_op_state in self.joint_operator_states]
        self.terminal_states = [state for state in self.states if state[0] in self.domain_terminal_states]
        self.goal_states = [state for state in self.states if state[0] in self.domain_goal_states]

        self.actions = [(opidx, domain_action) for opidx in range(self.num_operators) for domain_action in self.domain_actions]
        self.action_to_index = {a: i for i, a in enumerate(self.actions)}
        self.index_to_action = {i: a for i, a in enumerate(self.actions)}
        self.enabled_actions = self.get_enabled_actions()

        # initial state 
        init_op_state = tuple([operator.x_init for operator in operators])
        self.s0 = (self.domain_s0, init_op_state)


        self.T = {}
        self.R = {}
        self.build_dynamics()

        logging.info("BoltzmannSAS MDP initialized.")


    def get_enabled_actions(self):
        enabled_actions = {}
        for state in self.states:
            domain_state, _ = state
            if domain_state in self.domain.enabled_actions: 
                enabled_domain_actions = self.domain.enabled_actions[domain_state]
                enabled_joint_actions = []
                for opidx, operator in enumerate(self.operators):
                    operator_actions = [(opidx, domain_action) for domain_action in enabled_domain_actions if domain_action in self.operators[opidx].actions]
                    if operator in self.boltzmann_operators:
                        operator_actions.append((opidx, DEFER))
                    enabled_joint_actions.extend(operator_actions)
                    
                enabled_actions[state] = enabled_joint_actions 
            else:
                enabled_actions[state] = []

        return enabled_actions


    def is_goal(self, state):
        return state in self.goal_states
    
    
    def is_terminal(self, state):
        return state in self.terminal_states
    
    
    def build_dynamics(self):
        """
            Build T and R
        """
        for state in self.states:
            if not self.is_goal(state) and not self.is_terminal(state):
                for action in self.enabled_actions[state]:
                    # action here is an allocation decision (operator, communication signal)
                    successors, rewards = self.build_transition_and_rewards(state, action)
                    self.T[(state, action)] = successors
                    for next_state, reward in rewards.items():
                        self.R[(state, action, next_state)] = reward
                

    def build_transition_and_rewards(self, state, issued_action):
        """
        Construct:
            1. P(s' | s, issued_action)
            2. E[r | s, issued_action, s']

        The issued action is an unresolved allocation decision (operator index, communication signal)
        """

        domain_state, joint_op_state = state 
        active_opidx, advice = issued_action 

        active_operator = self.operators[active_opidx]
        active_op_state = joint_op_state[active_opidx]

        action_likelihoods = active_operator.get_action_likelihoods(
            domain_state=domain_state,
            internal_state=active_op_state,
            issued_domain_action=advice,
        )

        # For each successor domain state, accuumulate: 
        #  * probability mass 
        #  * probability-weighted domain reward

        domain_probabilities = {}
        domain_reward_numerators = {}

        for executed_domain_action, p_action in action_likelihoods.items():
            domain_successors = active_operator.get_domain_transitions(domain_state, active_op_state, executed_domain_action)

            for next_domain_state, p_transition in domain_successors.items():
                weight = p_action * p_transition 

                realized_domain_reward = self.domain.get_reward(domain_state, executed_domain_action, next_domain_state)

                domain_probabilities[next_domain_state] = domain_probabilities.get(next_domain_state, 0) + weight

                domain_reward_numerators[next_domain_state] = domain_reward_numerators.get(next_domain_state, 0) + weight*realized_domain_reward


        # operator state transitions are independent of executed action 
        next_op_states = {}
        for opidx, operator in enumerate(self.operators):
            op_state = joint_op_state[opidx]
            is_active = operator == active_operator
            next_op_states[opidx] = operator.get_operator_state_transitions(op_state, is_active)

        next_joint_op_states = list(itertools.product(*next_op_states.values()))

        operator_cost = active_operator.get_operator_cost(active_op_state)

        successors = {}
        rewards = {}

        for next_domain_state, p_domain in domain_probabilities.items(): 
            conditional_domain_reward = domain_reward_numerators[next_domain_state] / p_domain 

            for next_joint_op_state in next_joint_op_states: 
                p_operator = 1 
                for opidx, next_op_state in enumerate(next_joint_op_state):
                    p_operator *= next_op_states[opidx][next_op_state]

                next_state = (next_domain_state, next_joint_op_state)
                probability = p_domain * p_operator 

                assert next_state in self.states

                successors[next_state] = (
                    successors.get(next_state, 0.0) + probability
                )

                rewards[next_state] = (
                    conditional_domain_reward - operator_cost
                )

        return successors, rewards
    


    def get_reward(self, state, action, next_state):
        """
            Returns the reward obtained from the executed domain action inside `action`.
        """
        
        domain_state, joint_op_state = state 
        active_op_idx, domain_action = action 
        active_op_state = joint_op_state[active_op_idx]
        active_operator = self.operators[active_op_idx]

        next_domain_state, next_joint_op_state = next_state

        # get rewards 
        r_domain = self.domain.get_reward(domain_state, domain_action, next_domain_state)
        op_cost = active_operator.get_operator_cost(active_op_state)

        #print("domain r: ", r_domain)
        #print("op cost: ", op_cost)
        #input("Continue...")
        r = r_domain - op_cost 
        
        #self.R[(state, action, next_state)] = r
        return r 

    
    def get_resolved_successors(self, state, resolved_action):
        """
            Get successor states for an ALREADY-RESOLVED (executed) action --
            compliance has already been decided by resolve_domain_action_choice,
            this just applies the domain's raw transition dynamics for that
            concrete action. Deliberately separate from get_successors()/self.T,
            which are for PLANNING and expect unresolved advice -- reusing them
            here would silently re-marginalize an already-resolved action as if
            it were fresh advice. Otherwise the VI generated policy is evaluated on 
            different dynamics.
        """
        successors = {}
        domain_state, joint_op_state = state
        active_opidx, domain_action = resolved_action
        active_operator = self.operators[active_opidx]
        active_op_state = joint_op_state[active_opidx]

        domain_transitions = active_operator.get_domain_transitions(domain_state, active_op_state, domain_action)
 
        next_op_states = {}
        for opidx in range(self.num_operators):
            operator = self.operators[opidx]
            op_state = joint_op_state[opidx]
            is_active = operator == active_operator
            next_op_states[opidx] = operator.get_operator_state_transitions(op_state, is_active, domain_action)
 
        next_joint_op_states = list(itertools.product(*next_op_states.values()))

        for next_domain_state, p_domain in domain_transitions.items():
            for next_joint_op_state in next_joint_op_states:
                next_state = (next_domain_state, next_joint_op_state)
                p_op = 1
                for opidx, next_op_state in enumerate(next_joint_op_state):
                    p_op *= next_op_states[opidx][next_op_state]
                p = p_domain * p_op
                assert next_state in self.states, f"Next state: {next_state} not in state space"
                successors[next_state] = p
 
        return successors


    def step(self, state, issued_action, op_parametrizations: Optional[dict[int, OperatorParams]] = None):
        domain_state, joint_op_state = state 
        opidx, advice = issued_action
        op_state = joint_op_state[opidx]
        active_operator = self.operators[opidx]
        op_state = joint_op_state[opidx]

        # if op_parametrizations is not None and opidx in op_parametrizations:
        #     op_params = op_parametrizations[opidx]
        # else:
        #     op_params = None
        
        # retrieve executed domain action (i.e., resolve action choice)
        op_params = None 
        if active_operator in self.boltzmann_operators and op_parametrizations is not None: 
            op_params = op_parametrizations[opidx]
            
        #print("In step parameters: ", op_params)
        executed_domain_action = active_operator.resolve_domain_action_choice(domain_state, op_state, advice, op_params)

        executed_action = (opidx, executed_domain_action)

        # resolved successors
        successors = self.get_resolved_successors(state, executed_action)
        states = list(successors.keys())
        likelihoods = list(successors.values())
        next_state = random.choices(states, likelihoods, k=1)[0]
        reward = self.get_reward(state, executed_action, next_state)
        
        return next_state, reward, executed_action 
    

