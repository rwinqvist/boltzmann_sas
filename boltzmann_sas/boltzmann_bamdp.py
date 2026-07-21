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


class BoltzmannBAMDP():
    """
    Class representing the full SAS model with Boltzmann rational human operators. Represented by a BAMDP?
    """
    def __init__(self, domain, operators:list[Operator]):
        """
        Docstring for __init__
        
        :param domain: domain model describing the environment in which the system operates
        :param operators: list of Operator objects
        :param dynamics_to_build: Description
        """
        logging.debug(f"Initializing the Boltzmann SAS...")

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

        try: 
            self.domain_state_conds = self.domain.state_conds 
        except: 
            self.domain_state_conds = [] 

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
        self.belief = {}

        # initialize belief 
        for opidx, boltzmann_op in enumerate(self.boltzmann_operators):
            belief = boltzmann_op.init_belief.clone() 
            belief.attach_model(boltzmann_op.nom_scoring, boltzmann_op.enabled_actions)
            self.belief[opidx] = belief

        self.T = {}

        logging.info("Boltzmann SAS initialized.")


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

    #def update_belief(self, state, opidx, issued_domain_action, executed_domain_action):
    def update_belief(self, hist_data:Observation):
        operator = hist_data.operator
        if operator in self.boltzmann_operators:
            self.update_operator_belief(operator, hist_data)

        return self.belief
    
    def get_belief(self):
        belief = {}
        for opidx in self.boltzmann_operator_indices.values():
            belief[opidx] = self.belief[opidx].get_belief()
        
        return belief
    
    def get_belief_stats(self):
        param_stats = {}
        for opidx in self.boltzmann_operator_indices.values():
            param_stats[opidx] = self.belief[opidx].get_stats()

        return param_stats
    
    def get_params_est(self):
        beta_hat, alpha_hat = None, None
        params_ests = {}
        for opidx in self.boltzmann_operator_indices.values():
            beta_hat, alpha_hat = self.belief[opidx].get_params_est()
            params_ests[opidx] = OperatorParams(beta=beta_hat, alpha=alpha_hat)

        return params_ests

    def reset_belief(self):
        self.belief = {opidx: boltzmann_op.init_belief.clone() for opidx, boltzmann_op in enumerate(self.boltzmann_operators)}
        
    
    def update_operator_belief(self, operator: BoltzmannHumanOperator, hist_data: Observation):
        opidx = self.boltzmann_operator_indices[operator]
        belief = self.belief[opidx]

        likelihood_fn = lambda params: operator.boltzmann_kernel(hist_data.domain_state, hist_data.op_state, hist_data.executed_domain_action, hist_data.issued_domain_action, params)
        belief.update_belief(likelihood_fn, hist_data)


    def get_successors(self, state, action):
        """
            Get successor states and their likelihoods, based on the RESOLVED action choice. 
        """
        if (state, action) in self.T: 
            return self.T[(state, action)]
    
        successors = {}

        domain_state, joint_op_state = state 

        # unpack action info and find active operator
        active_opidx, domain_action = action 
        active_operator = self.operators[active_opidx]
        active_op_state = joint_op_state[active_opidx]

        # get transition distribution from operator 
        domain_transitions = active_operator.get_domain_transitions(domain_state, active_op_state, domain_action)

        #p_success = active_operator.get_domain_performance_rate(active_op_state, domain_action)
        #next_domain_states, next_domain_probs = self.domain.get_transitions(domain_state, domain_action, p_success)

        # get next operator performance state transitions 
        next_op_states = {}
        for opidx in range(self.num_operators):
            operator = self.operators[opidx]
            op_state = joint_op_state[opidx]
            is_active = operator == active_operator
            next_op_states[opidx] = operator.get_operator_state_transitions(op_state, is_active, domain_action)

        next_joint_op_states = list(itertools.product(*next_op_states.values()))

        # get full state transitions 
        for next_domain_state, p_domain in domain_transitions.items():
            for next_joint_op_state in next_joint_op_states:
                next_state = (next_domain_state, next_joint_op_state)
                p_op = 1 

                for opidx, next_op_state in enumerate(next_joint_op_state):
                    p_op *= next_op_states[opidx][next_op_state]

                p = p_domain * p_op 
                # ensure state is proper
                assert next_state in self.states, f"Next state: {next_state} not in state space"

                successors[next_state] = p 
        

        self.T[(state, action)] = successors
        return successors  
       

    def get_reward(self, state, action, next_state):
        #if (state, action, next_state) in self.R:
            #return self.R[(state, action, next_state)]
        
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

        successors = self.get_successors(state, executed_action)
        states = list(successors.keys())
        likelihoods = list(successors.values())
        next_state = random.choices(states, likelihoods, k=1)[0]
        reward = self.get_reward(state, executed_action, next_state)
        
        return next_state, reward, executed_action 
    
    
    def sample_parameters(self, num_samples:int):
        samples = []

        for _ in range(num_samples):
            sample = {}
            for opidx in range(len(self.boltzmann_operators)):
                sample[opidx] = self.belief[opidx].sample() 
            samples.append(sample)

        return samples 

