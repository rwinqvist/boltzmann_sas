import numpy as np
import datetime as dt
import math
import random
from typing import Optional
from boltzmann_sas.boltzmann_sas import BoltzmannSAS
from boltzmann_sas.globals import DEFER
from operators.utils import OperatorParams
from bamcp.history import History
from bamcp.objective import Objective
from bamcp.rollout_policies import random_rollout
from belief.observation import Observation

MIN_BIAS = 1

class Node():
    """
        A node in the BAMCP search tree. 
        Class for storing statistics about a node in the BAMCP tree.
    """
    def __init__(self, history:History, enabled_actions, max_objective, tree_level=0, parent=None, parent_action=None):
        self.history = history
        self.enabled_actions = enabled_actions
        #print(history)
        #print(enabled_actions)
        #exit()
        self.tree_level = tree_level
        self.parent = parent
        self.parent_action = parent_action
        self.children = set([])
        self.Q_values = {a: 0.0 for a in enabled_actions}

        # initialize values
        self.value = 0.0
        self.num_visits = 0 
        self.action_visits = {a: 0 for a in enabled_actions}
        
        # for evaluating and computing Q values 
        if max_objective:
            self.minmax = max 
        else:
            self.minmax = min
    

    @property
    def is_fully_explored(self):
        """ Returns True if all actions have been selected/visited at least once, False otherwise. """
        if 0 in self.action_visits.values():
            return False 
        return True 
    

    def get_value(self):
        """ Returns the value of the tree node """
        Q_values = list(self.Q_values.values())
        if len(Q_values) == 0:
            return 0.0 
        self.value = self.minmax(Q_values)
        return self.value 
    
    
    def update_Q(self, action, rollout_return):
        """
            Updates Q value of state.
        """
        n = self.action_visits[action]
        self.Q_values[action] += (rollout_return - self.Q_values[action])/(n + 1)


    def register_visit(self, action, rollout_return):
        """
            Increments visit counts based on action and updates Q values.

            Args:
                * action: the executed action
                * rollout_return: the return achieved from the tree rollout
        """
        self.update_Q(action, rollout_return)
        self.action_visits[action] += 1 
        self.num_visits += 1


    def add_child(self, child_pair):
        self.children.add(child_pair)

     

class BAMCP():
    """
        BAMCP based on UCT.
    """
    def __init__(self, bamdp:BoltzmannSAS, objective:Objective, max_entropy_rollout=False, uct_weight=None, max_depth=100, history=History(), debug=False):
        self.bamdp = bamdp 
        self.objective = objective 
        self.max_entropy_rollout = max_entropy_rollout
        self.uct_weight = (lambda node: uct_weight if uct_weight is not None else max(abs(node.get_value()), MIN_BIAS))
        self.max_depth = max_depth
        self.debug = debug
        self.executed_history = []
        self.nodes = {}
        self.solved = {}
        self.policy = None
        self.history = history

        if self.objective.max_objective:
            self.minmax = max 
            self.argminmax = np.argmax
            self.minmaxsign = 1 
        else:
            self.minmax = min 
            self.argminmax = np.argmin
            self.minmaxsign = -1

    
    def init_node(self, history:History, tree_level, max_objective, parent=None, parent_action=None):
        """ Initialize new node in the search tree if it doesn't already exist. """
        node_key = (history, tree_level)

        if node_key not in self.nodes:
            self.solved[node_key] = False 
            self.nodes[node_key] = {}

        parent_pair = (parent, parent_action)
        if parent_pair not in self.nodes[node_key]:
            enabled_actions = self.bamdp.enabled_actions[history.last_state] 
            node = Node(history, enabled_actions, max_objective=max_objective, tree_level=tree_level, parent=parent, parent_action=parent_action)
            self.nodes[node_key][parent_pair] = node


    def update_node(self, history:History, tree_level:int, parent, parent_action, action, rollout_return):
        """ Update node after trial. """
        node_key = (history, tree_level)
        parent_pair = (parent, parent_action)
        node = self.nodes[node_key][parent_pair]
        node.register_visit(action, rollout_return)



    def search(self, t=0, num_trials=0, history=None, initial_tree_level=0, parent_history:History=None, parent_action=None):
        """
            The planning phase. Runs BAMCP for time t or for num_trials. It builds/updates the tree, updates 
            the Q-values and visit counts. Builds a policy. 
        """

        if t == 0 and num_trials == 0:
            raise ValueError("A time limit or a number of trials must be provided to run BAMCP.")

        # initialize root node if needed 
        if history is None:
            initial_state = self.bamdp.s0
            history = History(items=(initial_state,))
            initial_tree_level = 0
            self.init_node(history=history, tree_level=initial_tree_level, max_objective=self.objective.max_objective)
            parent_node = None
            parent_action = None
        
        else:
            # Ensure passed history and tree level is in tree
            node_key = (history, initial_tree_level)
            if node_key not in self.nodes:
                #raise ValueError("Invalid history/tree level given.")        
                # This history has never been seen by this tree before (e.g. a fresh
                # starting state handed in after an out-of-tree warm-start phase).
                # Treat it as a new root, same as the history is None case above.
                self.init_node(history=history, tree_level=initial_tree_level, max_objective=self.objective.max_objective)
                parent_node = None
                parent_action = None
                
            else:
                # node key already exists in tree - find the matching parent entry

                # Find parent object (canonicalizing the parent reference)
                # I don't think this is strictly necessary as the correct objects should always be passed as arguments, but I will keep it here anyway...
                for (parent, action) in self.nodes[node_key]:
                    if parent.history == parent_history and action == parent_action:
                        (parent_node, parent_action) = (parent, action)  # turn the parent reference into a proper object pointer here. 
                        # break is probably unneccessary, but will save me a few loops possibly
                        break
    
        if num_trials != 0:
            self.search_num_trials(history, initial_tree_level, parent_node, parent_action, num_trials)
        else:
            self.search_for_time(history, initial_tree_level, parent_node, parent_action, t)

        self.clear_policy()
        return self.get_policy()
    

    def search_num_trials(self, history:History, tree_level:int, parent_node:Node, parent_action, num_trials:int):
        """ Run search for num_trials from initial_node. """

        parameters_list = self.bamdp.sample_parameters(num_samples=num_trials)
        for parameters in parameters_list: 
            self.trial(history, tree_level, parent_node, parent_action, parameters)

    
    def search_for_time(self, history:History, tree_level:int, parent_node:Node, parent_action, t):
        """ Run search for time t [ms] from initial_node. """
        start = dt.datetime.now() 
        max_duration = dt.timedelta(milliseconds=t)
        time_exceeded = False 

        while not time_exceeded:
            parameters = self.bamdp.sample_parameters(num_samples=1)[0]
            self.trial(history, tree_level, parent_node, parent_action, parameters)
            time_exceeded = dt.datetime.now() - start > max_duration

    
    def trial(self, initial_history:History, initial_tree_level=0, parent_node:Node=None, parent_action=None, op_parametrizations:Optional[dict[int, OperatorParams]]=None):
        """ 
        Runs a single trial of MCTS starting from node.  
                1: Start at root node 
                2. Traverse tree (selection) 
                3. Possibly expand ONE new node 
                4. Do rollout (based on rollout policy)
                5. Backpropagate 
        """
        trial_path = [] 
        max_depth = self.max_depth
        current_history = initial_history
        current_level = initial_tree_level
        current_state = current_history.last_state

        while max_depth >= 0:
            # check current state is not goal state 
            if self.bamdp.is_goal(current_state) or self.bamdp.is_terminal(current_state):
                break 

            # fetch current node   
            current_node_key = (current_history, current_level)     # current node key
            parent_pair = (parent_node, parent_action)              # parents of current history
            current_node = self.nodes[current_node_key][parent_pair]

            issued_action = self.select_action(current_node)    # select traversal action from current node
            next_state, reward, executed_action = self.bamdp.step(current_state, issued_action, op_parametrizations)     # step in BAMDP to continue traversal
            trial_path.append((current_history, current_level, parent_node, parent_action, issued_action, reward))      # append to trial path
            
            # keep track of parent 
            parent_action = issued_action          # the new "parent action" is the action we just issued 
            parent_node = current_node      # new parent node is the current node
            current_state = next_state      # new BAMDP state 
            current_history = current_history.add_entry(issued_action, executed_action, next_state)     # update history with (action, next_state)
            current_level = initial_tree_level + len(trial_path)                # update tree level

            # Has this successor state been explored/expanded?
            child_pair = (current_state, issued_action)
            if child_pair not in current_node.children:
                # TODO: check based on action too   : I'm not sure this is necessary. The init_node implicitly checks this (I think). The code has been used this way so I will assume it is correct.
                break 

            max_depth -= 1

        # Initialize new node
        if not (self.bamdp.is_goal(current_history.last_state) or self.bamdp.is_terminal(current_history.last_state)):
            self.init_node(current_history, current_level, max_objective=self.objective.max_objective, parent=parent_node, parent_action=parent_action)

        # Add child to parent 
        if parent_node is not None:
            parent_node.add_child(child_pair)

        # Simulate step (rollout from child)
        # rollout parameters: if maximum entropy, assume beta=0 for all human operators, else rely on belief
        if self.max_entropy_rollout:
            rollout_parameters = {}
            for opidx, op_parametrization in op_parametrizations.items():
                rollout_parameters[opidx] = OperatorParams(beta=0, alpha=op_parametrization.alpha)
        else: 
            rollout_parameters = op_parametrizations
        rollout_reward = self.rollout(current_history, rollout_parameters, max_depth)

        # Backpropagation 
        self.backpropagate(trial_path, rollout_reward)



    def rollout(self, current_history:History, rollout_parameters, max_depth:int):
        """ Perform rollout from current_history to terminal state or until max_depth is reached. """
        total_reward = 0 

        current_state = current_history.last_state
        at_goal = self.bamdp.is_goal(current_state)
        at_terminal = self.bamdp.is_terminal(current_state)
        
        while not at_goal and not at_terminal and max_depth >= 0:
            rollout_action = self.objective.rollout(state=current_history, model=self.bamdp)
            current_state, reward, executed_action = self.bamdp.step(current_state, rollout_action, rollout_parameters)
            total_reward += reward 
            current_history = current_history.add_entry(rollout_action, executed_action, current_state)

            at_goal = self.bamdp.is_goal(current_state)
            at_terminal = self.bamdp.is_terminal(current_state)

            max_depth -= 1 
        
        return total_reward


    def backpropagate(self, trial_path, rollout_reward):
        """ Backpropagate the information gained from the trial. """
        
        current_return = rollout_reward
        # go backwards in the trial path
        for i in range(len(trial_path)-1, -1, -1):
            (history, tree_level, parent_node, parent_action, action, reward) = trial_path[i]
            current_return += reward 
            self.update_node(history, tree_level, parent_node, parent_action, action, current_return)

        
    def select_action(self, node:Node):
        """ Selects action for tree traversal from node. """
        if node.is_fully_explored:
            # uct action 
            action = self.uct_action(node)
            return action
        
        else:
            # random exploration 
            actions = node.enabled_actions
            # PRINT
            #print(node.history)
            #print(actions)
            #input("Continue...")
            for action in random.sample(actions, len(actions)):
                if node.action_visits[action] == 0:
                    return action
                
        raise ValueError("No action tree traversal action found.")
    

    def uct_action(self, node:Node):
        """ Selects UCT action """
    
        num_state_visits = node.num_visits
        actions = node.enabled_actions
        scores = []

        for action in actions: 
            exploitation_term = node.Q_values[action]
            exploration_term = np.sqrt(np.log(num_state_visits) / node.action_visits[action])
            score = exploitation_term + self.minmaxsign * self.uct_weight(node) * exploration_term
            scores.append(score)

        return actions[self.argminmax(scores)]
    

    def clear_policy(self):
        self.policy = None
    

    def get_policy(self):
        """ Returns a policy from histories to the policy actions. """

        if self.policy is not None: 
            return self.policy 
        
        history_action_mapping = {}
        tree_levels = {}
        best_visits = {}

        for (history, level) in self.nodes: 
            for parent_pair in self.nodes[(history, level)]:
                node = self.nodes[(history, level)][parent_pair]
                if node.is_fully_explored:
                    # Get the action from the node closest to the root
                    if history not in tree_levels or level <= tree_levels[history]:
                        enabled_actions = node.enabled_actions 
                        # More common to get policy action based on num visits 
                        action_visits = [node.action_visits[action] for action in enabled_actions]
                        best_num_visits = max(action_visits)
                        if history in tree_levels and tree_levels[history] == level and best_num_visits <= best_visits[history]:
                            continue 
                        
                        policy_action = enabled_actions[np.argmax(action_visits)]
                        history_action_mapping[history] = policy_action
                        tree_levels[history] = level 
                        best_visits[history] = max(action_visits)

        self.policy = history_action_mapping
        return self.policy 



class BAMCPSolver():
    """ A BAMCP solver. """
    def __init__(self, bamdp:BoltzmannSAS, max_depth, t=2000, num_trials=0, objective:Objective=None, max_entropy_rollout=False):
        self.bamdp = bamdp 
        self.max_depth = max_depth 
        self.t = t 
        self.num_trials = num_trials 
        self.objective = objective
        self.max_entropy_rollout = max_entropy_rollout

        self.bamcp = self.generate_bamcp()
        self.bamcp_policy = self.get_next_action 

    def generate_bamcp(self):
        objective = self.objective if self.objective is not None else Objective(rollout=random_rollout, max_objective=True)
        
        return BAMCP(self.bamdp, objective, self.max_entropy_rollout, max_depth=self.max_depth)
    
    def get_next_action(self, history:History):
        """ Get next action for history from policy. """
        if len(history) == 1:
            policy = self.bamcp.search(t=self.t, num_trials=self.num_trials, history=history)
                
        else:
            # re-using the old tree
            self.bamcp.clear_policy() 
            self.bamcp.history = history 
            tree_level = int(len(history)/2)    # becuase history is [s0, (a1, a1'), s2, (a2, a2'), ...], so half of the list minus root (s0) is the tree level 
            parent_history = History(history.parent_history)
            parent_action = history.last_issued_action
            policy = self.bamcp.search(t=self.t, num_trials=self.num_trials, history=history, initial_tree_level=tree_level, parent_history=parent_history, parent_action=parent_action)

        return policy[history]
    
    
    def update_belief(self, hist_data:Observation):
        """ Update BAMDP belief. """
        self.bamdp.update_belief(hist_data)


    def unpack_action(self, action): 
        is_defer, is_advice, is_auto = 0, 0, 0

        opidx, advice = action 
        if not opidx in self.bamdp.boltzmann_operator_indices.values():
            is_auto = 1         
        else: 
            if advice == DEFER:
                is_defer = 1 
            else: 
                is_advice = 1

        return is_advice, is_defer, is_auto
    

    def get_belief(self):
        return self.bamdp.get_belief()
    
    def get_belief_stats(self):
        return self.bamdp.get_belief_stats()


    def run(self, s0=None, total_reward=0, total_steps=0, is_defer_vec=None, is_advice_vec=None, is_auto_vec=None, belief_vec=None, belief_stats=None, rewards=None, cum_rewards=None, max_steps=np.inf):
        state = s0 if s0 is not None else self.bamdp.s0 
        history = History(items=(state,))
        total_reward = total_reward 
        total_steps = total_steps
        is_defer_vec   = is_defer_vec   if is_defer_vec   is not None else []
        is_advice_vec  = is_advice_vec  if is_advice_vec  is not None else []
        is_auto_vec    = is_auto_vec    if is_auto_vec    is not None else []
        belief_vec     = belief_vec     if belief_vec     is not None else []
        belief_stats   = belief_stats   if belief_stats   is not None else []
        rewards        = rewards        if rewards        is not None else []
        cum_rewards    = cum_rewards    if cum_rewards    is not None else []
        followed_advice = [] 

        # initial belief stats
        if len(belief_stats) == 0:
            belief_stats.append(self.get_belief_stats())


        #print("Belief stats: ", belief_stats)
        #input("Start...")

        while not self.bamdp.is_goal(state) and not self.bamdp.is_terminal(state) and total_steps < max_steps:
            # plan 
            #print("current state: ", history.last_state)
            
            issued_action = self.get_next_action(history)
            is_advice, is_defer, is_auto = self.unpack_action(issued_action)
            is_advice_vec.append(is_advice)
            is_defer_vec.append(is_defer)
            is_auto_vec.append(is_auto)

            #print("next issued action: ", issued_action)

            # execute in real environment
            next_state, reward, executed_action = self.bamdp.step(state, issued_action)
            #print("executed action: ", executed_action)
            #print("reward: ", reward)

            if issued_action[1] != DEFER and executed_action[1] == issued_action[1]:
                followed_advice.append(1)
            else:
                followed_advice.append(0)

            rewards.append(reward)

            # update belief based on what we observed 
            opidx, executed_domain_action = executed_action
            operator = self.bamdp.operators[opidx]
            _, issued_domain_action = issued_action

            domain_state, joint_op_state = state 
            op_state = joint_op_state[opidx]

            hist_data = Observation(domain_state=domain_state, op_state=op_state, operator=operator, executed_domain_action=executed_domain_action, issued_domain_action=issued_domain_action)
            
            true_params = self.bamdp.operators[0].params
            enabled_actions = self.bamdp.domain.enabled_actions[state[0]]
            Phi_nom = self.bamdp.operators[0].nom_scoring

            #print("State: ", state)
            #print("Issued action: ", issued_action)
            #print("Is advice: ", is_advice)
            #print("Is defer: ", is_defer)
            #print("Is auto: ", is_auto)
            #print("Executed action: ", executed_action)
            #print("Reward: ", reward)
            #print(f"Phi values: {[(a, Phi_nom[0][(domain_state, a)]) for a in enabled_actions]}")
            #print(f"Beta: {true_params.beta}, Alpha: {true_params.alpha}")
            #scores = {a: true_params.beta * Phi_nom[0][(domain_state, a)] for a in enabled_actions}
            # if issued_domain_action != DEFER:
            #     scores[issued_domain_action] += true_params.alpha
            # max_s = max(scores.values())
            # exp_s = {a: math.exp(scores[a] - max_s) for a in scores}
            # Z = sum(exp_s.values())
            # probs = {a: exp_s[a]/Z for a in exp_s}
            # print(f"Boltzmann probs: {probs}")
            # print(f"Followed advice: {issued_domain_action == executed_domain_action and issued_domain_action != DEFER}")
            # input("Next...\n")

            #print("Updating belief...")
            self.update_belief(hist_data)
            belief_vec.append(self.get_belief())
            belief_stats.append(self.get_belief_stats())
            #print("New belief stats: ", self.get_belief_stats())
            #input("Next...\n")

            history = history.add_entry(issued_action, executed_action, next_state)
            state = next_state 
            total_reward += reward 
            total_steps += 1 
            cum_rewards.append(total_reward)

        results = {
            "cum_rewards": cum_rewards,
            "rewards": rewards,
            "total_reward": total_reward,
            "total_steps": total_steps, 
            "is_defer": is_defer_vec, 
            "is_advice": is_advice_vec, 
            "is_auto": is_auto_vec, 
            "belief": belief_vec,
            "belief_stats": belief_stats,
            "history": history,
            "followed_advice": followed_advice
        }

        return results

            

        
