import random 
import numpy as np 
from copy import deepcopy 
from domains.uav.enums import DomainAction, TerrainType




class UAVDomain(object):
    """
        Class representing the UAV gridworld domain with one UAV and M operators. 

        This domain consists of a 2D (square) grid of size (size, size) where each cell can be either: 
        - Normal terrain (represented by 'X')
        - An obstacle (represented by 'O')
        - A start cell (represented by 'S'), default to upper left corner (0, 0)
        - A goal cell (represented by 'G'), default to lower right corner (size-1, size-1)

        The grid supports four directional movements, which are defined as 
        LEFT = (0, -1)
        DOWN = (1, 0)
        RIGHT = (0, 1)
        UP = (-1, 0)

        Attributes: 
            - size (int): The width and height of the grid (square shape)
            - map (): 
            - map_name (str): name for predefined maps 
            - p_obs (float): obstacle likelihood in the grid 
            - operators (list): a list of available operators (human and autonomous) who take turns operating the UAV 
    """

    MAPS = {}

    DOMAIN_ACTIONS = [DomainAction.LEFT, DomainAction.DOWN, DomainAction.RIGHT, DomainAction.UP]
    DIRECTIONS = [DomainAction.LEFT.value, DomainAction.DOWN.value, DomainAction.RIGHT.value, DomainAction.UP.value]
    NAME_TO_ACTION = {"left": DomainAction.LEFT, "down": DomainAction.DOWN, "right": DomainAction.RIGHT, "up": DomainAction.UP}
    ACTION_TO_NAME = {DomainAction.LEFT: "left", DomainAction.DOWN: "down", DomainAction.RIGHT: "right", DomainAction.UP: "up"}

    TERRAIN = [TerrainType.NOM.value, TerrainType.OBSTACLE.value, TerrainType.START.value, TerrainType.GOAL.value]


    def __init__(self, size=3, p_obs=0.4, grid_map=None, map_name="", p_success=None):
        self.domain_name = "uav"
        self.size = size 
        self.p_obs = p_obs 
        self.col_min, self.col_max = 0, size-1  
        self.row_min, self.row_max = 0, size-1 
        self.col_init, self.row_init = 0, 0 
        self.col_goal, self.row_goal = size-1, size-1 

        self.states = [(row, col) for row in range(size) for col in range(size)]
        self.s0 = (0, 0)
        self.goal = (self.row_goal, self.col_goal)
        self.goal_states = [self.goal]
        self.terminal_states = [self.goal]

        self.actions = UAVDomain.DOMAIN_ACTIONS 
        self.action_names = list(UAVDomain.NAME_TO_ACTION.keys())
        self.name_to_action = UAVDomain.NAME_TO_ACTION
        self.action_to_name = UAVDomain.ACTION_TO_NAME
        self.state_action_map = {state: self.actions for state in self.states}
        self.p_success = p_success

        if grid_map is None and map_name == "":
            grid_map = UAVDomain.generate_random_grid(size, p_obs, self.s0)
        elif grid_map is None: 
            grid_map = UAVDomain.MAPS[map_name]

        self.grid_map = np.array([list(row) for row in grid_map], dtype='<U1')  # unicode string of length 1
        self.map_str = self.grid_map.tolist()
        self.obstacles = [(row, col) for row in range(self.row_min, self.row_max+1) for col in range(self.col_min, self.col_max+1) if grid_map[row][col] == TerrainType.OBSTACLE.value]

        self.collision_cost = 10
        self.nom_cost = 1
        self.goal_reward = 0
        self.state_costs = {state: 0 if state not in self.obstacles else self.collision_cost for state in self.states}
        self.state_action_costs = {(state, action): self.nom_cost for state in self.states for action in self.actions}

        self.T_structure, self.T_det, self.enabled_actions, self.state_rewards = self.build_dynamics()

        print(f"Generated UAV domain:")
        print(f"  Size:               {size}")
        print(f"  Obstacle rate:      {p_obs}")
        print(f"  Total states:       {len(self.states)}")
        print(f"  Terminal states:    {len(self.terminal_states)}")
        if size <= 30:
            print(f"  Map:\n{self}")

    @property
    def id_tag(self):
        """A short string identifying this domain instance for e.g. result paths."""
        return f"size{self.size}_pobs{self.p_obs}"

    @classmethod
    def generate_random_grid(cls, size, p_obs, start=(0, 0)):
        """
            Generate a random square grid map with height = width = size, and with 
            an obstacle rate of p_obs. Default start location is upper left corner (row, col) = (0, 0)
            and default goal location is lower right corner (row, col) = (size-1, size-1)
        """
        valid = False 
        board = [] 

        while not valid: 
            board = np.random.choice([TerrainType.NOM.value, TerrainType.OBSTACLE.value], (size, size), p=[1-p_obs, p_obs])
            board[start[0]][start[1]] = TerrainType.START.value
            board[-1][-1] = TerrainType.GOAL.value
            if p_obs > 0:
                if TerrainType.OBSTACLE.value in board: 
                    valid = cls.is_valid(board, size)
            else:
                valid = cls.is_valid(board, size)
        return ["".join(x) for x in board]


    @classmethod 
    def is_valid(cls, board, size, start=(0,0), non_traversables=None):
        """
            Checks if grid map is valid (if there's a clear path from start to goal).
        """

        frontier, discovered = [], set() 
        frontier.append(start)

        non_traversables = [TerrainType.OBSTACLE.value] if non_traversables is None else non_traversables

        while frontier: 
            r, c = frontier.pop()
            assert board[r][c] in cls.TERRAIN, "Invalid terrain type: {board[r][c]}"

            if (r, c) not in discovered:
                discovered.add((r, c))
                for (dr, dc) in cls.DIRECTIONS:
                    r_new = r + dr
                    c_new = c + dc

                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        # outside of map/board 
                        continue 
                    if board[r_new][c_new] == TerrainType.GOAL.value:
                        # reached goal
                        return True 
                    if board[r_new][c_new] not in non_traversables:
                        # append next cell is cell is traversable (i.e., ice)
                        frontier.append((r_new, c_new))

        return False

    @classmethod
    def from_map(cls, size, p_obs, layout):
        uav_domain = cls(size, p_obs=p_obs, grid_map=layout)
        return uav_domain

    def __str__(self):
        return "\n".join([" ".join(row) for row in self.map_str])

    def __repr__(self):
        return "\n".join([" ".join(row) for row in self.map_str])
    
    def print_domain(self): 
        print("\n".join([" ".join(row) for row in self.map_str]))


    def build_dynamics(self):
        """ Build transitions and rewards """
        T_structure = {}
        T_det = {0 :{}}
        R = {}
        state_rewards = {self.s0: 0}
        enabled_actions = {}

        for state in self.states: 
            state_rewards[state] = -self.state_costs.get(state, 0)
            if state in self.goal_states:
                state_rewards[state] = self.goal_reward

            # Only enable actions that actually change position -- exclude
            # actions that would just bump into a wall/boundary and self-loop.
            # Without this, a human (biased toward, or simply landing on, a
            # wall-facing action at an edge/corner state) can get permanently
            # stuck there regardless of advice, since every issuable action
            # collapses to "stay put".
            valid_actions = [a for a in self.actions if self.move(state, a) != state]
            enabled_actions[state] = valid_actions

            for action in valid_actions:
                if state in self.terminal_states:
                    T_structure[(state, action)] = [state]
                    state_rewards[state] = 0 
                else:
                    next_states = self.get_next_states(state, action)
                    T_structure[(state, action)] = next_states
                    T_det[0][(state, action)] = {
                        s: (1.0 if s == next_states[0] else 0.0)
                        for s in next_states
                    }

        return T_structure, T_det, enabled_actions, state_rewards

    def move(self, state, action): 
        """
        Move the UAV one step in the specified direction, respecting grid boundaries.

        Parameters:
            state (tuple): Current position as (row, col).
            action (tuple): Direction to move, must be one of self.DOMAIN_ACTIONS.

        Returns:
            tuple: New position (row, col) after applying the move (within grid boundaries)
        """
        row, col = state 

        assert state in self.states, f"State {state} not in state space"
        assert action in self.DOMAIN_ACTIONS, f"Invalid action {action}"
        
        if action == DomainAction.LEFT: 
            col = max(col-1, self.col_min) 

        elif action == DomainAction.DOWN: 
            row = min(row+1, self.row_max)
            
        elif action == DomainAction.RIGHT: 
            col = min(col+1, self.col_max) 

        elif action == DomainAction.UP: 
            row = max(row-1, self.row_min)

        return row, col
    
    def get_next_states(self, state, action): 
        """
        Returns possible next states given a state and an action. Note that there are no proababilities here. Those are 
        added later. For that reason, the ordering of the next states should not be changed. The order is such that 
        the first item in the list represents a successful outcome, and the other two a `failed` outcome.
        """
        if action == DomainAction.LEFT: 
            next_states = [self.move(state, DomainAction.LEFT), self.move(state, DomainAction.UP), self.move(state, DomainAction.DOWN)]

        elif action == DomainAction.DOWN: 
            next_states = [self.move(state, DomainAction.DOWN), self.move(state, DomainAction.RIGHT), self.move(state, DomainAction.LEFT)]
            
        elif action == DomainAction.RIGHT: 
            next_states = [self.move(state, DomainAction.RIGHT), self.move(state, DomainAction.UP), self.move(state, DomainAction.DOWN)]

        elif action == DomainAction.UP: 
            next_states = [self.move(state, DomainAction.UP), self.move(state, DomainAction.RIGHT), self.move(state, DomainAction.LEFT)]

        return next_states


    def get_next_state(self, state, action): 
        next_states = self.T_det[0][(state, action)]
        for next_state, p in next_states.items():
            if p != 0:
                return next_state
    


    # def get_reward(self, state, action, next_state): 
    #     return self.state_rewards[next_state] - self.state_action_costs[(state, action)]
    
    def get_reward(self, state, action, next_state):
        shaping = self._potential(next_state) - self._potential(state)
        return self.state_rewards[next_state] - self.state_action_costs[(state, action)] + shaping


    def _potential(self, state, shaping_scale=5.0):
        if state in self.terminal_states:
            return 0.0
        row, col = state
        dist = abs(row - self.row_goal) + abs(col - self.col_goal)
        return -dist * shaping_scale
            