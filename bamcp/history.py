
class History():
    """
        Class for a history of a path. 
        Encoded in a flat alternating tuple structure (s0, ¨(a1, a1'), s1, (a2, a2'), s2...)
        Where (a1, a1') is an issued-executed action pair 
    """
    def __init__(self, items=()):
        self.items = items 

    def __eq__(self, other):
        return self.items == other.items 
    
    def __hash__(self):
        return hash(self.items)
    
    def __repr__(self):
        return str(self.items)
    
    def __str__(self):
        return str(self.items)
    
    def __len__(self):
        return len(self.items)
    
    def add_entry(self, issued_action, executed_action, state):
        action_pair = (issued_action, executed_action)
        return History(self.items + (action_pair, state))
    
    def get_history(self):
        return self.items 
    
    @property
    def last_state(self):
        return self.items[-1]
    
    @property
    def last_action_pair(self):
        return self.items[-2] if len(self.items) > 1 else None
    
    @property
    def last_issued_action(self):
        return self.last_action_pair[0]
    
    @property
    def last_executed_action(self):
        return self.last_action_pair[1]
    
    @property
    def parent_history(self):
        return self.items[:-2]
