from typing import Optional
from dataclasses import dataclass
from operators.utils import OperatorParams, OperatorType
from belief.belief import Belief

@dataclass(frozen=True)
class OperatorContext:
    category: OperatorType
    n: int
    actions: list
    cost_nominals: dict 
    enabled_actions: Optional[dict] = None
    domain_transitions: Optional[dict] = None
    params: Optional[OperatorParams] = None
    transition_nominals: Optional[dict] = None 
    nom_scoring: Optional[dict] = None
    init_belief: Optional[Belief] = None
    