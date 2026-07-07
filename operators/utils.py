from enum import Enum
from dataclasses import dataclass
from typing import Optional


class OperatorType(Enum):
    HUMAN = "human"
    BHUMAN = "bhuman"
    AUTO = "auto"

@dataclass(frozen=True)
class OperatorParams:
    beta: Optional[float] = None 
    alpha: Optional[float] = None

class TransitionType(Enum):
    T = "T"
    TAU = "tau"
    REC = "rec"

    def __str__(self):
        return self.value
    

class PerformanceDecayType(Enum):
    LIN = "lin"
    EXP = "exponential"
    LOG = "logarithmic"



