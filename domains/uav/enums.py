from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class DomainContext:
    size: int
    p_obs: float 
    
    def __str__(self):
        return "domain_context"

class DomainAction(Enum):
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP = (-1, 0)
    DOWN = (1, 0)


class TerrainType(Enum):
    NOM = "X"
    OBSTACLE = "O"
    START = "S"
    GOAL = "G"
    