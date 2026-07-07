from enum import Enum

class Approach(Enum):
    BAMCP = "bamcp"
    NAIVE_FREQ_WARMSTART = "frequentist_warmstart"
    NAIVE_BAYESIAN_WARMSTART = "bayesian_warmstart"