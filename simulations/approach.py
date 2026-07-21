from enum import Enum

class Approach(Enum):
    BAMCP = "bamcp"
    BAMCP_ES = "early_stopping_bamcp"
    NAIVE_FREQ_WARMSTART = "frequentist_warmstart"
    NAIVE_BAYESIAN_WARMSTART = "bayesian_warmstart"
    BAMCP_ALPHA_COLLAPSE = "bamcp_alpha_collapse"
    VI = "VI"