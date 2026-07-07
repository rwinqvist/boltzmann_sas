from dataclasses import dataclass

@dataclass(frozen=True)
class Observation: 
    domain_state: any 
    op_state: any
    operator: any
    executed_domain_action: any 
    issued_domain_action: any
