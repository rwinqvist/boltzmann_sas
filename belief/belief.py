import numpy as np
from abc import ABC, abstractmethod
import random
import math
from itertools import product
from scipy.optimize import minimize_scalar
from scipy.stats import norm 
from boltzmann_sas.globals import DEFER
from operators.utils import OperatorParams
from belief.observation import Observation
from common.math_utils import normalize_distribution


# ============================================================
# Base Belief class
# ============================================================
class Belief(ABC):
    """ 
    Abstract base class for all belief representations over operator parameters (beta, alpha).
    Subclasses must implement sample(), update_belief(), and clone()
    """

    @abstractmethod
    def sample(self) -> OperatorParams:
        """ Sample a (beta, alpha) parameter pair from the belief. """

    @abstractmethod
    def update_belief(self, likelihood_fn, hist_data:Observation):
        """ Update belief given a new observation. """

    @abstractmethod
    def clone(self): 
        """ Return a deep copy of the belief. """

    @abstractmethod
    def get_stats(self): 
        """ Return mean and std of params under current belief. """

    @abstractmethod 
    def get_beta_credible_bound(self, confidence=0.1):
        """
        Return a PESSIMISTIC (lower-tail) credible bound on beta at the given
        confidence level -- e.g. confidence=0.1 returns roughly the 10th
        percentile of the current belief's marginal distribution over beta.
 
        Used to conservatively evaluate whether alpha is still identifiable
        (via Fisher information, which increases with beta up to a point):
        evaluating at a pessimistic-low beta means "collapse alpha" decisions
        only trigger when safe even in the worst plausible case, guarding
        against premature collapse from a temporarily-low beta estimate.
 
        :param confidence: tail probability, e.g. 0.1 = 10th percentile lower bound
        """

    @abstractmethod
    def freeze_alpha(self, alpha_value):
        """
        Collapse alpha to a single fixed value, preserving beta's current
        marginal distribution. After this call, subsequent update_belief()
        calls should naturally continue refining beta only (alpha has no
        remaining uncertainty to update).
        """



    def sample_discrete(self, distribution:dict):
        values = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(values, weights, k=1)[0]
    
    def normalize(self, distribution:dict):
        return normalize_distribution(distribution)
    
    def attach_model(self, scoring_function, enabled_actions):
        pass


# ============================================================
# 1. Joint Grid Belief
#    - Maintains a joint distribution over (beta, alpha) pairs
#    - Updates both parameters on every observation regardless of DEFER/ADVICE
#    - Baseline: does not exploit the identifiability asymmetry
# ============================================================
class JointGridBelief(Belief):
    """
    Joint belief over (beta, alpha) on a discrete grid. 
    Always updates both parameters regardless of whether action was DEFER or ADVICE. 
    Used as baseline - does not exploit identifiability asymmetry. 
    """

    def __init__(self, p_params:dict):
        self.p_params = p_params 

    @classmethod 
    def uniform(cls, beta_values, alpha_values): 
        """Initialize with a uniform distribution over all (beta, alpha) pairs."""
        n = len(beta_values) * len(alpha_values)
        p_params = {(b, a): 1/n for b, a in product(beta_values, alpha_values)}
        return cls(p_params)


    def clone(self): 
        return JointGridBelief(self.p_params.copy())
    
    def sample(self) -> OperatorParams:
        beta, alpha = self.sample_discrete(self.p_params)
        return OperatorParams(beta=beta, alpha=alpha)
    
    def update_belief(self, likelihood_fn, hist_data:Observation):
        """ Update joint belief over (beta, alpha) """
        #print("Updating JointGridBelief")
        new_p = {}
        for (beta, alpha), prior in self.p_params.items():
            params = OperatorParams(beta=beta, alpha=alpha)
            new_p[(beta, alpha)] = prior * likelihood_fn(params)

        #print("Old belief: ", self.p_params)
        #print("New belief: ", new_p)
        self.p_params = self.normalize(new_p)


    def get_belief(self):
        return self.p_params


    def get_stats(self): 
        betas  = np.array([b for (b, a) in self.p_params.keys()])
        alphas = np.array([a for (b, a) in self.p_params.keys()])
        probs  = np.array(list(self.p_params.values()))

        beta_mean  = np.sum(probs * betas)
        alpha_mean = np.sum(probs * alphas)
        beta_std   = np.sqrt(np.sum(probs * betas**2)  - beta_mean**2)
        alpha_std  = np.sqrt(np.sum(probs * alphas**2) - alpha_mean**2)

        stats = {
            "beta_mean": beta_mean, 
            "beta_std": beta_std, 
            "alpha_mean": alpha_mean, 
            "alpha_std": alpha_std
        }

        return stats
    
    def get_beta_marginal(self):
        """ Sum out alpha to get the marginal distribution over beta: {beta: prob} """
        marginal = {}
        for (beta, alpha), p in self.p_params.items():
            marginal[beta] = marginal.get(beta, 0.0) + p 
        return marginal 
    
    def get_alpha_marginal(self):
        """ Sum out beta to get the marginal distribution over alpha: {alpha: prob}. """
        marginal = {}
        for (beta, alpha), p in self.p_params.items():
            marginal[alpha] = marginal.get(alpha, 0.0) + p
        return marginal
    
    def get_beta_credible_bound(self, confidence=0.1):
        """
        Discrete VaR-style lower-tail bound: the smallest beta value such that
        the cumulative probability mass at or below it is >= confidence.
        Same logic as compute_discrete_VaR applied to the beta marginal specifically.
        """
        marginal = self.get_beta_marginal()
        cum_prob = 0.0
        prev_beta = next(iter(marginal)) # fallback: smallest grid value 
        for beta, p in sorted(marginal.items()):
            cum_prob += p 
            if cum_prob >= confidence:
                return beta 
            prev_beta = beta 
        return prev_beta 
    

    def freeze_alpha(self, alpha_value):
        """
        Collapse onto a single alpha value, keeping beta's current marginal
        distribution intact. After this call, p_params only has one alpha
        value in its support, so update_belief() naturally continues to
        refine beta only -- no changes needed there.
        """
        beta_marginal = self.get_beta_marginal()
        self.p_params = {(b, alpha_value): p for b, p in beta_marginal.items()}
    



    def __str__(self):
        items = sorted(self.p_params.items(), key=lambda x: -x[1])
        top_n = 10
        lines = [f"Top {top_n} beliefs:"]
        
        for (beta, alpha), prob in items[:top_n]:
            lines.append(f"(beta={beta}, alpha={alpha}): {prob:.4f}")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return self.__str__()




# ============================================================
# 2. Warm-Start Frequentist Belief
#    - Beta is estimated via MLE from DEFER observations (no prior needed)
#    - Alpha is maintained as a Bayesian belief, updated from ADVICE only
#    - Requires a warm-start phase to accumulate DEFER observations before
#      beta_hat is reliable enough to use for planning
# ============================================================

class FreqBelief(Belief):
    """
    Warm-start belief with frequentist beta estimation and Bayesian alpha belief.
        DEFER  -> accumulate observation, recompute MLE beta estimate
        ADVICE -> update p(alpha) using current beta_hat (point estimate)
 
    Requires a warm-start phase where only DEFER actions are executed,
    so that beta_hat is concentrated before ADVICE observations are used.
    """
 
    def __init__(self, p_alpha:dict, defer_observations=None, scoring_fn:dict=None, enabled_actions:dict=None, beta_hat=None):
        """
        Args:
            beta_hat:   initial MLE estimate of beta (can be a default value pre-warmstart)
            p_alpha:    dict mapping alpha values to probabilities
            scoring_fn: callable(domain_state, op_state, action) -> float
                        needed to recompute MLE from accumulated observations
        """
        self.beta_hat = beta_hat
        self.beta_std = None
        self.p_alpha = p_alpha
        self.scoring_fn = scoring_fn
        self.enabled_actions = enabled_actions
        self.defer_observations = defer_observations if defer_observations is not None else [] # accumulate (domain_state, op_state, executed_action) tuples

    @classmethod
    def uniform(cls, alpha_values, defer_observations=None):
        """
        Initialize with a default beta estimate and uniform alpha prior.
 
        Args:
            beta_init:    initial beta value to use before warm-start observations arrive
            alpha_values: discrete alpha grid
            scoring_fn:   scoring function for MLE computation
        """
        p_alpha = {a: 1/len(alpha_values) for a in alpha_values}
        return cls(p_alpha=p_alpha, defer_observations=defer_observations)
    
    def attach_model(self, scoring_function, enabled_actions):
        self.scoring_fn = scoring_function
        self.enabled_actions = enabled_actions
    
    def clone(self):
        cloned = FreqBelief(
            beta_hat   = self.beta_hat,
            p_alpha    = self.p_alpha.copy(),
            scoring_fn = self.scoring_fn,
            enabled_actions = self.enabled_actions.copy() if self.enabled_actions is not None else None
        )
        cloned.defer_observations = self.defer_observations.copy()
        return cloned
    

    def sample(self) -> OperatorParams:
        """Sample using MLE beta point estimate and Bayesian alpha belief."""
        if self.beta_hat is None:
            raise ValueError("beta_hat is None: no DEFER observations have been collected yet.")
        alpha = self.sample_discrete(self.p_alpha)
        return OperatorParams(beta=self.beta_hat, alpha=alpha)
    

    def update_belief(self, likelihood_fn, hist_data: Observation):
        if hist_data.issued_domain_action == DEFER:
            # accumulate DEFER observation and recompute MLE
            self.defer_observations.append(hist_data)
            if self.scoring_fn is not None:
                self.beta_hat = self.compute_mle()
                self.beta_std = self.compute_mle_variance()
            # alpha belief unchanged — DEFER tells us nothing about alpha
 
        else:
            # ADVICE: update alpha only using current beta_hat point estimate
            new_p_alpha = {}
            for alpha, prior in self.p_alpha.items():
                params = OperatorParams(beta=self.beta_hat, alpha=alpha)
                new_p_alpha[alpha] = prior * likelihood_fn(params)
            self.p_alpha = self.normalize(new_p_alpha)

    def compute_mle(self) -> float:
        """
        Compute MLE estimate of beta from accumulated DEFER observations.
        Solves: beta_hat = argmax_beta sum_t log p(a_t | s_t, defer, beta)
        This is a 1D convex optimization.
        """
        def neg_log_likelihood(beta):
            if not self.defer_observations:
                raise ValueError("No defer observations available. Cannot compute MLE estimate.")
            
            total = 0.0
            for obs in self.defer_observations:
                domain_state      = obs.domain_state
                op_state          = obs.op_state
                executed_action   = obs.executed_domain_action
 
                # get scores for all actions at this state
                enabled_actions = self.enabled_actions[domain_state]
                scores = [beta * self.scoring_fn[op_state][(domain_state, a)]
                          for a in enabled_actions]
 
                # log-sum-exp trick
                max_score = max(scores)
                log_Z = math.log(sum(math.exp(s - max_score) for s in scores)) + max_score
 
                # score of executed action
                executed_score = beta * self.scoring_fn[op_state][(domain_state, executed_action)]
                total += executed_score - log_Z
 
            return -total  # minimize negative log likelihood
 

        result = minimize_scalar(neg_log_likelihood, bounds=(0.0001, 1), method='bounded')
        return result.x
    
    def compute_mle_variance(self) -> float:
        """
        Estimate variance of MLE beta estimate via inverse Fisher information.
        Lower variance = more confident estimate.
        """
        if not self.defer_observations:
            return np.inf  # no observations = infinite uncertainty

        total_fisher = 0.0
        for obs in self.defer_observations:
            domain_state    = obs.domain_state
            op_state        = obs.op_state
            enabled_actions = self.enabled_actions[domain_state]   # fixed: was self.enabled_actions[(domain_state, op_state)]

            # compute Boltzmann probabilities under current beta_hat
            scores = np.array([self.beta_hat * self.scoring_fn[op_state][(domain_state, a)]
                            for a in enabled_actions])
            scores -= scores.max()  # numerical stability
            probs   = np.exp(scores)
            probs  /= probs.sum()

            phi_vals  = np.array([self.scoring_fn[op_state][(domain_state, a)]
                                for a in enabled_actions])
            E_phi     = np.dot(probs, phi_vals)
            var_phi   = np.dot(probs, (phi_vals - E_phi)**2)

            total_fisher += var_phi

        return 1.0 / total_fisher if total_fisher > 0 else np.inf

    def get_stats(self): 
    # alpha is Bayesian
        alphas     = np.array(list(self.p_alpha.keys()))
        probs      = np.array(list(self.p_alpha.values()))
        alpha_mean = np.sum(probs * alphas)
        alpha_std  = np.sqrt(np.sum(probs * alphas**2) - alpha_mean**2)

        stats = {
            "beta_mean": self.beta_hat, 
            "beta_std": self.beta_std, 
            "alpha_mean": alpha_mean, 
            "alpha_std": alpha_std
        }

        return stats
    

    def get_alpha_marginal(self):
        """ Alpha is already tracked marginally (not jointly with beta) in this belief. """
        return self.p_alpha
    
    def freeze_alpha(self, alpha_value):
        """ Collapse alpha's belief to a point mass at alpha_value. """
        self.p_alpha = {alpha_value: 1.0}

        
    def get_beta_credible_bound(self, confidence=0.1):
        """
        Gaussian-approximation lower-tail bound, using the MLE beta_hat and its
        asymptotic std (from inverse Fisher information, see compute_mle_variance).
        confidence=0.1 -> beta_hat + z_{0.1} * beta_std, where z_{0.1} < 0, i.e.
        the 10th percentile of a Normal(beta_hat, beta_std^2).
        """
        if self.beta_hat is None or self.beta_std is None:
            raise ValueError("beta_hat/beta_std unavailable: no DEFER observations collected yet.")
        if not np.isfinite(self.beta_std):
            return 0.0  # infinite uncertainty -> most conservative possible bound
        z = norm.ppf(confidence)  # negative for confidence < 0.5
        return max(0.0, self.beta_hat + z * self.beta_std)  # clip at 0 since beta > 0 by construction
 



    