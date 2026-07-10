from belief.fisher_info import fisher_info_alpha_ceiling

class AlphaCollapseMonitor:
    """
    Tracks whether it's safe to stop updating alpha's belief and freeze it to
    a fixed value, based on whether the current belief's information about
    beta implies further ADVICE observations can meaningfully narrow alpha.
 
    Design, matching the safety requirements worked out in discussion:
      - Burn-in: does not check at all until `min_advice_obs` ADVICE
        observations have been seen (checking on a flat/near-flat belief is
        just measuring prior noise, not signal).
      - Periodic: only re-evaluates every `check_every` ADVICE observations,
        not every step (a single quiet window means little on its own).
      - Pessimistic beta: uses belief.get_beta_credible_bound(confidence) --
        a LOW-tail estimate of beta, not the mean -- so a temporarily-low
        beta reading can't trigger a premature collapse; the collapse is
        only trusted when safe even in the worst plausible case.
      - Optimistic alpha-info ceiling: uses fisher_info_alpha_ceiling (the
        best possible per-step alpha information, at p_advised=0.5) as the
        forecast for remaining steps. Since this is an upper bound on what's
        achievable, if even this optimistic forecast says "not worth it",
        that conclusion is trustworthy.
      - Hysteresis: requires `required_consecutive` consecutive triggering
        checks before actually collapsing, guarding against one noisy check.
 
    Usage: call .check(belief, n_advice_so_far, n_remaining) once per step (or
    at least once per ADVICE observation). When it returns True, freeze
    alpha's belief to .collapse_value and stop calling .check() further calls
    are harmless no-ops once .collapsed is True).
    """

    def __init__(self, min_advice_obs=15, check_every=10, tolerance=0.05, confidence=0.1, required_consecutive=2):
        self.min_advice_obs = min_advice_obs
        self.check_every = check_every
        self.tolerance = tolerance
        self.confidence = confidence
        self.required_consecutive = required_consecutive

        self._consecutive_triggers = 0 
        self._collapsed = False 
        self.collapse_value = None 
        self.collapse_step = None   # n_advice_so_far at the moment of collapse, for diagnostics
        self.history = [] # list of (n_advice_so_far, pessimistic_beta, current_var, forecast_var, triggered) for later inspection/plotting

    @property
    def collapsed(self):
        return self._collapsed
    

    def check(self, belief, n_advice_so_far, n_remaining):
        """
        Returns True exactly on the step collapse is triggered. Returns False
        on every other call, including all calls after collapse has already
        happened (check `.collapsed` to know if that's why).
        """
        if self._collapsed:
            return False
        if n_remaining <= 0:
            return False  # nothing left to decide either way -- not a meaningful trigger
        if n_advice_so_far < self.min_advice_obs:
            return False
        if n_advice_so_far % self.check_every != 0:
            return False

        pessimistic_beta = belief.get_beta_credible_bound(confidence=self.confidence)
        info_rate_ceiling = fisher_info_alpha_ceiling(pessimistic_beta)
 
        stats = belief.get_stats()
        current_var = stats["alpha_std"] ** 2

        if current_var <= 1e-10:
            # already essentially a point mass -- nothing left to learn regardless of forecast
            triggered = True
            forecast_var = current_var
        else:
            forecast_var = 1.0 / (1.0 / current_var + n_remaining * info_rate_ceiling)
            relative_improvement = (current_var - forecast_var) / current_var
            triggered = relative_improvement < self.tolerance
 
        self.history.append({
            "n_advice_so_far": n_advice_so_far,
            "pessimistic_beta": pessimistic_beta,
            "current_var": current_var,
            "forecast_var": forecast_var,
            "triggered": triggered,
        })
 
        if triggered:
            self._consecutive_triggers += 1
        else:
            self._consecutive_triggers = 0
 
        if self._consecutive_triggers >= self.required_consecutive:
            self._collapsed = True
            self.collapse_value = self._compute_collapse_value(belief)
            self.collapse_step = n_advice_so_far
            return True
 
        return False
    

    def _compute_collapse_value(self, belief):
        """
        Freeze to the MODE of the current alpha marginal (not the mean --
        the mean can land in a low-probability valley if the belief is
        multimodal at the moment of collapse; see discussion).
 
        NOTE: this does not yet check for multimodality before collapsing --
        a belief with two separated peaks would still collapse to whichever
        peak happens to be tallest, without flagging that the situation is
        genuinely ambiguous. That check is a known, not-yet-implemented
        refinement (see conversation) -- worth adding if collapse quality
        turns out to be a problem in practice.
        """
        marginal = belief.get_alpha_marginal()
        return max(marginal, key=marginal.get)