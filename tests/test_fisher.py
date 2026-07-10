import sys
from common.math_utils import boltzmann
from belief.fisher_info import fisher_info_joint
 
nominal_scores = [1.0, 2.0, 0.0]  # a1, a2, a3 -- matches the user's original example
beta, alpha = 1.0, 1.0
 
def check(label, actual, expected, tol=1e-3):
    ok = abs(actual - expected) < tol
    status = "OK" if ok else "MISMATCH"
    print(f"[{status}] {label}: got {actual:.4f}, expected {expected:.4f}")
    return ok
 
all_ok = True
 
# --- DEFER ---
r = fisher_info_joint(nominal_scores, advised_idx=None, beta=beta, alpha=alpha)
all_ok &= check("DEFER I(beta)", r["I_beta_naive"], 0.4244)
all_ok &= check("DEFER I(alpha) (should be 0)", r["I_alpha_naive"], 0.0)
all_ok &= check("DEFER I(cross) (should be 0)", r["I_cross"], 0.0)
all_ok &= check("DEFER I(beta) marginal == naive", r["I_beta_marginal"], 0.4244)
 
print()
 
# --- ADVICE on a1 (idx 0), gap=1 ---
r = fisher_info_joint(nominal_scores, advised_idx=0, beta=beta, alpha=alpha)
all_ok &= check("advise a1: I(beta) naive", r["I_beta_naive"], 0.2374)
all_ok &= check("advise a1: I(alpha) naive", r["I_alpha_naive"], 0.2490)
all_ok &= check("advise a1: I(cross)", r["I_cross"], 0.0594)
all_ok &= check("advise a1: I(beta) marginal", r["I_beta_marginal"], 0.2233)
all_ok &= check("advise a1: I(alpha) marginal", r["I_alpha_marginal"], 0.2342)
 
print()
 
# --- ADVICE on a2 (idx 1), gap=0, already best ---
r = fisher_info_joint(nominal_scores, advised_idx=1, beta=beta, alpha=alpha)
all_ok &= check("advise a2: I(beta) naive", r["I_beta_naive"], 0.7093)
all_ok &= check("advise a2: I(alpha) naive", r["I_alpha_naive"], 0.1318)
all_ok &= check("advise a2: I(cross)", r["I_cross"], 0.2991)
all_ok &= check("advise a2: I(beta) marginal", r["I_beta_marginal"], 0.0307)
all_ok &= check("advise a2: I(alpha) marginal", r["I_alpha_marginal"], 0.0057)
 
print()
 
# --- ADVICE on a3 (idx 2), gap=2, worst ---
r = fisher_info_joint(nominal_scores, advised_idx=2, beta=beta, alpha=alpha)
all_ok &= check("advise a3: I(beta) naive", r["I_beta_naive"], 0.2442)
all_ok &= check("advise a3: I(alpha) naive", r["I_alpha_naive"], 0.1670)
all_ok &= check("advise a3: I(cross)", r["I_cross"], -0.1221)
all_ok &= check("advise a3: I(beta) marginal", r["I_beta_marginal"], 0.1549)
all_ok &= check("advise a3: I(alpha) marginal", r["I_alpha_marginal"], 0.1060)
 
print()
 
# --- sanity check: ADVICE with alpha=0 should exactly equal DEFER ---
r_zero_advice = fisher_info_joint(nominal_scores, advised_idx=0, beta=beta, alpha=0.0)
r_defer = fisher_info_joint(nominal_scores, advised_idx=None, beta=beta, alpha=alpha)
all_ok &= check("advise a1 with alpha=0 == DEFER I(beta)", r_zero_advice["I_beta_naive"], r_defer["I_beta_naive"])
 
print()
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see above")