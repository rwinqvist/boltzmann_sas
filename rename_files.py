import os
import re
from global_config import ROOT_DIR

def clean_layout_suffix(filename):
    """
    '..._L1_L2_L3_...pkl' -> '..._L3_...pkl'
    The last number in a run of _L + segments is the actual layout index
    (accumulation was monotonic: layout i's tag is _L1_L2_..._Li).
    Filenames with only a single _L segment are already clean and pass through.
    """
    match = re.search(r"(?:_L\d+)+", filename)
    if not match:
        return filename
    numbers = re.findall(r"_L(\d+)", match.group(0))
    return filename[:match.start()] + f"_L{numbers[-1]}" + filename[match.end():]


def rename_layout_files(results_dir, dry_run=True):
    renamed = []
    for fname in os.listdir(results_dir):
        cleaned = clean_layout_suffix(fname)
        if cleaned != fname:
            src = os.path.join(results_dir, fname)
            dst = os.path.join(results_dir, cleaned)
            if os.path.exists(dst):
                print(f"SKIP (target exists): {fname} -> {cleaned}")
                continue
            renamed.append((fname, cleaned))
            if not dry_run:
                os.rename(src, dst)
    return renamed


# First pass: check what would happen before touching anything
filenames = [f"{ROOT_DIR}/simulations/layered_mdp_simulations/results/d100_a3_h1_a1/bamcpbeta0-10_alpha0-5_res0.1/true_b0.5_a1.2",
             f"{ROOT_DIR}/simulations/layered_mdp_simulations/results/d100_a3_h1_a1/early_stopping_bamcpbeta0-10_alpha0-5_res0.1/true_b0.5_a1.2",
             f"{ROOT_DIR}/simulations/layered_mdp_simulations/results/d150_a3_h1_a1/bamcpbeta0-10_alpha0-5_res0.1/true_b0.5_a1.2",
             f"{ROOT_DIR}/simulations/layered_mdp_simulations/results/d150_a3_h1_a1/early_stopping_bamcpbeta0-10_alpha0-5_res0.1/true_b0.5_a1.2"]

for fn in filenames:
    changes = rename_layout_files(fn, dry_run=False)
    for old, new in changes:
        print(f"{old}  ->  {new}")

# Once the printed list looks right:
# rename_layout_files("/path/to/your/results/dir", dry_run=False)