import numpy as np
from scqubits.core.hilbert_space import HilbertSpace
from chencrafts.cqed.flexible_sweep import FlexibleSweep

from typing import List, Tuple

def CZ_analyzer(
    fs: FlexibleSweep, 
    q1_idx: int, 
    q2_idx: int, 
    param_indices: np.ndarray, 
    comp_labels: List[Tuple[int, ...]], 
):
    """
    fs must have the following keys:
    - "drive_op_0_1"    (or other q1_idx, q2_idx, similar follows)
    - "full_CZ_0_1"
    - "pure_CZ_0_1"
    """
    full_indices = fs.full_slice(param_indices)
    
    full_CZ = fs[f"full_CZ_{q1_idx}_{q2_idx}"][param_indices]
    drive_op = fs[f"drive_op_{q1_idx}_{q2_idx}"][param_indices]
    pure_CZ = fs[f"pure_CZ_{q1_idx}_{q2_idx}"][param_indices]
    hspace = fs.sweep.hilbertspace
    
    # subspace info
    print("Subspace diagonal: ")
    print(np.diag(pure_CZ.full()), "\n")
    
    # leakage 
    for bare_label in comp_labels:
        ravel_idx = np.ravel_multi_index(bare_label, hspace.subsystem_dims)
        drs_idx = fs[f"dressed_indices"][param_indices][ravel_idx]
        print(f"Leakage from {bare_label}:")
        
        # leakage destination by propagator
        dest_drs_list = np.argsort(np.abs(full_CZ.full())[:, drs_idx])[::-1]
        for dest in dest_drs_list[1:3]:
            trans_prob = np.abs(full_CZ.full())[dest, drs_idx]**2
            dest_bare_comp, occ_prob = fs.sweep.dressed_state_component(
                dest, truncate=3, param_npindices = full_indices
            )
            print(
                f"\t{trans_prob:.3f} --> drs state {dest}.",
                "Compoent:", dest_bare_comp, 
                "Probability:", [f"{p:.3f}" for p in occ_prob],
            )
        print()
            
        
    