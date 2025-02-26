__all__ = [
    "compute_failure_rate", "failure_rate_is_not_nan"
]

import numpy as np
from . import (
    ResonatorTransmon, ResonatorFluxonium,
    QEC_graph, batched_sweep_general, batched_sweep_dressed_op, batched_sweep_jump_rates, batched_sweep_pulse, cat_ideal
)
from chencrafts.cqed import (
    FlexibleSweep, cat
)
import scqubits as scq
from typing import Dict, Any, Literal

def compute_failure_rate(
    para: Dict[str, float],
    sim_para: Dict[str, Any],
    ancilla_para: Dict[str, float],
    fixed_para: Dict[str, float],
    ancilla: Literal["transmon", "fluxonium"],
) -> float:
    """
    Compute the failure rate of the system.
    
    Parameters:
    para: 
        the parameters that is optimized, which may have
        some overlap between the ancilla_para and fixed_para.
    sim_para: 
        the parameters for configuring the simulation.
    ancilla_para: 
        the parameters of the ancilla
    fixed_para: 
        the parameters that is fixed in the simulation.
    ancilla: 
        the type of the ancilla.
    """
    # if we do multi-processing, we need to set the global variable
    # for every call of the function
    scq.settings.MULTIPROC = "ray"
    QEC_graph.settings.ISSUE_WARNING = False
    
    # Build system
    if ancilla == "transmon":
        sys = ResonatorTransmon(
            sim_para = sim_para,
            para = ancilla_para,
        )
    elif ancilla == "fluxonium":
        sys = ResonatorFluxonium(
            sim_para = sim_para,
            para = ancilla_para,
        )
        
    all_para = fixed_para | ancilla_para | para
    fs = FlexibleSweep(
        para = all_para,
        swept_para={},
        **sys.dict,
        **sim_para,
        lookup_scheme="LX",
        lookup_subsys_priority=[1, 0],
    )

    qubit_op_names = ["n_operator", "proj_11"]
    res_op_names = ["a_m_adag", "adag_a"]
    batched_sweep_general(fs.sweep)
    batched_sweep_dressed_op(
        fs.sweep,
        res_trunc_dim = sim_para["res_me_dim"],
        qubit_trunc_dim = sim_para["qubit_me_dim"],
        qubit_op_names = qubit_op_names,
        res_op_names = res_op_names,
    )
    batched_sweep_jump_rates(
        fs.sweep,
        qubit_op_names = qubit_op_names,
        res_op_names = res_op_names,
    )
    batched_sweep_pulse(
        fs.sweep,
        res_mode_idx = 0,
        qubit_mode_idx = 1,
        sigma_exists = "sigma" in all_para.keys(),
        bound_by_nonlin = True,
        bound_by_freq = True,
        min_sigma = 2.0,
        max_sigma = 50.0,
    )
    # Build graph
    builder = QEC_graph.FullCatTreeBuilder(
        fsweep = fs,
        sim_para=sim_para,
        new_recipe=True,
    )
    builder.idling_is_ideal = sim_para["idling_is_ideal"]

    builder.gate_1_is_ideal = sim_para["gate_1_is_ideal"]
    builder.parity_mapping_is_ideal = sim_para["parity_mapping_is_ideal"]
    builder.gate_2_is_ideal = sim_para["gate_2_is_ideal"]

    builder.qubit_measurement_is_ideal = sim_para["qubit_measurement_is_ideal"]
    builder.qubit_reset_is_ideal = sim_para["qubit_reset_is_ideal"]

    builder.build_all_processes(num_cpus=sim_para["num_cpus"])

    # Evolve
    alpha = float(fs["disp"])
    state_vec = np.array([0, 1]).astype(complex)
    state_vec /= np.linalg.norm(state_vec)

    logical_0 = cat(
        [(1, alpha), (1, -alpha)],
        basis = [cat_ideal.res_qubit_basis(
            builder.res_dim, builder.qubit_dim, (idx, 0)
        ) for idx in range(0, builder.res_dim)]
    )
    logical_1 = cat(
        [(1, 1j * alpha), (1, -1j * alpha)],
        basis = [cat_ideal.res_qubit_basis(
            builder.res_dim, builder.qubit_dim, (idx, 0)
        ) for idx in range(0, builder.res_dim)]
    )

    graph = builder.generate_tree(
        init_prob_amp_01 = tuple(state_vec),
        logical_0=logical_0,
        logical_1=logical_1,
        QEC_rounds=sim_para["QEC_rounds"],
        with_check_point=sim_para["with_check_point"],
    )

    selected_ensemble = graph.traverse(sim_para["QEC_rounds"] * 6 - 1, evolve=True)
    fid = selected_ensemble.fidelity_by_process(type="etg")
    
    # exp(-rate * t) = fid --> rate = -log(fid) / t
    return -np.log(fid) / (para["T_W"] * 1e-6 * sim_para["QEC_rounds"]) # unit: ms^-1

def failure_rate_is_not_nan(
    para: Dict[str, float],
    sim_para: Dict[str, Any],
    ancilla_para: Dict[str, float],
    fixed_para: Dict[str, float],
    ancilla: Literal["transmon", "fluxonium"],
) -> bool:
    return not np.isnan(
        compute_failure_rate(
            para, sim_para, ancilla_para, fixed_para, ancilla
        )
    )