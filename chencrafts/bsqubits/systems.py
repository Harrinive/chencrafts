import scqubits as scq

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Dict, List, Tuple, Callable
import copy    

# ##############################################################################
# Systems built for FlexibleSweep
# Transmon -- Resonator 

def transmon_resonator(
    sim_para: Dict[str, float],
    convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
    update_ncut: bool = True,
):
    """
    Build a transmon-resonator system using scq.HilbertSpace, 
    set update_hilbertspace_by_keyword, then return all of them as keyword arguments 
    (a dictionary) in order to be passed to FlexibleSweep. 

    Parameters are set randomly and should be determined in the FlexibleSweep.para and 
    FlexibleSweep.swept_para.
    """
    # HilbertSpace
    cavity = scq.Oscillator(
        E_osc = 3,
        truncated_dim = int(sim_para["res_dim"]),
        id_str = "res",
        l_osc = 1
    )

    while True:
        qubit = scq.Transmon(
            EJ = 5,
            EC = 0.5,
            ng = 0.25,
            ncut = int(sim_para["qubit_ncut"]),
            truncated_dim = int(sim_para["qubit_dim"]),
            id_str = "qubit",
        )

        _, bare_evecs = qubit.eigensys(int(sim_para["qubit_dim"]))
        # convergence checker: for all eigenvectors, check the smallness for 
        # the last three numbers
        conv = np.max(np.abs(bare_evecs[-3:, :]))  
        # conv = np.max(np.abs(bare_evecs[-1][-3:]))

        qubit_ncut = sim_para["qubit_ncut"]
        if convergence_range is None or not update_ncut:
            break
        elif conv > convergence_range[1]:
            qubit_ncut = int(sim_para["qubit_ncut"] * 1.5)
        elif conv < convergence_range[0]:
            qubit_ncut = int(sim_para["qubit_ncut"] / 1.5)
            break
        else:
            break

    if update_ncut:
        sim_para["qubit_ncut"] = qubit_ncut

    h_space = HilbertSpace([cavity, qubit])

    h_space.add_interaction(
        g = 0.01,
        op1 = cavity.n_operator,
        op2 = qubit.n_operator,
        add_hc = False,
        id_str = "res-qubit"
    )

    # update_hilbertspace_by_keyword
    def update_hilbertspace_by_keyword(ps: ParameterSweep,
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, ng: float, g_GHz: float,
        **kwargs
    ):
        res: scq.Oscillator = ps.subsys_by_id_str("res")
        qubit: scq.Transmon = ps.subsys_by_id_str("qubit")
        interaction = ps.hilbertspace.interaction_list[0]
        res.E_osc = E_osc_GHz
        qubit.EJ = EJ_GHz
        qubit.EC = EC_GHz
        qubit.ng = ng
        interaction.g_strength = g_GHz

    # subsys_update_info
    subsys_update_info = {
        "E_osc_GHz": ["res"],
        "EJ_GHz": ["qubit"],
        "EC_GHz": ["qubit"],
        "ng": ["qubit"],
        "g_GHz": [],
    }
    
    return {
        "sim_para": sim_para,
        "hilbertspace": h_space,
        "update_hilbertspace_by_keyword": update_hilbertspace_by_keyword,
        "subsys_update_info": subsys_update_info,
    }

