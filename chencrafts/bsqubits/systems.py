import scqubits as scq

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

import numpy as np

from typing import Dict, List, Tuple, Union, Any

# ##############################################################################
# Systems built for FlexibleSweep
# Transmon -- Resonator 

def resonator_transmon(
    sim_para: Dict[str, Any],
    para: Dict[str, Any] = {},
    convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
    update_ncut: bool = True,
):
    """
    Build a resonator-fluxonium system using scq.HilbertSpace, 
    set update_hilbertspace_by_keyword, then return all of them as keyword arguments 
    (a dictionary) in order to be passed to FlexibleSweep. 

    Parameters are set randomly and should be determined in the FlexibleSweep.para and 
    FlexibleSweep.swept_para.
    """
    # HilbertSpace
    cavity = scq.Oscillator(
        E_osc = para.get("E_osc_GHz", 5),
        truncated_dim = int(sim_para["res_dim"]),
        id_str = "res",
        l_osc = 1
    )

    qubit_ncut = sim_para["qubit_ncut"]
    while True:
        qubit = scq.Transmon(
            EJ = para.get("EJ_GHz", 5),
            EC = para.get("EC_GHz", 0.2),
            ng = para.get("ng", 0.25),
            ncut = qubit_ncut,
            truncated_dim = int(sim_para["qubit_dim"]),
            id_str = "qubit",
        )

        _, bare_evecs = qubit.eigensys(int(sim_para["qubit_dim"]))
        # convergence checker: for all eigenvectors, check the smallness for 
        # the last three numbers
        conv = np.max(np.abs(bare_evecs[-3:, :]))  
        # conv = np.max(np.abs(bare_evecs[-1][-3:]))

        if convergence_range is None or not update_ncut:
            break
        elif conv > convergence_range[1]:
            qubit_ncut = int(qubit_ncut * 1.5)
        elif conv < convergence_range[0]:
            qubit_ncut = int(qubit_ncut / 1.5)
            break
        else:
            break
    if update_ncut:
        sim_para["qubit_ncut"] = qubit_ncut

    h_space = HilbertSpace([cavity, qubit])

    h_space.add_interaction(
        g = para.get("g_GHz", 0.01),
        op1 = cavity.n_operator,
        op2 = qubit.n_operator,
        add_hc = False,
        id_str = "res-qubit"
    )

    # update_hilbertspace
    def update_hilbertspace(
        hilbertspace: HilbertSpace, 
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, ng: float, g_GHz: float,
        **kwargs
    ):
        res: scq.Oscillator = hilbertspace.subsys_by_id_str("res")
        qubit: scq.Transmon = hilbertspace.subsys_by_id_str("qubit")
        interaction = hilbertspace.interaction_list[0]

        res.E_osc = E_osc_GHz
        qubit.EJ = EJ_GHz
        qubit.EC = EC_GHz
        qubit.ng = ng
        interaction.g_strength = g_GHz

    # update_hilbertspace_by_keyword
    def update_hilbertspace_by_keyword(
        ps: ParameterSweep,
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, ng: float, g_GHz: float,
        **kwargs
    ):
        update_hilbertspace(
            ps.hilbertspace, E_osc_GHz, EJ_GHz, EC_GHz, ng, g_GHz
        )

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
        "qubit": qubit,
        "res": cavity,
        "update_hilbertspace": update_hilbertspace,
        "update_hilbertspace_by_keyword": update_hilbertspace_by_keyword,
        "subsys_update_info": subsys_update_info,
    }


def resonator_fluxonium(
    sim_para: Dict[str, Any],
    para: Dict[str, Any] = {},
    convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
    update_cutoff: bool = True,
):
    """
    Build a resonator-fluxonium system using scq.HilbertSpace, 
    set update_hilbertspace_by_keyword, then return all of them as keyword arguments 
    (a dictionary) in order to be passed to FlexibleSweep. 

    Parameters are set randomly and should be determined in the FlexibleSweep.para and 
    FlexibleSweep.swept_para.
    """
    # HilbertSpace
    cavity = scq.Oscillator(
        E_osc = para.get("E_osc_GHz", 5),
        truncated_dim = int(sim_para["res_dim"]),
        id_str = "res",
        l_osc = 1
    )

    qubit_ncut = sim_para["qubit_cutoff"]
    while True:
        qubit = scq.Fluxonium(
            EJ = para.get("EJ_GHz", 5),
            EC = para.get("EC_GHz", 0.2),
            EL = para.get("EL_GHz", 0.5),
            flux = para.get("flux", 0.5),
            cutoff = qubit_ncut,
            truncated_dim = int(sim_para["qubit_dim"]),
            id_str = "qubit",
        )

        _, bare_evecs = qubit.eigensys(int(sim_para["qubit_dim"]))
        # convergence checker: for all eigenvectors, check the smallness for 
        # the last three numbers
        conv = np.max(np.abs(bare_evecs[-3:, :]))  
        # conv = np.max(np.abs(bare_evecs[-1][-3:]))

        if convergence_range is None or not update_cutoff:
            break
        elif conv > convergence_range[1]:
            qubit_ncut = int(qubit_ncut * 1.5)
        elif conv < convergence_range[0]:
            qubit_ncut = int(qubit_ncut / 1.5)
            break
        else:
            break
    if update_cutoff:
        sim_para["qubit_cutoff"] = qubit_ncut

    h_space = HilbertSpace([cavity, qubit])

    h_space.add_interaction(
        g = para.get("g_GHz", 0.01),
        op1 = cavity.n_operator,
        op2 = qubit.n_operator,
        add_hc = False,
        id_str = "res-qubit"
    )

    # update_hilbertspace
    def update_hilbertspace(
        hilbertspace: HilbertSpace,
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, EL_GHz: float, flux: float,
        g_GHz: float,
        **kwargs
    ):
        res: scq.Oscillator = hilbertspace.subsys_by_id_str("res")
        qubit: scq.Fluxonium = hilbertspace.subsys_by_id_str("qubit")
        interaction = hilbertspace.interaction_list[0]
        res.E_osc = E_osc_GHz
        qubit.EJ = EJ_GHz
        qubit.EC = EC_GHz
        qubit.EL = EL_GHz
        qubit.flux = flux
        interaction.g_strength = g_GHz

    # update_hilbertspace_by_keyword
    def update_hilbertspace_by_keyword(ps: ParameterSweep,
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, EL_GHz: float, flux: float, 
        g_GHz: float,
        **kwargs
    ):
        update_hilbertspace(
            ps.hilbertspace, E_osc_GHz, EJ_GHz, EC_GHz, EL_GHz, flux, g_GHz
        )

    # subsys_update_info
    subsys_update_info = {
        "E_osc_GHz": ["res"],
        "EJ_GHz": ["qubit"],
        "EC_GHz": ["qubit"],
        "EL_GHz": ["qubit"],
        "flux": ["qubit"],
        "g_GHz": [],
    }

    return {
        "sim_para": sim_para,
        "hilbertspace": h_space,
        "qubit": qubit,
        "res": cavity,
        "update_hilbertspace": update_hilbertspace,
        "update_hilbertspace_by_keyword": update_hilbertspace_by_keyword,
        "subsys_update_info": subsys_update_info,
    }


def fluxonium_resonator_fluxonium(
    sim_para: Dict[str, Any],
    para: Dict[str, Any] = {},
    convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
    update_cutoff: bool = True,
):
    """
    Build a fl-res-fl system using scq.HilbertSpace, 
    set update_hilbertspace_by_keyword, then return all of them as keyword arguments 
    (a dictionary) in order to be passed to FlexibleSweep. 

    Parameters are set randomly and should be determined in the FlexibleSweep.para and 
    FlexibleSweep.swept_para.
    """
    # HilbertSpace
    cavity = scq.Oscillator(
        E_osc = para.get("E_osc_GHz", 5),
        truncated_dim = int(sim_para["res_dim"]),
        id_str = "res",
        l_osc = 1
    )

    qubit_ncut1 = sim_para["qubit_cutoff1"]
    while True:
        qubit1 = scq.Fluxonium(
            EJ = para.get("EJ1_GHz", 5),
            EC = para.get("EC1_GHz", 0.2),
            EL = para.get("EL1_GHz", 0.5),
            flux = para.get("flux1", 0.5),
            cutoff = qubit_ncut1,
            truncated_dim = int(sim_para["qubit_dim1"]),
            id_str = "qubit1",
        )

        _, bare_evecs = qubit1.eigensys(int(sim_para["qubit_dim1"]))
        # convergence checker: for all eigenvectors, check the smallness for 
        # the last three numbers
        conv = np.max(np.abs(bare_evecs[-3:, :]))  
        # conv = np.max(np.abs(bare_evecs[-1][-3:]))

        if convergence_range is None or not update_cutoff:
            break
        elif conv > convergence_range[1]:
            qubit_ncut1 = int(qubit_ncut1 * 1.5)
        elif conv < convergence_range[0]:
            qubit_ncut1 = int(qubit_ncut1 / 1.5)
            break
        else:
            break
    if update_cutoff:
        sim_para["qubit_cutoff1"] = qubit_ncut1

    qubit_ncut2 = sim_para["qubit_cutoff2"]
    while True:
        qubit2 = scq.Fluxonium(
            EJ = para.get("EJ2_GHz", 5),
            EC = para.get("EC2_GHz", 0.2),
            EL = para.get("EL2_GHz", 0.5),
            flux = para.get("flux2", 0.5),
            cutoff = qubit_ncut2,
            truncated_dim = int(sim_para["qubit_dim2"]),
            id_str = "qubit2",
        )

        _, bare_evecs = qubit2.eigensys(int(sim_para["qubit_dim2"]))
        # convergence checker: for all eigenvectors, check the smallness for 
        # the last three numbers
        conv = np.max(np.abs(bare_evecs[-3:, :]))  
        # conv = np.max(np.abs(bare_evecs[-1][-3:]))

        if convergence_range is None or not update_cutoff:
            break
        elif conv > convergence_range[1]:
            qubit_ncut2 = int(qubit_ncut2 * 1.5)
        elif conv < convergence_range[0]:
            qubit_ncut2 = int(qubit_ncut2 / 1.5)
            break
        else:
            break
    if update_cutoff:
        sim_para["qubit_cutoff2"] = qubit_ncut2

    h_space = HilbertSpace([qubit1, cavity, qubit2])

    h_space.add_interaction(
        g = para.get("g1_GHz", 0.01),
        op1 = qubit1.n_operator,
        op2 = cavity.n_operator,
        add_hc = False,
        id_str = "qubit1-res"
    )

    h_space.add_interaction(
        g = para.get("g2_GHz", 0.01),
        op1 = cavity.n_operator,
        op2 = qubit2.n_operator,
        add_hc = False,
        id_str = "res-qubit2"
    )

    # update_hilbertspace
    def update_hilbertspace(
        hilbertspace: HilbertSpace,
        E_osc_GHz: float, 
        EJ1_GHz: float, EC1_GHz: float, EL1_GHz: float, flux1: float,
        EJ2_GHz: float, EC2_GHz: float, EL2_GHz: float, flux2: float,
        g1_GHz: float, g2_GHz: float,
        **kwargs
    ):
        res: scq.Oscillator = hilbertspace.subsys_by_id_str("res")
        qubit1: scq.Fluxonium = hilbertspace.subsys_by_id_str("qubit1")
        qubit2: scq.Fluxonium = hilbertspace.subsys_by_id_str("qubit2")
        interaction1 = hilbertspace.interaction_list[0]
        interaction2 = hilbertspace.interaction_list[1]

        res.E_osc = E_osc_GHz

        qubit1.EJ = EJ1_GHz
        qubit1.EC = EC1_GHz
        qubit1.EL = EL1_GHz
        qubit1.flux = flux1

        qubit2.EJ = EJ2_GHz
        qubit2.EC = EC2_GHz
        qubit2.EL = EL2_GHz
        qubit2.flux = flux2

        interaction1.g_strength = g1_GHz
        interaction2.g_strength = g2_GHz


    # update_hilbertspace_by_keyword
    def update_hilbertspace_by_keyword(ps: ParameterSweep,
        E_osc_GHz: float, 
        EJ1_GHz: float, EC1_GHz: float, EL1_GHz: float, flux1: float,
        EJ2_GHz: float, EC2_GHz: float, EL2_GHz: float, flux2: float,
        g1_GHz: float, g2_GHz: float,
        **kwargs
    ):
        update_hilbertspace(
            ps.hilbertspace, 
            E_osc_GHz,
            EJ1_GHz, EC1_GHz, EL1_GHz, flux1,
            EJ2_GHz, EC2_GHz, EL2_GHz, flux2,
            g1_GHz, g2_GHz,
            **kwargs
        )

    # subsys_update_info
    subsys_update_info = {
        "E_osc_GHz": ["res"],
        "EJ1_GHz": ["qubit1"],
        "EC1_GHz": ["qubit1"],
        "EL1_GHz": ["qubit1"],
        "flux1": ["qubit1"],
        "EJ2_GHz": ["qubit2"],
        "EC2_GHz": ["qubit2"],
        "EL2_GHz": ["qubit2"],
        "flux2": ["qubit2"],
        "g1_GHz": [],
        "g2_GHz": [],
    }

    return {
        "sim_para": sim_para,
        "hilbertspace": h_space,
        "qubit1": qubit1,
        "res": cavity,
        "qubit2": qubit2,
        "update_hilbertspace": update_hilbertspace,
        "update_hilbertspace_by_keyword": update_hilbertspace_by_keyword,
        "subsys_update_info": subsys_update_info,
    }

