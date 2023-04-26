import scqubits as scq

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

import numpy as np

from typing import Dict, List, Tuple, Union, Any

class JointSystems:
    def __init__(
        self, 
        sim_para: Dict[str, Any],
        para: Dict[str, Any] = {},
    ):
        self.sim_para = sim_para
        self.para = para

    def check_conv(
        self, 
        bare_evecs: np.ndarray,
        current_cut: int,
        convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
        update: bool = True,
    ) -> int:
        # convergence checker: for all eigenvectors, check the smallness for 
        # the last three numbers
        conv = np.max(np.abs(bare_evecs[-3:, :]))  
        # conv = np.max(np.abs(bare_evecs[-1][-3:]))

        if convergence_range is None or not update:
            return -1
        elif conv > convergence_range[1]:
            return int(current_cut * 1.5)
        elif conv < convergence_range[0]:
            return int(current_cut / 1.5)
        else:
            return -1

    _dict: Dict[str, Any]

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __iter__(self):
        return self._dict.__iter__()
    


# ##############################################################################
# Systems built for FlexibleSweep
# Transmon -- Resonator 

class ResonatorTransmon(JointSystems):
    def __init__(
        self,
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
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        qubit_ncut = sim_para["qubit_ncut"]
        while True:
            self.qubit = scq.Transmon(
                EJ = para.get("EJ_GHz", 5),
                EC = para.get("EC_GHz", 0.2),
                ng = para.get("ng", 0.25),
                ncut = qubit_ncut,
                truncated_dim = int(sim_para["qubit_dim"]),
                id_str = "qubit",
            )

            _, bare_evecs = self.qubit.eigensys(int(sim_para["qubit_dim"]))
            new_cut = self.check_conv(
                bare_evecs, qubit_ncut, convergence_range, update_ncut
            )
            if new_cut == -1:
                break
            else:
                qubit_ncut = new_cut
        if update_ncut:
            sim_para["qubit_ncut"] = qubit_ncut

        self.hilbertspace = HilbertSpace([self.res, self.qubit])

        self.hilbertspace.add_interaction(
            g = para.get("g_GHz", 0.01),
            op1 = self.res.n_operator,
            op2 = self.qubit.n_operator,
            add_hc = False,
            id_str = "res-qubit"
        )

        # subsys_update_info
        self.subsys_update_info = {
            "E_osc_GHz": ["res"],
            "EJ_GHz": ["qubit"],
            "EC_GHz": ["qubit"],
            "ng": ["qubit"],
            "g_GHz": [],
        }

    def update_hilbertspace(
        self,
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

    def update_hilbertspace_by_keyword(
        self,
        ps: ParameterSweep,
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, ng: float, g_GHz: float,
        **kwargs
    ):
        self.update_hilbertspace(
            ps.hilbertspace, E_osc_GHz, EJ_GHz, EC_GHz, ng, g_GHz
        )

    def _dict(self) -> Dict[str, Any]:
        return {
            "sim_para": self.sim_para,
            "hilbertspace": self.hilbertspace,
            "qubit": self.qubit,
            "res": self.res,
            "update_hilbertspace": self.update_hilbertspace,
            "update_hilbertspace_by_keyword": self.update_hilbertspace_by_keyword,
            "subsys_update_info": self.subsys_update_info,
        }

class ResonatorFluxonium(JointSystems):

    def __init__(
        self,
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
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        qubit_ncut = sim_para["qubit_cutoff"]
        while True:
            self.qubit = scq.Fluxonium(
                EJ = para.get("EJ_GHz", 5),
                EC = para.get("EC_GHz", 0.2),
                EL = para.get("EL_GHz", 0.5),
                flux = para.get("flux", 0.5),
                cutoff = qubit_ncut,
                truncated_dim = int(sim_para["qubit_dim"]),
                id_str = "qubit",
            )

            _, bare_evecs = self.qubit.eigensys(int(sim_para["qubit_dim"]))
            new_cut = self.check_conv(
                bare_evecs, qubit_ncut, convergence_range, update_cutoff
            )
            if new_cut == -1:
                break
            else:
                qubit_ncut = new_cut
        if update_cutoff:
            sim_para["qubit_cutoff"] = qubit_ncut

        self.hilbertspace = HilbertSpace([self.res, self.qubit])

        self.hilbertspace.add_interaction(
            g = para.get("g_GHz", 0.01),
            op1 = self.res.n_operator,
            op2 = self.qubit.n_operator,
            add_hc = False,
            id_str = "res-qubit"
        )

        # subsys_update_info
        self.subsys_update_info = {
            "E_osc_GHz": ["res"],
            "EJ_GHz": ["qubit"],
            "EC_GHz": ["qubit"],
            "EL_GHz": ["qubit"],
            "flux": ["qubit"],
            "g_GHz": [],
        }

    def update_hilbertspace(
        self,
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

    def update_hilbertspace_by_keyword(
        self,
        ps: ParameterSweep,
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, EL_GHz: float, flux: float, 
        g_GHz: float,
        **kwargs
    ):
        self.update_hilbertspace(
            ps.hilbertspace, E_osc_GHz, EJ_GHz, EC_GHz, EL_GHz, flux, g_GHz
        )

    def _dict(self) -> Dict[str, Any]:
        return {
            "sim_para": self.sim_para,
            "hilbertspace": self.hilbertspace,
            "qubit": self.qubit,
            "res": self.res,
            "update_hilbertspace": self.update_hilbertspace,
            "update_hilbertspace_by_keyword": self.update_hilbertspace_by_keyword,
            "subsys_update_info": self.subsys_update_info,
        }

class FluxoniumResonatorFluxonium(JointSystems):
    def __init__(
        self,
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
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        qubit_ncut1 = sim_para["qubit_cutoff1"]
        while True:
            self.qubit1 = scq.Fluxonium(
                EJ = para.get("EJ1_GHz", 5),
                EC = para.get("EC1_GHz", 0.2),
                EL = para.get("EL1_GHz", 0.5),
                flux = para.get("flux1", 0.5),
                cutoff = qubit_ncut1,
                truncated_dim = int(sim_para["qubit_dim1"]),
                id_str = "qubit1",
            )

            _, bare_evecs = self.qubit1.eigensys(int(sim_para["qubit_dim1"]))
            new_cut1 = self.check_conv(
                bare_evecs, qubit_ncut1, convergence_range, update_cutoff
            )
            if new_cut1 == -1:
                break
            else:
                qubit_ncut1 = new_cut1
        if update_cutoff:
            sim_para["qubit_cutoff1"] = qubit_ncut1

        qubit_ncut2 = sim_para["qubit_cutoff2"]
        while True:
            self.qubit2 = scq.Fluxonium(
                EJ = para.get("EJ2_GHz", 5),
                EC = para.get("EC2_GHz", 0.2),
                EL = para.get("EL2_GHz", 0.5),
                flux = para.get("flux2", 0.5),
                cutoff = qubit_ncut2,
                truncated_dim = int(sim_para["qubit_dim2"]),
                id_str = "qubit2",
            )

            _, bare_evecs = self.qubit2.eigensys(int(sim_para["qubit_dim2"]))
            new_cut2 = self.check_conv(
                bare_evecs, qubit_ncut2, convergence_range, update_cutoff
            )
            if new_cut2 == -1:
                break
            else:
                qubit_ncut2 = new_cut2
        if update_cutoff:
            sim_para["qubit_cutoff2"] = qubit_ncut2

        self.hilbertspace = HilbertSpace([self.qubit1, self.res, self.qubit2])

        self.hilbertspace.add_interaction(
            g = para.get("g1_GHz", 0.01),
            op1 = self.qubit1.n_operator,
            op2 = self.res.n_operator,
            add_hc = False,
            id_str = "qubit1-res"
        )

        self.hilbertspace.add_interaction(
            g = para.get("g2_GHz", 0.01),
            op1 = self.res.n_operator,
            op2 = self.qubit2.n_operator,
            add_hc = False,
            id_str = "res-qubit2"
        )

        self.subsys_update_info = {
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

    def update_hilbertspace(
        self,
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


    def update_hilbertspace_by_keyword(
        self,    
        ps: ParameterSweep,
        E_osc_GHz: float, 
        EJ1_GHz: float, EC1_GHz: float, EL1_GHz: float, flux1: float,
        EJ2_GHz: float, EC2_GHz: float, EL2_GHz: float, flux2: float,
        g1_GHz: float, g2_GHz: float,
        **kwargs
    ):
        self.update_hilbertspace(
            ps.hilbertspace, 
            E_osc_GHz,
            EJ1_GHz, EC1_GHz, EL1_GHz, flux1,
            EJ2_GHz, EC2_GHz, EL2_GHz, flux2,
            g1_GHz, g2_GHz,
            **kwargs
        )


    def _dict(self) -> Dict[str, Any]:
        return {
            "sim_para": self.sim_para,
            "hilbertspace": self.hilbertspace,
            "qubit1": self.qubit1,
            "res": self.res,
            "qubit2": self.qubit2,
            "update_hilbertspace": self.update_hilbertspace,
            "update_hilbertspace_by_keyword": self.update_hilbertspace_by_keyword,
            "subsys_update_info": self.subsys_update_info,
        }

