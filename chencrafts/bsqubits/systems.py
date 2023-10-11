import scqubits as scq

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

import numpy as np

from abc import ABC, abstractproperty
from typing import Dict, List, Tuple, Union, Any, Callable
import copy

class JointSystems(ABC):
    def __init__(
        self, 
        sim_para: Dict[str, Any],
        para: Dict[str, Any] = {},
    ):
        self.sim_para = sim_para
        self.para = para

    def qubit_init_w_conv_check(
        self, 
        qubit_init_ret_evecs: Callable,
        init_cut: int,
        convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
        update: bool = True,
    ) -> None:
        """
        Check the convergence of the eigensolver for the qubit subsystem.
        
        Parameters
        ----------
        qubit_init_ret_evecs : Callable
            A function that initializes the qubit subsystem and returns 
            the bare eigenvectors.
        init_cut : int
            The initial cutoff dimension of the qubit Hilbert space.
        convergence_range : Tuple[float, float] | None, optional
            The convergence range for the eigensolver. The default is (1e-10, 1e-6).
        update : bool, optional
            Whether to update the cutoff dimension of the qubit Hilbert space. 
            The default is True.

        """
    
        last_operation = 0
        current_cut = init_cut
        
        while True:
            bare_evecs = qubit_init_ret_evecs(current_cut)

            # convergence checker: for all eigenvectors, check the smallness for 
            # the last three numbers
            conv = np.max(np.abs(bare_evecs[-3:, :]))  
            # conv = np.max(np.abs(bare_evecs[-1][-3:]))

            # to avoid the back-and-forth oscillation, we need to check the
            # operational history. last_operation = 1 means the previous cut was large and 
            # was reduced. And last_operation = -1 means the last cut is smaller and was 
            # increased. 
            if convergence_range is None or not update:
                break
            elif conv > convergence_range[1]:
                last_operation = -1
                current_cut = int(current_cut * 1.5)
            elif conv < convergence_range[0]:
                if last_operation == -1:
                    # When last_operation = -1 (meaning the conv was too BIG and the cut 
                    # number was reduced to a value that is too small) and the current conv is 
                    # too SMALL, we should not increase the cut number again, it might leads to 
                    # back-and-forth oscillation.
                    break
                last_operation = 1
                current_cut = int(current_cut / 1.5)
            else:
                break

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()
    
    def items(self):
        return self.dict.items()
    
    def __getitem__(self, key):
        return self.dict[key]
    
    def __iter__(self):
        return self.dict.__iter__()

    @abstractproperty
    def dict(self) -> Dict[str, Any]:
        ...


# ##############################################################################
# Systems built for FlexibleSweep
# Transmon -- Resonator 

class ResonatorTransmon(JointSystems):
    def _qubit_init(self, cut):
        self.qubit = scq.Transmon(
            EJ = self.para.get("EJ_GHz", 5),
            EC = self.para.get("EC_GHz", 0.2),
            ng = self.para.get("ng", 0.25),
            ncut = cut,
            truncated_dim = int(self.sim_para["qubit_dim"]),
            id_str = "qubit",
        )

        _, bare_evecs = self.qubit.eigensys(int(self.sim_para["qubit_dim"]))

        return bare_evecs
    
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

        Parameters
        ----------
        sim_para : Dict[str, Any]
            A dictionary containing simulation parameters.
            sim_para should contain the following keys:
                "res_dim": int
                    The dimension of the resonator Hilbert space.
                "qubit_dim": int
                    The dimension of the qubit Hilbert space.
                "qubit_ncut": int
                    The cutoff dimension of the qubit Hilbert space.
        para : Dict[str, Any], optional
            A dictionary containing system parameters. The default is {}.
            para should contain the following keys:
                "E_osc_GHz": float
                    The frequency of the resonator in GHz.
                "EJ_GHz": float
                    The Josephson energy of the fluxonium qubit in GHz.
                "EC_GHz": float
                    The charging energy of the fluxonium qubit in GHz.
                "ng": float
                    The offset charge of the fluxonium qubit.
                "g_GHz": float
                    The coupling strength between the resonator and the qubit in GHz.
        convergence_range : Tuple[float, float] | None, optional
            The convergence range for the eigensolver. The default is (1e-10, 1e-6).
        
        """
        super().__init__(sim_para, para)

        # HilbertSpace
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        self.qubit_init_w_conv_check(
            self._qubit_init, 
            sim_para["qubit_ncut"], 
            convergence_range, 
            update_ncut
        )
        if update_ncut:
            sim_para["qubit_ncut"] = self.qubit.ncut

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
            "E_osc_GHz": [self.res],
            "EJ_GHz": [self.qubit],
            "EC_GHz": [self.qubit],
            "ng": [self.qubit],
            "g_GHz": [],
        }

    def update_hilbertspace(
        self,
        hilbertspace: HilbertSpace, 
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, ng: float, g_GHz: float,
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
        **kwargs
    ):
        kwargs = copy.deepcopy(kwargs)
        self.update_hilbertspace(
            ps.hilbertspace, 
            E_osc_GHz = kwargs.pop("E_osc_GHz", self.res.E_osc), 
            EJ_GHz = kwargs.pop("EJ_GHz", self.qubit.EJ),
            EC_GHz = kwargs.pop("EC_GHz", self.qubit.EC),
            ng = kwargs.pop("ng", self.qubit.ng),
            g_GHz = kwargs.pop("g_GHz", ps.hilbertspace.interaction_list[0].g_strength),
        )

    @property
    def dict(self) -> Dict[str, Any]:
        """
        Return a dictionary containing the system information.

        Returns
        -------
        Dict[str, Any]
            Dict[str, Any]
            A dictionary containing the following keys:
                "sim_para": Dict[str, Any]
                    A dictionary containing simulation parameters.
                "hilbertspace": HilbertSpace
                    The Hilbert space of the system.
                "qubit": scq.Transmon
                    The transmon qubit subsystem.
                "res": scq.Oscillator
                    The resonator subsystem.
                "update_hilbertspace": Callable
                    A function for updating the Hilbert space of the system.
                "update_hilbertspace_by_keyword": Callable
                    A function for updating the Hilbert space of the system using keyword arguments.
                "subsys_update_info": Dict[str, List[Subsystem]]
                    A dictionary containing information about the subsystems that can be updated.               
        """
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
    def _qubit_init(self, cut):
        """
        Initialize the fluxonium qubit subsystem.

        Parameters
        ----------
        cut : int
            The cutoff dimension of the qubit Hilbert space.

        Returns
        -------
        np.ndarray
            The bare eigenvectors of the qubit subsystem.
        """
        self.qubit = scq.Fluxonium(
            EJ = self.para.get("EJ_GHz", 5),
            EC = self.para.get("EC_GHz", 0.2),
            EL = self.para.get("EL_GHz", 0.5),
            flux = self.para.get("flux", 0.5),
            cutoff = cut,
            truncated_dim = int(self.sim_para["qubit_dim"]),
            id_str = "qubit",
        )

        _, bare_evecs = self.qubit.eigensys(int(self.sim_para["qubit_dim"]))

        return bare_evecs

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

        Parameters
        ----------
        sim_para : Dict[str, Any]
            A dictionary containing simulation parameters.
            sim_para should contain the following keys:
                "res_dim": int
                    The dimension of the resonator Hilbert space.
                "qubit_dim": int
                    The dimension of the qubit Hilbert space.
                "qubit_cutoff": int
                    The cutoff dimension of the qubit Hilbert space.
        para : Dict[str, Any], optional
            A dictionary containing system parameters. The default is {}.
            para should contain the following keys:
                "E_osc_GHz": float
                    The frequency of the resonator in GHz.
                "EJ_GHz": float
                    The Josephson energy of the fluxonium qubit in GHz.
                "EC_GHz": float
                    The charging energy of the fluxonium qubit in GHz.
                "EL_GHz": float
                    The inductive energy of the fluxonium qubit in GHz.
                "flux": float
                    The flux bias of the fluxonium qubit.
                "g_GHz": float
                    The coupling strength between the resonator and the qubit in GHz.
        convergence_range : Tuple[float, float] | None, optional
            The convergence range for the eigensolver. The default is (1e-10, 1e-6).
        update_cutoff : bool, optional
            Whether to update the qubit cutoff dimension in sim_para. The default is True.
        """
        super().__init__(sim_para, para)
        
        # HilbertSpace
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        self.qubit_init_w_conv_check(
            self._qubit_init, 
            sim_para["qubit_cutoff"], 
            convergence_range, 
            update_cutoff
        )
        if update_cutoff:
            sim_para["qubit_cutoff"] = self.qubit.cutoff

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
            "E_osc_GHz": [self.res],
            "EJ_GHz": [self.qubit],
            "EC_GHz": [self.qubit],
            "EL_GHz": [self.qubit],
            "flux": [self.qubit],
            "g_GHz": [],
        }

    def update_hilbertspace(
        self,
        hilbertspace: HilbertSpace,
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, EL_GHz: float, flux: float,
        g_GHz: float,
    ):
        """
        Update the Hilbert space of the system.

        Parameters
        ----------
        hilbertspace : HilbertSpace
            The Hilbert space of the system.
        E_osc_GHz : float
            The frequency of the resonator in GHz.
        EJ_GHz : float
            The Josephson energy of the fluxonium qubit in GHz.
        EC_GHz : float
            The charging energy of the fluxonium qubit in GHz.
        EL_GHz : float
            The inductive energy of the fluxonium qubit in GHz.
        flux : float
            The flux bias of the fluxonium qubit.
        g_GHz : float
            The coupling strength between the resonator and the qubit in GHz.
        """
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
        **kwargs,
    ):
        """
        Update the Hilbert space of the system using keyword arguments.

        Parameters
        ----------
        ps : ParameterSweep
            The parameter sweep object.
        **kwargs : Dict[str, Any]
            Keyword arguments for updating the Hilbert space.
            The following keys are supported:  
                "E_osc_GHz": float
                    The frequency of the resonator in GHz.
                "EJ_GHz": float
                    The Josephson energy of the fluxonium qubit in GHz.
                "EC_GHz": float
                    The charging energy of the fluxonium qubit in GHz.
                "EL_GHz": float
                    The inductive energy of the fluxonium qubit in GHz.
                "flux": float
                    The flux bias of the fluxonium qubit.
                "g_GHz": float
                    The coupling strength between the resonator and the qubit in GHz.
        """
        kwargs = copy.deepcopy(kwargs)
        self.update_hilbertspace(
            ps.hilbertspace, 
            E_osc_GHz = kwargs.pop("E_osc_GHz", self.res.E_osc), 
            EJ_GHz = kwargs.pop("EJ_GHz", self.qubit.EJ),
            EC_GHz = kwargs.pop("EC_GHz", self.qubit.EC),
            EL_GHz = kwargs.pop("EL_GHz", self.qubit.EL),
            flux = kwargs.pop("flux", self.qubit.flux),
            g_GHz = kwargs.pop("g_GHz", self.hilbertspace.interaction_list[0].g_strength),
        )

    @property
    def dict(self) -> Dict[str, Any]:
        """
        Return a dictionary containing the system information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the following keys:
                "sim_para": Dict[str, Any]
                    A dictionary containing simulation parameters.
                "hilbertspace": HilbertSpace
                    The Hilbert space of the system.
                "qubit": scq.Fluxonium
                    The fluxonium qubit subsystem.
                "res": scq.Oscillator
                    The resonator subsystem.
                "update_hilbertspace": Callable
                    A function for updating the Hilbert space of the system.
                "update_hilbertspace_by_keyword": Callable
                    A function for updating the Hilbert space of the system using keyword arguments.
                "subsys_update_info": Dict[str, List[Subsystem]]
                    A dictionary containing information about the subsystems that can be updated.
        """
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
    def _qubit1_init(self, cut):
        self.qubit1 = scq.Fluxonium(
            EJ = self.para.get("EJ1_GHz", 5),
            EC = self.para.get("EC1_GHz", 0.2),
            EL = self.para.get("EL1_GHz", 0.5),
            flux = self.para.get("flux1", 0.5),
            cutoff = cut,
            truncated_dim = int(self.sim_para["qubit_dim1"]),
            id_str = "qubit1",
        )

        _, bare_evecs = self.qubit1.eigensys(int(self.sim_para["qubit_dim1"]))

        return bare_evecs

    def _qubit2_init(self, cut):
        self.qubit2 = scq.Fluxonium(
            EJ = self.para.get("EJ2_GHz", 5),
            EC = self.para.get("EC2_GHz", 0.2),
            EL = self.para.get("EL2_GHz", 0.5),
            flux = self.para.get("flux2", 0.5),
            cutoff = cut,
            truncated_dim = int(self.sim_para["qubit_dim2"]),
            id_str = "qubit2",
        )

        _, bare_evecs = self.qubit2.eigensys(int(self.sim_para["qubit_dim2"]))

        return bare_evecs
    
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

        Parameters
        ----------
        sim_para : Dict[str, Any]
            A dictionary containing simulation parameters.
            sim_para should contain the following keys:
                "res_dim": int
                    The dimension of the resonator Hilbert space.
                "qubit_dim1": int
                    The dimension of the qubit1 Hilbert space.
                "qubit_dim2": int
                    The dimension of the qubit2 Hilbert space.
                "qubit_cutoff1": int
                    The cutoff dimension of the qubit1 Hilbert space.
                "qubit_cutoff2": int
                    The cutoff dimension of the qubit2 Hilbert space.
        para : Dict[str, Any], optional
            A dictionary containing system parameters. The default is {}.
            para should contain the following keys:
                "E_osc_GHz": float
                    The frequency of the resonator in GHz.
                "EJ1_GHz": float
                    The Josephson energy of the fluxonium qubit1 in GHz.
                "EC1_GHz": float
                    The charging energy of the fluxonium qubit1 in GHz.
                "EL1_GHz": float
                    The inductive energy of the fluxonium qubit1 in GHz.
                "flux1": float
                    The flux bias of the fluxonium qubit1.
                "EJ2_GHz": float
                    The Josephson energy of the fluxonium qubit2 in GHz.
                "EC2_GHz": float
                    The charging energy of the fluxonium qubit2 in GHz.
                "EL2_GHz": float
                    The inductive energy of the fluxonium qubit2 in GHz.
                "flux2": float
                    The flux bias of the fluxonium qubit2.
                "g1_GHz": float
                    The coupling strength between the resonator and the qubit1 in GHz.
                "g2_GHz": float
                    The coupling strength between the resonator and the qubit2 in GHz.
        convergence_range : Tuple[float, float] | None, optional
            The convergence range for the eigensolver. The default is (1e-10, 1e-6).
        update_cutoff : bool, optional
            Whether to update the qubit cutoff dimension in sim_para. The default is True.
        """
        super().__init__(sim_para, para)

        # HilbertSpace
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        self.qubit_init_w_conv_check(
            self._qubit1_init, 
            sim_para["qubit_cutoff1"], 
            convergence_range, 
            update_cutoff
        )
        if update_cutoff:
            sim_para["qubit_cutoff1"] = self.qubit1.cutoff

        self.qubit_init_w_conv_check(
            self._qubit2_init, 
            sim_para["qubit_cutoff2"], 
            convergence_range, 
            update_cutoff
        )
        if update_cutoff:
            sim_para["qubit_cutoff2"] = self.qubit2.cutoff

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
            "E_osc_GHz": [self.res],
            "EJ1_GHz": [self.qubit1],
            "EC1_GHz": [self.qubit1],
            "EL1_GHz": [self.qubit1],
            "flux1": [self.qubit1],
            "EJ2_GHz": [self.qubit2],
            "EC2_GHz": [self.qubit2],
            "EL2_GHz": [self.qubit2],
            "flux2": [self.qubit2],
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
        **kwargs
    ):
        kwargs = copy.deepcopy(kwargs)
        self.update_hilbertspace(
            ps.hilbertspace, 
            E_osc_GHz = kwargs.pop("E_osc_GHz", self.res.E_osc),
            EJ1_GHz = kwargs.pop("EJ1_GHz", self.qubit1.EJ),
            EC1_GHz = kwargs.pop("EC1_GHz", self.qubit1.EC),
            EL1_GHz = kwargs.pop("EL1_GHz", self.qubit1.EL),
            flux1 = kwargs.pop("flux1", self.qubit1.flux),
            EJ2_GHz = kwargs.pop("EJ2_GHz", self.qubit2.EJ),
            EC2_GHz = kwargs.pop("EC2_GHz", self.qubit2.EC),
            EL2_GHz = kwargs.pop("EL2_GHz", self.qubit2.EL),
            flux2 = kwargs.pop("flux2", self.qubit2.flux),
            g1_GHz = kwargs.pop("g1_GHz", self.hilbertspace.interaction_list[0].g_strength),
            g2_GHz = kwargs.pop("g2_GHz", self.hilbertspace.interaction_list[1].g_strength),
        )


    @property
    def dict(self) -> Dict[str, Any]:
        """
        Return a dictionary containing the system information.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the following keys:
                "sim_para": Dict[str, Any]
                    A dictionary containing simulation parameters.
                "hilbertspace": HilbertSpace
                    The Hilbert space of the system.
                "qubit1": scq.Fluxonium
                    The fluxonium qubit1 subsystem.
                "res": scq.Oscillator
                    The resonator subsystem.
                "qubit2": scq.Fluxonium
                    The fluxonium qubit2 subsystem.
                "update_hilbertspace": Callable
                    A function for updating the Hilbert space of the system.
                "update_hilbertspace_by_keyword": Callable
                    A function for updating the Hilbert space of the system using keyword arguments.
                "subsys_update_info": Dict[str, List[Subsystem]]
                    A dictionary containing information about the subsystems that can be updated.
        """
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

