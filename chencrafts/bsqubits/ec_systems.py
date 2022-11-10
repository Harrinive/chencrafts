from turtle import update
import scqubits as scq
from scqubits.core.qubit_base import QubitBaseClass1d as Qubit
import numpy as np
import qutip as qt

from typing import Dict, List, Tuple

from collections import OrderedDict

from chencrafts.bsqubits.basis_n_states import dressed_basis

class JointSystemBase():
    base_para_name = []
    sweep_available_name = []

    def __init__(
        self,
        para: Dict[str, float],
        sim_para: Dict[str, float],
        swept_para: Dict[str, List[float]] = {},
        return_array = False
    ):
        self.para = para
        self.sim_para = sim_para
        self.swept_para = swept_para

        self.return_array = return_array

        self.base_para_value: List[float] 
        self.subsys: List[Qubit]
        self.h_space: scq.HilbertSpace

    def _base_para_initialize(self, para):
        self.base_para_value = []
        for name in self.base_para_name:
            try: 
                self.base_para_value.append(para[name])
            except KeyError:
                if name not in self.sweep_available_name:
                    raise KeyError(f"Parameter value {name} not found in para, "
                    "which is supposed to be fixed (not avaialbe to sweep)")
                if name not in self.swept_para:
                    raise KeyError(f"Parameter value {name} not found")
                self.base_para_value.append(self.swept_para[name][0])

    def _qobj2array(self, qobj: qt.Qobj) -> np.ndarray:
        return qobj.data.toarray()

    def _qobj_wrapper(self, qobj: qt.Qobj) -> np.ndarray | qt.Qobj:
        if self.return_array:
            return self._qobj2array(qobj)
        else:
            return qobj

    def bare_oprt(
        self, 
        oprt: np.ndarray | str | qt.Qobj, 
        sys: Qubit, 
        bare_esys: Tuple[np.ndarray, np.ndarray] = None
    ) -> qt.Qobj:
        if isinstance(oprt, str): 
            op_func = getattr(sys, oprt)
            oprt = op_func()

        if bare_esys is not None:
            evecs = bare_esys[1]
        else:
            evecs = None
        
        return scq.identity_wrap(oprt, sys, self.subsys, evecs=evecs)

    def _scq_sweep_input_para(self) -> OrderedDict:
        """
        For multi-dimensional sweep, the sweep function need all value in the swept_para
        to be a list
        """
        swept_para = OrderedDict({})
        for var_name in self.sweep_available_name:
            if var_name in self.swept_para.keys():
                swept_para[var_name] = self.swept_para[var_name]
            else:
                swept_para[var_name] = [self.para[var_name]]

        return swept_para

    @property
    def dim_list(self) -> Tuple[int]:
        return tuple(sys.truncated_dim for sys in self.subsys)
    
    @property
    def dim(self) -> int:
        return np.prod(self.dim_list)

    def bare_esys(self):
        esys_dict = {}
        for idx, sys in enumerate(self.subsys):
            esys_dict[idx] = sys.eigensys(evals_count=self.subsys[idx].truncated_dim)

        return esys_dict

    def eigensys(
        self, 
        bare_esys_dict: Dict[int, np.ndarray | List[np.ndarray]] = None
    ):
        return self.h_space.eigensys(bare_esys=bare_esys_dict)

    def drs_basis(self, esys=None):        
        return dressed_basis(
            self.h_space,
            self.dim_list,
            esys
        )


class CavityAncSystem(JointSystemBase):
    def __init__(
        self, para: Dict[str, float], 
        sim_para: Dict[str, float], 
        swept_para: Dict[str, List[float]] = {}, 
        return_array=False
    ):
        super().__init__(
            para, 
            sim_para, 
            swept_para, 
            return_array
        )

        self.system: Qubit
        self.ancilla: Qubit
        self.subsys: List[Qubit]

    @property
    def hamiltonian(self) -> np.ndarray | qt.Qobj:
        return self._qobj_wrapper(self.h_space.hamiltonian())

    def a_s(
        self,
        sys_bare_esys: Tuple[np.ndarray, np.ndarray] = None,
    ) -> np.ndarray | qt.Qobj:
        """
        Annihilation operator for the system
        """
        a = self.bare_oprt("annihilation_operator", self.system, sys_bare_esys)
        return self._qobj_wrapper(a)

    def proj_a(self, ket_fock_num, bra_fock_num) -> np.ndarray | qt.Qobj:
        """
        projector for the ancilla from one eigen-basis to another,  
        it doesn't need to know what the eigenstate in bare basis
        """
        proj = qt.projection(
            self.ancilla.truncated_dim, 
            ket_fock_num,
            bra_fock_num,
        )
        return qt.tensor(
            qt.qeye(self.system.truncated_dim),
            proj
        )


class CavityTmonSys(CavityAncSystem):
    base_para_name = ["omega_s", "EJ", "EC", "ng", "g_sa"]
    sweep_available_name = ["omega_s", "g_sa", "EJ", "EC"]

    def __init__(
        self,
        para: Dict[str, float],
        sim_para: Dict[str, float],
        swept_para: Dict[str, List[float]] = {},
        h_space: scq.HilbertSpace = None,
        convergence_range: Tuple[float] = (1e-10, 1e-6),
        update_ncut: bool = True,
        return_array = False
    ):
        """
        Initialize from h_space is dangerous, currently I dont have a good idea of
        how to do this. Sweeping is not allowed in this case. 
        """
        super().__init__(
            para,
            sim_para,
            swept_para,
            return_array,
        )

        if h_space is None:
            self._base_para_initialize(self.para)
            self._h_space_initalize(sim_para, convergence_range, update_ncut)
        else:
            self.h_space = h_space
            self.subsys = h_space.subsys_list
            self.system = self.subsys[0]
            self.ancilla = self.subsys[1]

    def _h_space_initalize(
        self, 
        sim_para: Dict[str, float], 
        convergence_range: Tuple[float], 
        update_ncut: bool
    ):
        omega_s, EJ, EC, ng, g_sa = self.base_para_value
        sys_dim, anc_ncut, anc_dim = [self.sim_para[key] 
            for key in ["sys_dim", "anc_ncut", "anc_dim"]]

        # build system and ancilla
        self.system = scq.Oscillator(
            E_osc = omega_s,
            truncated_dim = sys_dim,
            id_str = "system",
            l_osc = 1
        )

        while True:
            self.ancilla = scq.Transmon(
                EJ = EJ,
                EC = EC,
                ng = ng,
                ncut = anc_ncut,
                truncated_dim = anc_dim,
                id_str = "ancilla" 
            )

            _, bare_evecs = self.ancilla.eigensys(anc_dim)
            conv = np.max(np.abs(bare_evecs[-1][-3:]))

            if convergence_range is None:
                break
            elif conv > convergence_range[1]:
                anc_ncut = int(anc_ncut * 1.5)
            elif conv < convergence_range[0]:
                anc_ncut = int(anc_ncut / 1.5)
                break
            else:
                break

        if update_ncut:
            sim_para["anc_ncut"] = anc_ncut

        self.subsys = [self.system, self.ancilla]
        self.h_space = scq.HilbertSpace(self.subsys)

        self.h_space.add_interaction(
            g = g_sa,
            op1 = self.system.n_operator,
            op2 = self.ancilla.n_operator,
            add_hc = False,
            id_str = "sys-anc"
        )
    
    @property
    def controls(
        self, 
        anc_bare_esys: Tuple[np.ndarray, np.ndarray] = None,
    ) -> List[np.ndarray | qt.Qobj]:
        n_anc = self.bare_oprt("n_operator", self.ancilla, anc_bare_esys)
        oprts = [n_anc]
        return [self._qobj_wrapper(op) for op in oprts]

    @property
    def a_a(
        self,
        anc_esys: Tuple[np.ndarray, np.ndarray] = None,
    ) -> np.ndarray | qt.Qobj:
        """
        Annihilation operator for the ancilla
        Currently it is generated by adding the off diagonal charge matrix element 
        to a zero matrix, and then normalize the first matrix element to 1
        """
        n_anc_data = self.ancilla.n_operator()

        # transform to energy eigen basis
        if anc_esys is None:
            _, evecs = self.ancilla.eigensys(evals_count=self.ancilla.truncated_dim)
        else:
            _, evecs = anc_esys
        n_anc_data = evecs.T @ n_anc_data @ evecs

        mat_elems = n_anc_data.diagonal(1)
        mat_elems = mat_elems / mat_elems[0]
        
        a = qt.qdiags(mat_elems, 1)
        a = qt.tensor(qt.identity(self.system.truncated_dim), a)
        return self._qobj_wrapper(a)

    def detuning(self):
        """
        omega_s - omega_a
        """
        evals, _ = self.ancilla.eigensys(evals_count=2)

        return self.para["omega_s"] - (evals[1] - evals[0])

    def _update_h_space(self, omega_s, g_sa, EJ, EC, **kwargs):
            self.system.E_osc = omega_s
            self.h_space.interaction_list[0].g_strength = g_sa
            self.ancilla.EJ = EJ
            self.ancilla.EC = EC

    def sweep(self,) -> scq.ParameterSweep:
        # "sweep" for variables on omega_s, g_sa, EJ and EC

        if self.sim_para["sweep_eval_count"] >= np.prod(self.dim_list):
            raise ValueError(f"""
                Too much sweep_eval_count 
                ({self.sim_para["sweep_eval_count"]:d}) for 
                a {np.prod(self.dim_list):d}-dimensional system
            """)

        paramvals_by_name = self._scq_sweep_input_para()

        subsys_update_info =  {
            "omega_s": [self.system],
            "g_sa": [],
            "EJ": [self.ancilla],
            "EC": [self.ancilla]
        }

        update_h_space = lambda *args: self._update_h_space(*args)

        sweep = scq.ParameterSweep(
            hilbertspace = self.h_space,
            paramvals_by_name = paramvals_by_name,
            update_hilbertspace = update_h_space,
            evals_count = self.sim_para["sweep_eval_count"],
            subsys_update_info = subsys_update_info,
            num_cpus = self.sim_para["num_cpus"]
        )

        return sweep