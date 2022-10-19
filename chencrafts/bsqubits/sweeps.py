import numpy as np
import scqubits as scq
from scipy.constants import h, k

from collections import OrderedDict
from typing import Union

PI2 = np.pi * 2

class DerivedVariableBase():
    def __init__(
        self,
        para_dict: dict, 
        sim_para: dict,
        swept_para_dict: dict = {},
    ):
        self.para_dict = para_dict
        self.sim_para = sim_para
        
        self.full_sweep_dict = swept_para_dict
        if self.full_sweep_dict != {}:
            # meshgrid and calculate self.para_dict_to_use
            raise ValueError("Currently it doesn't support sweeping")
            self.para_dict_to_use = _

        else:
            self.para_dict_to_use = self.para_dict.copy()


        self._initialize_scq_sweep_dict([])

        self.derived_dict = OrderedDict({})

    def __getitem__(
        self,
        name: str,
    ):
        try:
            return self.para_dict_to_use[name]
        except KeyError:
            pass

        try:
            return self.derived_dict[name]
        except KeyError:
            raise KeyError(f"{name} not found in the parameters including the derived one. "
            "If you didn't call use `evaluate()`, try it.")

    def _initialize_scq_sweep_dict(self, available_scq_sweep_name):
        """available_scq_sweep_name can be ["omega_s", "g_sa", "EJ", "EC"], for example"""
        self.scq_sweep_dict = OrderedDict(
            [(key, val) for key, val in self.full_sweep_dict 
                if key in available_scq_sweep_name]
        )
        self.other_sweep_dict = OrderedDict(
            [(key, val) for key, val in self.full_sweep_dict 
                if key not in available_scq_sweep_name]
        )

    def _meshgrid(self):
        pass

    def _sweep2float(self, sweep_data, idx=None):
        if idx is None:
            return sweep_data.reshape(-1)[0]
        else:
            return sweep_data.reshape(-1)[1]

    def _sweep2grid(self, sweep_data, idx=None):
        if idx is None:
            return self._add_dimensions(
                sweep_data,
                self.full_sweep_dict,
                self.scq_sweep_dict
            )
        else:
            return self._add_dimensions(
                sweep_data[..., idx],
                self.full_sweep_dict,
                self.scq_sweep_dict
            )

    def _add_dimensions(self, sweep_data, var_list_dict, sweep_name_list):
        target_name_list = list(var_list_dict.keys())
        size_list = [len(lst) for lst in var_list_dict.values()]

        new_mesh = np.array(sweep_data).copy()
        new_shape = np.array(list(sweep_data.shape), dtype=int)
        for idx, var_name in enumerate(target_name_list):
            if var_name not in sweep_name_list:
                new_shape = np.insert(new_shape, idx, 1)    # insert a "1" to reshape data
                new_mesh = new_mesh.reshape(new_shape)
                new_mesh = np.repeat(new_mesh, size_list[idx], axis=idx)
                new_shape[idx] = size_list[idx]

        return new_mesh

    def _n_th(self, omega, temp):
        """omega is in the unit of GHz"""
        return 1 / (np.exp(omega * h * 1e9 / temp / k) - 1)

class DerivedVariableTmon(DerivedVariableBase):
    def __init__(
        self, 
        para_dict: OrderedDict, 
        sim_para: OrderedDict, 
        swept_para_dict: dict = {}
    ):
        super().__init__(
            para_dict, 
            sim_para, 
            swept_para_dict
        )
        
        self._initialize_scq_sweep_dict(["omega_s", "g_sa", "EJ", "EC"])

    def evaluate(
        self,
        convergence_range = (1e-8, 1e-4),
        update_ncut = True,
        return_full_para = True,
    ):

        if self.scq_sweep_dict == {}:
            sweep = single_sweep_tmon(
                self.para_dict,
                self.sim_para,
                convergence_range = convergence_range,
                update_ncut = update_ncut,
            )
            sweep_warpper = self._sweep2float
        else:
            # full sweep
            sweep_warpper = self._sweep2grid
            raise ValueError("Currently it doesn't support sweeping")

        self.derived_dict.update({
            "chi_sa": PI2 * sweep_warpper(
                sweep["chi"]["subsys1": 0, "subsys2": 1], 
                idx=1
            ), 
            "K_s": PI2 * sweep_warpper(
                sweep["kerr"]["subsys1": 0, "subsys2": 0], 
            ), 
            "chi_prime": PI2 * sweep_warpper(
                sweep["chi_prime"]["subsys1": 0, "subsys2": 1], 
                idx=1
            ), 
            "Gamma_up": sweep_warpper(sweep["gamma_up"]), 
            "Gamma_down": sweep_warpper(sweep["gamma_down"]), 
            "Gamma_phi": sweep_warpper(sweep["gamma_phi"]), 
            "Gamma_up_ro": sweep_warpper(sweep["gamma_up"]), 
            "Gamma_down_ro": sweep_warpper(sweep["gamma_down"]), 
            "min_detuning": PI2 * sweep_warpper(sweep["min_detuning"]),
        })

        self.derived_dict.update({
            "n_th": self._n_th(self["omega_s"], self["temp_s"]), 
            "kappa_s": PI2 * self["omega_s"] / self["Q_s"]
                + self["Gamma_down"] * (PI2 * self["g_sa"] / self["min_detuning"])**2, 
            "T_M": self["T_W"] + self["tau_FD"] + self["tau_m"] + np.pi / np.abs(self["chi_sa"])
                + 12 * self["sigma"], 
        })

        if not return_full_para:
            return self.derived_dict
        else:
            full_dict = self.para_dict.copy()
            full_dict.update(self.derived_dict)
            return full_dict

# ##############################################################################
def sweep_for_params(
    sweep: scq.HilbertSpace, 
    ancilla: Union[scq.Transmon, scq.Fluxonium], 
    para_dict,
):
    Temp = para_dict["temp_a"]
    Q_tphi_coef = para_dict["Q_tphi_coef"]
    Q_t1_coef = para_dict["Q_t1_coef"]

    if isinstance(ancilla, scq.Transmon):

        def check_qubit_convergence(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evecs = np.array(paramsweep["bare_evecs"]["subsys": 1][paramindex_tuple])
            return np.max(np.abs(bare_evecs[-1][-3:]))

        def sweep_gamma_up(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
            bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
            return ancilla.t1_capacitive(
                i=0, 
                j=1, 
                get_rate=True, 
                total=False, 
                T=Temp, 
                esys=(bare_evals, bare_evecs), 
            ) / Q_t1_coef

        def sweep_gamma_down(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
            bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
            return ancilla.t1_capacitive(
                i=1, 
                j=0, 
                get_rate=True, 
                total=False, 
                T=Temp, 
                esys=(bare_evals, bare_evecs),
            ) / Q_t1_coef

        def sweep_gamma_phi(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
            bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
            return (
                ancilla.tphi_1_over_f_ng(
                    i=0, 
                    j=1, 
                    get_rate=True, 
                    esys=(bare_evals, bare_evecs)
                ) + ancilla.tphi_1_over_f_cc(
                    i=0, 
                    j=1, 
                    get_rate=True, 
                    esys=(bare_evals, bare_evecs)
                )
            ) / Q_tphi_coef

        # def sweep_gamma_ng(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
        #     bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
        #     bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
        #     return (
        #         ancilla.tphi_1_over_f_ng(
        #             i=0, 
        #             j=1, 
        #             get_rate=True, 
        #             esys=(bare_evals, bare_evecs)
        #         )
        #     ) / Q_a_coef

        # def sweep_gamma_cc(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
        #     bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
        #     bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
        #     return (
        #         ancilla.tphi_1_over_f_cc(
        #             i=0, 
        #             j=1, 
        #             get_rate=True, 
        #             esys=(bare_evals, bare_evecs)
        #         )
        #     ) / Q_a_coef

        def sweep_drs_coef(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            """
            The overlap between dressed states
            """
            return

        def sweep_min_detuning(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evals0 = paramsweep["bare_evals"]["subsys":0][paramindex_tuple]
            bare_evals1 = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

            sys_freq = bare_evals0[1] - bare_evals0[0]

            anc_freq_0x = bare_evals1[1:3] - bare_evals1[0]
            # anc_freq_1x = bare_evals1[2:3] - bare_evals1[1]
            # anc_freq_2x = bare_evals1[3:4] - bare_evals1[2]

            return np.min(np.abs(anc_freq_0x - sys_freq))

        def sweep_detuning(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evals0 = paramsweep["bare_evals"]["subsys":0][paramindex_tuple]
            bare_evals1 = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

            sys_freq = bare_evals0[1] - bare_evals0[0]

            anc_freq_0x = bare_evals1[1:3] - bare_evals1[0]
            # anc_freq_1x = bare_evals1[2:3] - bare_evals1[1]
            # anc_freq_2x = bare_evals1[3:4] - bare_evals1[2]

            return np.min(np.abs(anc_freq_0x - sys_freq))
            
        sweep.add_sweep(sweep_gamma_up, "gamma_up")
        sweep.add_sweep(sweep_gamma_down, "gamma_down")
        sweep.add_sweep(sweep_gamma_phi, "gamma_phi")
        # sweep.add_sweep(sweep_gamma_cc, "gamma_phi_cc")
        # sweep.add_sweep(sweep_gamma_ng, "gamma_phi_ng")
        sweep.add_sweep(sweep_min_detuning, "min_detuning")
        sweep.add_sweep(check_qubit_convergence, "convergence")

    elif isinstance(ancilla, scq.Fluxonium):

        def check_qubit_convergence(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evecs = np.array(paramsweep["bare_evecs"]["subsys": 1][paramindex_tuple])
            return np.max(np.abs(bare_evecs[-1][-3:]))

        def sweep_gamma_up(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
            bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
            return (
                ancilla.t1_capacitive(
                    i = 0, 
                    j = 1, 
                    get_rate = True, 
                    total = False, 
                    T = Temp, 
                    esys = (bare_evals, bare_evecs), 
                ) + ancilla.t1_inductive(
                    i=0, 
                    j=1, 
                    get_rate=True, 
                    total=False, 
                    T=Temp, 
                    esys=(bare_evals, bare_evecs), 
                # ) + ancilla.t1_charge_impedance(
                #     i = 0,
                #     j = 1,
                #     get_rate=True, 
                #     total = False,
                #     T=Temp, 
                #     esys=(bare_evals, bare_evecs), 
                ) + ancilla.t1_flux_bias_line(
                    i = 0,
                    j = 1,
                    get_rate=True, 
                    total = False,
                    T=Temp, 
                    esys=(bare_evals, bare_evecs), 
                ) + ancilla.t1_quasiparticle_tunneling(
                    i = 0,
                    j = 1,
                    get_rate=True, 
                    total = False,
                    T=Temp, 
                    esys=(bare_evals, bare_evecs), 
                )
            ) / Q_t1_coef

        def sweep_gamma_down(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
            bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
            return (
                ancilla.t1_capacitive(
                    i = 1, 
                    j = 0, 
                    get_rate = True, 
                    total = False, 
                    T = Temp, 
                    esys = (bare_evals, bare_evecs), 
                ) + ancilla.t1_inductive(
                    i = 1, 
                    j = 0, 
                    get_rate = True, 
                    total = False, 
                    T = Temp, 
                    esys=(bare_evals, bare_evecs), 
                # ) + ancilla.t1_charge_impedance(
                #     i = 1,
                #     j = 0,
                #     get_rate = True, 
                #     total = False,
                #     T=Temp, 
                #     esys=(bare_evals, bare_evecs), 
                ) + ancilla.t1_flux_bias_line(
                    i = 1,
                    j = 0,
                    get_rate=True, 
                    total = False,
                    T=Temp, 
                    esys=(bare_evals, bare_evecs), 
                ) + ancilla.t1_quasiparticle_tunneling(
                    i = 1,
                    j = 0,
                    get_rate=True, 
                    total = False,
                    T=Temp, 
                    esys=(bare_evals, bare_evecs), 
                )
            ) / Q_t1_coef

        def sweep_gamma_phi(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
            bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
            return (
                ancilla.tphi_1_over_f_flux(
                    i=0, 
                    j=1, 
                    get_rate=True, 
                    esys=(bare_evals, bare_evecs)
                ) + ancilla.tphi_1_over_f_cc(
                    i=0, 
                    j=1, 
                    get_rate=True, 
                    esys=(bare_evals, bare_evecs)
                )
            ) / Q_tphi_coef

        def sweep_min_detuning(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evals0 = paramsweep["bare_evals"]["subsys":0][paramindex_tuple]
            bare_evals1 = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

            sys_freq = bare_evals0[1] - bare_evals0[0]

            anc_freq_0x = bare_evals1[1:3] - bare_evals1[0]
            anc_freq_1x = bare_evals1[2:3] - bare_evals1[1]
            # anc_freq_2x = bare_evals1[3:4] - bare_evals1[2]

            return np.min(np.abs(np.concatenate([
                anc_freq_0x - sys_freq,
                anc_freq_1x - sys_freq,
                # anc_freq_2x - sys_freq,
            ], axis=0)))

        sweep.add_sweep(sweep_gamma_up, "gamma_up")
        sweep.add_sweep(sweep_gamma_down, "gamma_down")
        sweep.add_sweep(sweep_gamma_phi, "gamma_phi")
        sweep.add_sweep(sweep_min_detuning, "min_detuning")
        sweep.add_sweep(check_qubit_convergence, "convergence")

# ################################ Transmon ####################################
def single_sweep_tmon(
    para,
    sim_para,
    convergence_range = (1e-8, 1e-4),
    update_ncut = True,
):
    omega_s, EJ, EC, ng, g_sa = [para[key] 
        for key in ["omega_s", "EJ", "EC", "ng", "g_sa"]]
    sys_dim, anc_ncut, anc_dim, eval_count, num_cpus = [sim_para[key] 
        for key in ["sys_dim", "anc_ncut", "anc_dim", "sweep_eval_count", "num_cpus"]]

    # build system and ancilla
    system = scq.Oscillator(
        E_osc = omega_s,
        truncated_dim = sys_dim,
        id_str = "system",
        l_osc = 1
    )

    while True:
        ancilla = scq.Transmon(
            EJ = EJ,
            EC = EC,
            ng = ng,
            ncut = anc_ncut,
            truncated_dim = anc_dim,
            id_str = "ancilla" 
        )

        _, bare_evecs = ancilla.eigensys(anc_dim)
        conv = np.max(np.abs(bare_evecs[-1][-3:]))

        if conv > convergence_range[1]:
            anc_ncut = int(anc_ncut * 1.5)
        elif conv < convergence_range[0]:
            anc_ncut = int(anc_ncut / 1.5)
            break
        else:
            break

    subsystem_list = [system, ancilla]
    h_space = scq.HilbertSpace(subsystem_list)

    h_space.add_interaction(
        g = g_sa,
        op1 = system.n_operator,
        op2 = ancilla.n_operator,
        add_hc = False,
        id_str = "sys-anc"
    )

    # "sweep" for variables 
    paramvals_by_name = {
        "omega_s": [omega_s],
        "g_sa": [g_sa],
        "EJ": [EJ],
        "EC": [EC]
    }
    subsys_update_info =  {
        "omega_s": [system],
        "g_sa": [],
        "EJ": [ancilla],
        "EC": [ancilla]
    }
    def update_h_space(omega_s, g_sa, EJ, EC):
        system.E_osc = omega_s
        h_space.interaction_list[0].g_strength = g_sa
        ancilla.EJ = EJ
        ancilla.EC = EC

    sweep = scq.ParameterSweep(
        hilbertspace = h_space,
        paramvals_by_name = paramvals_by_name,
        update_hilbertspace = update_h_space,
        evals_count = eval_count,
        subsys_update_info = subsys_update_info,
        num_cpus = num_cpus
    )

    sweep_for_params(sweep, ancilla, para)

    if update_ncut:
        sim_para["anc_ncut"] = anc_ncut

    return sweep

def sweep_tmon(
    para,
    swept_para,
    sim_para,
):
    """
    Only support sweeping with four list: omega_s, EJ, EC, g_sa
    """

    ng = para["ng"]
    omega_s_list, EJ_list, EC_list, g_sa_list = [swept_para[key]
        for key in ["omega_s", "EJ", "EC", "g_sa"]]
    sys_dim, anc_ncut, anc_dim, eval_count, num_cpus = [sim_para[key] 
        for key in ["sys_dim", "anc_ncut", "anc_dim", "sweep_eval_count", "num_cpus"]]

    system = scq.Oscillator(
        E_osc = omega_s_list[0],
        truncated_dim = sys_dim,
        id_str = "system",
        l_osc = 1
    )

    ancilla = scq.Transmon(
        EJ = EJ_list[0],
        EC = EC_list[0],
        ng = ng,
        ncut = anc_ncut,
        truncated_dim = anc_dim,
        id_str = "ancilla" 
    )

    subsystem_list = [system, ancilla]
    h_space = scq.HilbertSpace(subsystem_list)

    h_space.add_interaction(
        g = g_sa_list[0],
        op1 = system.n_operator,
        op2 = ancilla.n_operator,
        add_hc = False,
        id_str = "sys-anc"
    )

    subsys_update_info =  {
        "omega_s": [system],
        "g_sa": [],
        "EJ": [ancilla],
        "EC": [ancilla]
    }
    def update_h_space(omega_s, g_sa, EJ, EC):
        system.E_osc = omega_s
        h_space.interaction_list[0].g_strength = g_sa
        ancilla.EJ = EJ
        ancilla.EC = EC

    sweep = scq.ParameterSweep(
        hilbertspace = h_space,
        paramvals_by_name = swept_para,
        update_hilbertspace = update_h_space,
        evals_count = eval_count,
        subsys_update_info = subsys_update_info,
        num_cpus = num_cpus
    )

    sweep_for_params(sweep, ancilla, para)

    return sweep

# ################################ Fluxonium ###################################
def single_sweep_fl(
    omega_s, EJ, EC, EL, flux, g_sa,
    sys_dim, cutoff, anc_dim, eval_count, cpu_num,
    Temp=None, Q_a_coef=None, convergence_range=(1e-8, 1e-4),
):
    system = scq.Oscillator(
        E_osc = omega_s,
        truncated_dim = sys_dim,
        id_str = "system",
        l_osc = 1
    )

    while True:
        ancilla = scq.Fluxonium(
            EJ = EJ,
            EC = EC,
            EL = EL,
            flux = flux,
            cutoff=cutoff,
            truncated_dim = anc_dim,
            id_str = "ancilla" 
        )

        _, bare_evecs = ancilla.eigensys(anc_dim)
        conv = np.max(np.abs(bare_evecs[-1][-3:]))

        if conv > convergence_range[1]:
            cutoff = int(cutoff * 1.5)
        elif conv < convergence_range[0]:
            cutoff = int(cutoff / 1.5)
            break
        else:
            break

    subsystem_list = [system, ancilla]
    h_space = scq.HilbertSpace(subsystem_list)

    h_space.add_interaction(
        g = g_sa,
        op1 = system.n_operator,
        op2 = ancilla.n_operator,
        add_hc = False,
        id_str = "sys-anc"
    )

    paramvals_by_name = {
        "omega_s": [omega_s],
        "g_sa": [g_sa],
        "EJ": [EJ],
        "EC": [EC],
        "EL": [EL],
        "flux": [flux],
    }
    subsys_update_info =  {
        "omega_s": [system],
        "g_sa": [],
        "EJ": [ancilla],
        "EC": [ancilla],
        "EL": [ancilla],
        "flux": [ancilla],
    }
    def update_h_space(omega_s, g_sa, EJ, EC, EL, flux):
        system.E_osc = omega_s
        h_space.interaction_list[0].g_strength = g_sa
        ancilla.EJ = EJ
        ancilla.EC = EC
        ancilla.EL = EL
        ancilla.flux = flux

    sweep = scq.ParameterSweep(
        hilbertspace=h_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_h_space,
        evals_count=eval_count,
        subsys_update_info=subsys_update_info,
        num_cpus=cpu_num
    )

    sweep_for_params(sweep, ancilla, Q_a_coef, Temp)

    return sweep, cutoff

def sweep_fl(
    omega_s_list, EJ_list, EC_list, EL_list, flux_list, g_sa_list,
    sys_dim, cutoff, anc_dim, eval_count, cpu_num,
    Temp=None, Q_a_coef=None
):
    system = scq.Oscillator(
        E_osc = omega_s_list[0],
        truncated_dim = sys_dim,
        id_str = "system",
        l_osc = 1
    )

    ancilla = scq.Fluxonium(
        EJ = EJ_list[0],
        EC = EC_list[0],
        EL = EL_list[0],
        flux = flux_list[0],
        cutoff=cutoff,
        truncated_dim = anc_dim,
        id_str = "ancilla" 
    )

    subsystem_list = [system, ancilla]
    h_space = scq.HilbertSpace(subsystem_list)

    h_space.add_interaction(
        g = g_sa_list[0],
        op1 = system.n_operator,
        op2 = ancilla.n_operator,
        add_hc = False,
        id_str = "sys-anc"
    )

    paramvals_by_name = {
        "omega_s": omega_s_list,
        "g_sa": g_sa_list,
        "EJ": EJ_list,
        "EC": EC_list,
        "EL": EL_list,
        "flux": flux_list,
    }
    subsys_update_info =  {
        "omega_s": [system],
        "g_sa": [],
        "EJ": [ancilla],
        "EC": [ancilla],
        "EL": [ancilla],
        "flux": [ancilla],
    }
    def update_h_space(omega_s, g_sa, EJ, EC, EL, flux):
        system.E_osc = omega_s
        h_space.interaction_list[0].g_strength = g_sa
        ancilla.EJ = EJ
        ancilla.EC = EC
        ancilla.EL = EL
        ancilla.flux = flux

    sweep = scq.ParameterSweep(
        hilbertspace=h_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_h_space,
        evals_count=eval_count,
        subsys_update_info=subsys_update_info,
        num_cpus=cpu_num
    )

    sweep_for_params(sweep, ancilla, Q_a_coef, Temp)

    return sweep