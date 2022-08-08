import numpy as np
import scqubits as scq
from typing import Union


def sweep_for_params(
    sweep: scq.HilbertSpace, 
    ancilla: Union[scq.Transmon, scq.Fluxonium], 
    Q_a_coef, 
    Temp
):

    if Temp is None:
        Temp = 0.015

    if isinstance(ancilla, scq.Transmon):
        if Q_a_coef is None:
            Q_a_coef = 1

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
            ) / Q_a_coef

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
            ) / Q_a_coef

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
            ) * 10 / Q_a_coef

        def sweep_min_detuning(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
            bare_evals0 = paramsweep["bare_evals"]["subsys":0][paramindex_tuple]
            bare_evals1 = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

            sys_freq = bare_evals0[1] - bare_evals0[0]

            anc_freq_0x = bare_evals1[1:2] - bare_evals1[0]
            # anc_freq_1x = bare_evals1[2:3] - bare_evals1[1]
            # anc_freq_2x = bare_evals1[3:4] - bare_evals1[2]

            return np.min(np.abs(anc_freq_0x - sys_freq))
            
        sweep.add_sweep(sweep_gamma_up, "gamma_up")
        sweep.add_sweep(sweep_gamma_down, "gamma_down")
        sweep.add_sweep(sweep_gamma_phi, "gamma_phi")
        sweep.add_sweep(sweep_min_detuning, "min_detuning")
        sweep.add_sweep(check_qubit_convergence, "convergence")

    elif isinstance(ancilla, scq.Fluxonium):

        if Q_a_coef is None:
            Q_a_coef = 1

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
            ) / Q_a_coef

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
            ) / Q_a_coef

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
            ) / Q_a_coef

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
    omega_s, EJ, EC, ng, g_sa,
    sys_dim, anc_ncut, anc_dim, eval_count, cpu_num,
    Temp=None, Q_a_coef=None
):
    system = scq.Oscillator(
        E_osc = omega_s,
        truncated_dim = sys_dim,
        id_str = "system",
        l_osc = 1
    )

    ancilla = scq.Transmon(
        EJ = EJ,
        EC = EC,
        ng = ng,
        ncut = anc_ncut,
        truncated_dim = anc_dim,
        id_str = "ancilla" 
    )

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
        hilbertspace=h_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_h_space,
        evals_count=eval_count,
        subsys_update_info=subsys_update_info,
        num_cpus=cpu_num
    )

    sweep_for_params(sweep, ancilla, Q_a_coef, Temp)

    return sweep

def sweep_tmon(
    omega_s_list, EJ_list, EC_list, g_sa_list, ng,
    sys_dim, anc_ncut, anc_dim, eval_count, cpu_num,
    Temp=None, Q_a_coef=None
):
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

    paramvals_by_name = {
        "omega_s": omega_s_list,
        "g_sa": g_sa_list,
        "EJ": EJ_list,
        "EC": EC_list
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
        hilbertspace=h_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_h_space,
        evals_count=eval_count,
        subsys_update_info=subsys_update_info,
        num_cpus=cpu_num
    )

    sweep_for_params(sweep, ancilla, Q_a_coef, Temp)

    return sweep

# ################################ Fluxonium ###################################
def single_sweep_fl(
    omega_s, EJ, EC, EL, flux, g_sa,
    sys_dim, cutoff, anc_dim, eval_count, cpu_num,
    Temp=None, Q_a_coef=None
):
    system = scq.Oscillator(
        E_osc = omega_s,
        truncated_dim = sys_dim,
        id_str = "system",
        l_osc = 1
    )

    ancilla = scq.Fluxonium(
        EJ = EJ,
        EC = EC,
        EL = EL,
        flux = flux,
        cutoff=cutoff,
        truncated_dim = anc_dim,
        id_str = "ancilla" 
    )

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

    return sweep

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