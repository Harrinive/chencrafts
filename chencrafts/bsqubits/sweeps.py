import numpy as np
import scqubits as scq
import qutip as qt
import matplotlib.pyplot as plt

from typing import Dict, Callable, List, Tuple

from chencrafts.toolbox.data_processing import NSArray

from chencrafts.bsqubits.basis_n_states import cat

PI2 = np.pi * 2

# ##############################################################################
# dictionary key is a str or a tuple: output_name or (output_names)
# dict values is a tuple: (function, input_names)
# the function should return a np.array object
tmon_sweep_dict: Dict[str, Tuple[Callable, Tuple[str]]] = {}

def sweep_loss_rate(
    paramsweep: scq.ParameterSweep, paramindex_tuple, paramvals_tuple, 
    disp, a_dag_a, sig_p_sig_m, **kwargs
):
    sys_dim, anc_dim = paramsweep.hilbertspace.subsystem_dims

    # obtain a set of eigenstates with bare index (n, 0)
    full_drs_idx = paramsweep["dressed_indices"][paramindex_tuple]
    basis = np.ndarray(sys_dim, dtype=qt.Qobj)
    anc_idx = 0
    for idx, bare_idx in enumerate(range(anc_idx, len(full_drs_idx), anc_dim)):
        drs_idx = full_drs_idx[bare_idx]
        if drs_idx is not None:
            basis[idx] = paramsweep["evecs"][paramindex_tuple][drs_idx]
        else:
            basis[idx] = None

    # get a cat state, now uses logical plus state

    alpha = np.sqrt(disp)

    logical_plus = cat(basis, [(1, alpha), (1, -alpha), (1, 1j * alpha), (1, -1j * alpha)])

    # evaluate photon loss rate
    pure_cavity_loss = qt.expect(a_dag_a, logical_plus)
    inverse_purcell = qt.expect(sig_p_sig_m, logical_plus)

    return np.array([pure_cavity_loss, inverse_purcell])
tmon_sweep_dict[("n_bar", "anc_excitation")] = (sweep_loss_rate, ("disp", ))

def sweep_tmon_relaxation(
    paramsweep: scq.ParameterSweep, paramindex_tuple, paramvals_tuple, 
    temp_a, Q_t1_coef, Q_tphi_coef, **kwargs):

    ancilla: scq.Transmon = paramsweep.hilbertspace.subsys_list[1]
    bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
    bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

    default_Q = 5e5

    gamma_up = ancilla.t1_capacitive(
        i=0, 
        j=1, 
        get_rate=True, 
        total=False, 
        T=temp_a, 
        esys=(bare_evals, bare_evecs), 
        Q_cap = default_Q,
    ) / Q_t1_coef

    gamma_down = ancilla.t1_capacitive(
        i=1, 
        j=0, 
        get_rate=True, 
        total=False, 
        T=temp_a, 
        esys=(bare_evals, bare_evecs),
        Q_cap = default_Q,
    ) / Q_t1_coef

    gamma_phi_ng = ancilla.tphi_1_over_f_ng(
        i=0, 
        j=1, 
        get_rate=True, 
        esys=(bare_evals, bare_evecs)
    )/ Q_tphi_coef
        
    gamma_phi_cc = ancilla.tphi_1_over_f_cc(
        i=0, 
        j=1, 
        get_rate=True, 
        esys=(bare_evals, bare_evecs)
    ) / Q_tphi_coef

    return np.array([gamma_down, gamma_up, gamma_phi_ng, gamma_phi_cc])

tmon_sweep_dict[
    ("Gamma_down", "Gamma_up", "Gamma_phi_ng", "Gamma_phi_cc")
    ] = (sweep_tmon_relaxation, ("temp_a", "Q_t1_coef", "Q_tphi_coef"))


# ##############################################################################

def _sweep_fl(
    sweep: scq.ParameterSweep,
    ancilla: scq.Transmon,
    para_dict: Dict[str, float],
):
    Temp = para_dict["temp_a"]
    Q_tphi_coef = para_dict["Q_tphi_coef"]
    Q_t1_coef = para_dict["Q_t1_coef"]

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

