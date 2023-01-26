import numpy as np
import scqubits as scq
import qutip as qt
import matplotlib.pyplot as plt
from scipy.constants import h, k

from typing import Dict, Callable, List, Tuple, Any

from chencrafts.toolbox.data_processing import NSArray
from chencrafts.bsqubits.basis_n_states import cat

PI2 = np.pi * 2

# ##############################################################################
def therm_factor(freq, temp, n_th_base):
    """freq is in the unit of GHz, temp is in the unit of K"""
    assert n_th_base == 0, "currently the code only support n_th_base = 0"
    therm_ratio = freq * h * 1e9 / temp / k
    return (1 / np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))

# ##############################################################################
def sweep_convergence(
    paramsweep: scq.ParameterSweep, paramindex_tuple, paramvals_tuple,
    **kwargs
):
    bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
    return np.max(np.abs(bare_evecs[-3:, :]))

def sweep_loss_rate(
    paramsweep: scq.ParameterSweep, paramindex_tuple, paramvals_tuple, 
    disp, a_dag_a, sig_p_sig_m, **kwargs
):
    sys_dim, anc_dim = paramsweep.hilbertspace.subsystem_dims

    # minimal number of basis
    min_basis_num = int(np.abs(disp)**2 + np.abs(disp))

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
            if idx < min_basis_num:
                return np.nan * np.zeros(4)


    # get a state for calculating decay rate, now uses logical plus state
    logical_plus = cat(basis, [(1, disp), (1, -disp), (1, 1j * disp), (1, -1j * disp)])
    fock_1 = basis[1]

    # evaluate photon loss rate
    try:
        cavity_excitation_l = qt.expect(a_dag_a, logical_plus)
        qubit_excitation_l = qt.expect(sig_p_sig_m, logical_plus)
        cavity_excitation_f = qt.expect(a_dag_a, fock_1)
        qubit_excitation_f = qt.expect(sig_p_sig_m, fock_1)
    except TypeError:
        # debugging
        bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]
        print(
            paramvals_tuple,
            bare_evals[1] - bare_evals[0],
            fock_1,
            len([b for b in basis if b is not None])
        )
        return np.nan * np.zeros(4)

    # # Sanity check 0:
    # print(len(basis))

    # # Sanity check 1: 
    # sys_dim_plot = 4
    # fig, axs = plt.subplots(sys_dim_plot, 2, figsize=(4, sys_dim_plot*2))
    # for idx in range(sys_dim_plot):
    #     qt.plot_fock_distribution(qt.ptrace(basis[idx], 0), ax = axs[idx, 0])
    #     qt.plot_fock_distribution(qt.ptrace(basis[idx], 1), ax = axs[idx, 1])

    # plt.tight_layout()
    # plt.show()

    # # Sanity check 2:
    # qt.plot_wigner_fock_distribution(qt.ptrace(logical_plus, 0))
    # plt.show()

    return np.array([
        cavity_excitation_l, 
        qubit_excitation_l, 
        cavity_excitation_f, 
        qubit_excitation_f
    ])

# ##############################################################################
# dictionary key is a str or a tuple: output_name or (output_names)
# dict values is a tuple: (function, input_names)
# the function should return a np.array object
tmon_sweep_dict: Dict[Any, Tuple[Callable, Tuple[str]]] = {}

tmon_sweep_dict["conv"] = (sweep_convergence, tuple())

tmon_sweep_dict[(
    "n_bar_s", "n_bar_a", "n_fock1_s", "n_fock1_a"
)] = (sweep_loss_rate, ("disp", ))

def sweep_tmon_relaxation(
    paramsweep: scq.ParameterSweep, paramindex_tuple, paramvals_tuple, 
    ancilla_copy: scq.Transmon, temp_a, Q_cap=5e5, A_ng=1e-4, A_cc=1e-7, **kwargs):

    # the calculation of those relaxation rate doesn't consist with our definition 
    # for n_th, but transmon is special - whose n_th is always << 1 in a nornal 
    # experimental setup. So the diviation is small

    # default amplitude: 
    # "A_flux": 1e-6,  # Flux noise strength. Units: Phi_0
    # "A_ng": 1e-4,  # Charge noise strength. Units of charge e
    # "A_cc": 1e-7,  # Critical current noise strength. Units of critical current I_c

    ancilla_copy.EJ = paramvals_tuple[list(paramsweep.param_info.keys()).index("EJ_GHz")]
    ancilla_copy.EC = paramvals_tuple[list(paramsweep.param_info.keys()).index("EC_GHz")]

    bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
    bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

    gamma_down = ancilla_copy.t1_capacitive(
        i=1, 
        j=0, 
        get_rate=True, 
        total=False, 
        T=temp_a, 
        esys=(bare_evals, bare_evecs),
        Q_cap = Q_cap,
    )

    gamma_phi_ng = ancilla_copy.tphi_1_over_f_ng(
        A_noise=A_ng,
        i=0, 
        j=1, 
        get_rate=True, 
        esys=(bare_evals, bare_evecs)
    )

    gamma_phi_cc = ancilla_copy.tphi_1_over_f_cc(
        A_noise=A_cc,
        i=0, 
        j=1, 
        get_rate=True, 
        esys=(bare_evals, bare_evecs)
    )

    return np.array([gamma_down, gamma_phi_ng, gamma_phi_cc])

tmon_sweep_dict[
    ("kappa_a_cap", "kappa_phi_ng", "kappa_phi_cc")
] = (sweep_tmon_relaxation, ("temp_a", "Q_cap", "A_ng", "A_cc"))


# ##############################################################################
# dictionary key is a str or a tuple: output_name or (output_names)
# dict values is a tuple: (function, input_names)
# the function should return a np.array object
flxn_sweep_dict: Dict[Any, Tuple[Callable, Tuple[str]]] = {}

flxn_sweep_dict["conv"] = (sweep_convergence, tuple())

flxn_sweep_dict[(
    "n_bar_s", "n_bar_a", "n_fock1_s", "n_fock1_a"
)] = (sweep_loss_rate, ("disp", ))

def sweep_flxn_depolarization(
    paramsweep: scq.ParameterSweep, paramindex_tuple, paramvals_tuple, 
    starting_level, ancilla_copy: scq.Fluxonium, temp_a=0.015, n_th_base=0.0, 
    Q_cap=5e5, Q_ind=5e8, Z_char=50, Z_fbl=50, A_qsp_tnl=1, n_th_threshold=1e-3, 
    **kwargs
):
    assert n_th_base == 0, "Now the code only support n_th_base = 0."

    param_name = list(paramsweep.param_info.keys())
    ancilla_copy.EJ = paramvals_tuple[param_name.index("EJ_GHz")]
    ancilla_copy.EC = paramvals_tuple[param_name.index("EC_GHz")]
    ancilla_copy.EL = paramvals_tuple[param_name.index("EL_GHz")]
    ancilla_copy.flux = paramvals_tuple[param_name.index("flux")]

    bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
    bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

    kappa_a_cap_list = []
    kappa_a_ind_list = []
    kappa_a_impd_list = []
    kappa_a_fbl_list = []
    kappa_a_qsp_tnl_list = []
    
    for level in range(0, ancilla_copy.truncated_dim):
        if level == starting_level:
            kappa_a_cap_list.append(0)
            kappa_a_ind_list.append(0)
            kappa_a_impd_list.append(0)
            kappa_a_fbl_list.append(0)
            kappa_a_qsp_tnl_list.append(0)
            continue

        freq = bare_evals[starting_level] - bare_evals[level]
        n_th = therm_factor(freq, temp_a, n_th_base)
        if n_th < n_th_threshold:
            break

        kappa_a_cap_list.append(ancilla_copy.t1_capacitive(
            i = starting_level, 
            j = level, 
            get_rate = True, 
            total = False, 
            T = temp_a, 
            esys = (bare_evals, bare_evecs),
            Q_cap = Q_cap,
        ))
        kappa_a_ind_list.append(ancilla_copy.t1_inductive(
            i = starting_level, 
            j = level, 
            get_rate = True, 
            total = False, 
            T = temp_a, 
            esys = (bare_evals, bare_evecs),
            Q_ind = Q_ind,
        ))
        # kappa_a_impd_list.append(ancilla_copy.t1_charge_impedance(
        #     i = starting_level, 
        #     j = level, 
        #     get_rate = True, 
        #     total = False,
        #     T = temp_a, 
        #     esys = (bare_evals, bare_evecs),
        #     Z = Z_char,
        # ))
        kappa_a_fbl_list.append(ancilla_copy.t1_flux_bias_line(
            i = starting_level, 
            j = level, 
            get_rate = True, 
            total = False,
            T = temp_a, 
            esys = (bare_evals, bare_evecs),
            Z = Z_fbl,
        ))
        kappa_a_qsp_tnl_list.append(ancilla_copy.t1_quasiparticle_tunneling(
            i = starting_level, 
            j = level, 
            get_rate = True, 
            total = False,
            T = temp_a, 
            esys = (bare_evals, bare_evecs), 
        ) * A_qsp_tnl)

    level_used = level - 1

    kappa_a_cap = np.sum(kappa_a_cap_list)
    kappa_a_cap_dominant = np.argmax(kappa_a_cap_list)

    kappa_a_ind = np.sum(kappa_a_ind_list)
    kappa_a_ind_dominant = np.argmax(kappa_a_ind_list)

    # kappa_a_impd = np.sum(kappa_a_impd_list)
    # kappa_a_impd_dominant = np.argmax(kappa_a_impd_list)

    kappa_a_fbl = np.sum(kappa_a_fbl_list)
    kappa_a_fbl_dominant = np.argmax(kappa_a_fbl_list)

    kappa_a_qsp_tnl = np.sum(kappa_a_qsp_tnl_list)
    kappa_a_qsp_tnl_dominant = np.argmax(kappa_a_qsp_tnl_list)

    return np.array([
        kappa_a_cap, 
        kappa_a_ind, 
        # kappa_a_impd, 
        kappa_a_fbl, 
        kappa_a_qsp_tnl,
        np.mean([
            kappa_a_cap_dominant, 
            kappa_a_ind_dominant, 
            # kappa_a_impd_dominant, 
            kappa_a_fbl_dominant, 
            kappa_a_qsp_tnl_dominant
        ]),
        level_used,
    ])

sweep_flxn_up = lambda ps, paramindex_tuple, paramvals_tuple, **kwargs: \
    sweep_flxn_depolarization(ps, paramindex_tuple, paramvals_tuple, starting_level=0, **kwargs)

sweep_flxn_down = lambda ps, paramindex_tuple, paramvals_tuple, **kwargs: \
    sweep_flxn_depolarization(ps, paramindex_tuple, paramvals_tuple, starting_level=1, **kwargs)

flxn_sweep_dict[(
    "kappa_a_up_cap", 
    "kappa_a_up_ind", 
    # "kappa_a_up_impd", 
    "kappa_a_up_fbl", 
    "kappa_a_up_qsp_tnl",
    "kappa_a_up_avg_transition",
    "kappa_a_up_levels_used",
)] = (sweep_flxn_up, ("temp_a", "n_th_base", 
    "Q_cap", "Q_ind", "Z_char", "Z_fbl", "A_qsp_tnl",))

flxn_sweep_dict[(
    "kappa_a_down_cap", 
    "kappa_a_down_ind", 
    # "kappa_a_down_impd", 
    "kappa_a_down_fbl", 
    "kappa_a_down_qsp_tnl",
    "kappa_a_down_avg_transition",
    "kappa_a_down_levels_used",
)] = (sweep_flxn_down, ("temp_a", "n_th_base", 
    "Q_cap", "Q_ind", "Z_char", "Z_fbl", "A_qsp_tnl",))

def sweep_flxn_dephasing(paramsweep, paramindex_tuple, paramvals_tuple, 
    ancilla_copy: scq.Fluxonium, A_flux=1e-6, A_cc=1e-7, **kwargs):

    ancilla_copy.EJ = paramvals_tuple[list(paramsweep.param_info.keys()).index("EJ_GHz")]
    ancilla_copy.EC = paramvals_tuple[list(paramsweep.param_info.keys()).index("EC_GHz")]
    ancilla_copy.EL = paramvals_tuple[list(paramsweep.param_info.keys()).index("EL_GHz")]
    ancilla_copy.flux = paramvals_tuple[list(paramsweep.param_info.keys()).index("flux")]
    bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
    bare_evals = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

    # default: 
    # "A_flux": 1e-6,  # Flux noise strength. Units: Phi_0
    # "A_ng": 1e-4,  # Charge noise strength. Units of charge e
    # "A_cc": 1e-7,  # Critical current noise strength. Units of critical current I_c

    kappa_phi_flux = ancilla_copy.tphi_1_over_f_flux(
        i = 0, 
        j = 1, 
        get_rate = True, 
        esys = (bare_evals, bare_evecs),
        A_flux = A_flux
    )
    kappa_phi_cc = ancilla_copy.tphi_1_over_f_cc(
        i = 0, 
        j = 1, 
        get_rate = True, 
        esys = (bare_evals, bare_evecs),
        A_cc = A_cc,
    )

    return np.array([kappa_phi_flux, kappa_phi_cc])

flxn_sweep_dict[(
    "kappa_phi_flux", 
    "kappa_phi_cc", 
)] = (sweep_flxn_dephasing, ("A_flux", "A_cc"))

# ##############################################################################
