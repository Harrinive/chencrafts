import numpy as np

import scqubits as scq
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray

from chencrafts.cqed.custom_sweeps import (
    sweep_purcell_factor,
    sweep_gamma_1,
    sweep_gamma_phi,
    sweep_convergence,
)

from chencrafts.cqed.decoherence import (
    n_th,
    readout_error,
    qubit_addi_energy_relax_w_res,
    qubit_shot_noise_dephasing_w_res,
)

from chencrafts.cqed.special_states import (
    cat
)

PI2 = np.pi * 2


def batched_sweep_general(
    sweep: ParameterSweep, res_mode_idx = 0, qubit_mode_idx = 1, **kwargs
):
    """
    Notes: 
    temps, tempa --> T
    n_th_a is missing

    """

    # Re-store the data from the sweep
    qubit_evals = sweep["bare_evals"][qubit_mode_idx]
    sweep.store_data(
        omega_a_GHz = (qubit_evals[..., 1] 
            - qubit_evals[..., 0]),
        omega_s_GHz = (sweep["bare_evals"][res_mode_idx][..., 1] 
            - sweep["bare_evals"][res_mode_idx][..., 0]),
        chi_sa = sweep["chi"][qubit_mode_idx, res_mode_idx][..., 1] * PI2,
        K_s = sweep["kerr"][res_mode_idx, res_mode_idx] * PI2,
        chi_prime = sweep["chi_prime"][qubit_mode_idx, res_mode_idx][..., 1] * PI2,
    )

    # non-linearities
    non_lin = np.min(np.abs([
        qubit_evals[..., 2:] - qubit_evals[..., 0:1] - sweep["omega_a_GHz"][..., None],
        qubit_evals[..., 2:] - qubit_evals[..., 1:2] - sweep["omega_a_GHz"][..., None],
    ]), axis=(0, -1)) * PI2
    sweep.store_data(non_lin = NamedSlotsNdarray(non_lin, sweep.param_info))

    # convergence
    sweep.add_sweep(
        sweep_convergence, "convergence", mode_idx = qubit_mode_idx
    )


def batched_sweep_bare_decoherence(
    sweep: ParameterSweep, res_mode_idx = 0, qubit_mode_idx = 1, **kwargs
):
    """
    Should be called after batched_sweep_general

    keyword arguments required for the decoherence function will be filled in automatically. 
    Arguments will be filled in by the following priority (from high to low):
    - swept parameters
    - sweep[<name>]
    - kwargs[<name>] (kwargs of this function)
    """
    # some parameters for use
    qubit_dim = sweep.hilbertspace.subsystem_dims[qubit_mode_idx]
    params = kwargs | sweep.parameters.meshgrid_by_name() 
    
    # cavity relaxation
    sweep.store_data(
        kappa_s = PI2 * sweep["omega_s_GHz"] / params["Q_s"]
    )
    sweep.store_data(
        n_th_s = n_th(sweep["omega_s_GHz"], params["temp_s"], params["n_th_base"])
    )

    # qubit decoherence
    for channel in ["capacitive", "inductive", "quasiparticle_tunneling"]:
        sweep.add_sweep(
            sweep_gamma_1, "kappa_a_1_"+channel, 
            mode_idx=qubit_mode_idx, channel_name="t1_"+channel, 
            i_list=[0, 1], j_list=range(qubit_dim),
            **kwargs
        )
    for channel in ["flux", "ng", "cc"]:
        sweep.add_sweep( 
            sweep_gamma_phi, "kappa_a_phi_"+channel, 
            mode_idx=qubit_mode_idx, channel_name="tphi_1_over_f_"+channel, i=0, j=1,
            **kwargs
        )
    total_kappa_1 = np.sum([
        sweep["kappa_a_1_"+channel] for channel in ["capacitive", "inductive", "quasiparticle_tunneling"]
    ], axis=(0, -1))
    total_kappa_1 = NamedSlotsNdarray(total_kappa_1, sweep.param_info)
    total_kappa_phi = np.sum([
        sweep["kappa_a_phi_"+channel] for channel in ["flux", "ng", "cc"]
    ], axis=0)
    total_kappa_phi = NamedSlotsNdarray(total_kappa_phi, sweep.param_info)
    sweep.store_data(
        kappa_a_up_base = total_kappa_1[..., 0],
        kappa_a_down_base = total_kappa_1[..., 1],
        kappa_a_phi_base = total_kappa_phi,
    )

def batched_sweep_purcell_fock(
    sweep: ParameterSweep, res_mode_idx = 0, qubit_mode_idx = 1, **kwargs
):
    sweep.add_sweep(
        sweep_purcell_factor, "purcell_factor_fock",
        res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
        res_state_func = 1, qubit_state_index = 0,   
    )


def batched_sweep_purcell_cats(
    sweep: ParameterSweep, res_mode_idx = 0, qubit_mode_idx = 1, **kwargs
):
    # disp will be filled in by the following priority (from high to low):
    # - swept parameters
    # - sweep[<name>]
    # - kwargs[<name>] (kwargs of this function)

    def cat_x(basis, disp, **kwargs):
        if len(basis) < disp**2 + disp:
            raise RuntimeError("basis is too small for the displacement")
        return cat([(1, disp), (1, -disp), (1, 1j * disp), (1, -1j * disp)], basis)
    def cat_y(basis, disp, **kwargs):
        if len(basis) < disp**2 + disp:
            raise RuntimeError("basis is too small for the displacement")
        return cat([(1, disp), (1, -disp), (1j, 1j * disp), (1j, -1j * disp)], basis)
    def cat_z(basis, disp, **kwargs):
        if len(basis) < disp**2 + disp:
            raise RuntimeError("basis is too small for the displacement")
        return cat([(1, disp), (1, -disp)], basis)
    
    sweep.add_sweep(
        sweep_purcell_factor, "purcell_factor_x",
        res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
        res_state_func = cat_x, qubit_state_index = 0,
        **kwargs
    )
    sweep.add_sweep(
        sweep_purcell_factor, "purcell_factor_y",
        res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
        res_state_func = cat_y, qubit_state_index = 0,
        **kwargs
    )
    sweep.add_sweep(
        sweep_purcell_factor, "purcell_factor_z",
        res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
        res_state_func = cat_z, qubit_state_index = 0,
        **kwargs
    )

    sweep.store_data(
        purcell_factor_cat = (
            sweep["purcell_factor_x"] 
            + sweep["purcell_factor_y"] 
            + sweep["purcell_factor_z"]
        ) / 3,
    )


def batched_sweep_readout(
    sweep: ParameterSweep, res_mode_idx = 0, qubit_mode_idx = 1, **kwargs
):
    """
    Should be called after batched_sweep_general and batched_sweep_bare_decoherence
    """
    # some parameters for use
    qubit = sweep.hilbertspace.subsystem_list[qubit_mode_idx]
    params = kwargs | sweep.parameters.meshgrid_by_name()

    # some assupmtions should be made for a readout resonator
    detuning_ar = 2 * PI2   
    omega_r_GHz = sweep["omega_a_GHz"] \
        + detuning_ar * np.sign(sweep["omega_a_GHz"] - sweep["omega_s_GHz"])


    sweep.store_data(
        chi_ar = params["chi_ar_by_kappa_r"] * params["kappa_r"] * np.ones_like(sweep["omega_a_GHz"]),
    )

    # critical photon number
    # for the readout qubit - readout_resonator system, lambda_2 = (g / delta)**2
    # n_crit = g^2 / delta^2 / 4  
    # (only for a two-level system model, providing an upper bound)
    if isinstance(qubit, scq.Transmon):
        # chi = non_lin * g^2 / delta^2
        lambda_2 = np.abs(sweep["chi_ar"] / sweep["non_lin"])
    elif isinstance(qubit, scq.Fluxonium):
        # assume chi is only contributed by the coupling of two levels
        # chi = 2 * g^2 / delta, so lambda_2 = (chi / 2) / delta
        lambda_2 = np.abs(sweep["chi_ar"] / 2 / detuning_ar)
    else:
        raise RuntimeError(f"{type(qubit)} is not supported for calculating n_crit")

    sweep.store_data(
        n_crit = 1 / 4 / lambda_2,
    )
    sweep.store_data(
        n_ro = params["n_ro_by_n_crit"] * sweep["n_crit"]
    )

    # readout infidelity
    M_ge = readout_error(
        np.sqrt(sweep["n_ro"]), 
        params["kappa_r"], 
        params["tau_m"],
    )
    M_eg = M_ge.copy()
    sweep.store_data(
        M_ge = M_ge,
        M_eg = M_eg,
    )

    # qubit relaxation with the presence of the readout resonator
    kappa_a_down_ro, kappa_a_up_ro = qubit_addi_energy_relax_w_res(
        sweep["kappa_a_up_base"], sweep["kappa_a_phi_base"], np.sqrt(lambda_2),
        sweep["n_ro"], sweep["n_crit"], 
        params["kappa_r"],
    ) 
    kappa_a_down_r, kappa_a_up_r = qubit_addi_energy_relax_w_res(
        sweep["kappa_a_up_base"], sweep["kappa_a_phi_base"], np.sqrt(lambda_2),
        0, sweep["n_crit"], 
        params["kappa_r"],
    ) 

    sweep.store_data(
        kappa_a_down_ro = kappa_a_down_ro,
        kappa_a_up_ro = kappa_a_up_ro,
        kappa_a_down_r = kappa_a_down_r * params["purcell_filt"],
        kappa_a_up_r = kappa_a_up_r * params["purcell_filt"],
    )

    # shot noise dephasing
    try:
        n_th_r = params["n_th_r"] 
    except KeyError:
        n_th_r = n_th(omega_r_GHz, params["temp_r"], params["n_th_base"])
    kappa_a_phi_r = qubit_shot_noise_dephasing_w_res(
        params["kappa_r"], sweep["chi_ar"], n_th_r
    )

    sweep.store_data(
        kappa_a_phi_r = kappa_a_phi_r,
    )

def batched_sweep_total_decoherence(
    sweep: ParameterSweep, res_mode_idx = 0, qubit_mode_idx = 1, **kwargs
):
    """
    Should be called after calling
    batched_sweep_general, 
    batched_sweep_bare_decoherence,
    batched_sweep_purcell_fock,
    batched_sweep_purcell_cat,
    and batched_sweep_readout
    """

    # resonator decoherence
    sweep.store_data(
        gamma_down = sweep["kappa_s"] * sweep["purcell_factor_cat"][..., 0]
            * (1 + sweep["n_th_s"])
            + sweep["kappa_a_down_base"] * sweep["purcell_factor_cat"][..., 1],
        gamma_01_down = sweep["kappa_s"] * sweep["purcell_factor_fock"][..., 0]
            * (1 + sweep["n_th_s"])
            + sweep["kappa_a_down_base"] * sweep["purcell_factor_cat"][..., 1],
        gamma_up = sweep["kappa_s"] * (sweep["purcell_factor_cat"][..., 0] + 1)
            * sweep["n_th_s"]
            + sweep["kappa_a_up_base"] * sweep["purcell_factor_cat"][..., 1],
    )

    # qubit decoherence
    sweep.store_data(
        Gamma_down = sweep["kappa_a_down_base"] + sweep["kappa_a_down_r"],
        Gamma_up = sweep["kappa_a_up_base"] + sweep["kappa_a_up_r"],
        Gamma_phi = sweep["kappa_a_phi_base"] + sweep["kappa_a_phi_r"],
        Gamma_down_ro = sweep["kappa_a_down_base"] + sweep["kappa_a_down_ro"],
        Gamma_up_ro = sweep["kappa_a_up_base"] + sweep["kappa_a_up_ro"],
    )
    sweep.store_data(
        T1_a = 1 / sweep["Gamma_down"],
        T2_a = 1 / (sweep["Gamma_phi"] + sweep["Gamma_down"] / 2),
        T_s = 1 / sweep["gamma_01_down"],
    )

def batched_sweep_pulse(
    sweep: ParameterSweep, res_mode_idx = 0, qubit_mode_idx = 1, **kwargs
):
    """
    Should be called after calling
    batched_sweep_general,
    """
    # some parameters for use
    params = sweep.parameters.meshgrid_by_name() | kwargs

    # find sigma based on:
    # 1. should be larger than (1 / non_lin) to reduce leakage
    # 2. should be larger than (1 / freq) to reduce non-RWA error
    # 3. should be larger than a minimum value (2) empirically
    min_sigma_by_nonlin = np.abs(params["sigma_2_K_a"] / sweep["non_lin"])
    min_sigma_by_freq = np.abs(params["sigma_omega"] / sweep["omega_a_GHz"] / np.pi / 2)
    min_sigma = np.ones_like(sweep["omega_a_GHz"]) * 2.0
    sigma = np.max([
        min_sigma, min_sigma_by_nonlin, min_sigma_by_freq
    ], axis=0)
    sweep.store_data(
        sigma = sigma,
    )
    sweep.store_data(
        tau_p = sweep["sigma"] * np.abs(params["tau_p_by_sigma"]),
        tau_p_eff = sweep["sigma"] * np.abs(params["tau_p_eff_by_sigma"])
    )

def batched_sweep_cat_code(
    sweep: ParameterSweep, res_mode_idx = 0, qubit_mode_idx = 1, **kwargs
):
    """
    Call all of the sweeps defined above
    """
    batched_sweep_general(sweep, res_mode_idx, qubit_mode_idx, **kwargs)
    batched_sweep_bare_decoherence(sweep, res_mode_idx, qubit_mode_idx, **kwargs)
    batched_sweep_purcell_fock(sweep, res_mode_idx, qubit_mode_idx, **kwargs)
    batched_sweep_purcell_cats(sweep, res_mode_idx, qubit_mode_idx, **kwargs)
    batched_sweep_readout(sweep, res_mode_idx, qubit_mode_idx, **kwargs)
    batched_sweep_total_decoherence(sweep, res_mode_idx, qubit_mode_idx, **kwargs)
    batched_sweep_pulse(sweep, res_mode_idx, qubit_mode_idx, **kwargs)

    # other parameters ralated to cat code's failure rate

    params = kwargs | sweep.parameters.meshgrid_by_name() 

    sweep.store_data(
        n_bar_s = sweep["purcell_factor_cat"][..., 0]
    )

    sweep.store_data(
        T_M = params["T_W"] + params["tau_FD"] + params["tau_m"] \
            + np.pi / np.abs(sweep["chi_sa"]) + 3 * sweep["tau_p"]
    )