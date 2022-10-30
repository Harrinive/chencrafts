import numpy as np
import scqubits as scq
import qutip as qt
import matplotlib.pyplot as plt

from typing import Dict

from chencrafts.toolbox.data_processing import NSArray

PI2 = np.pi * 2

# ##############################################################################
def _sweep_tmon(
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
        return ancilla.t1_capacitive(
            i=0, 
            j=1, 
            get_rate=True, 
            total=False, 
            T=Temp, 
            esys=(bare_evals, bare_evecs), 
        ) / Q_t1_coef
    sweep.add_sweep(sweep_gamma_up, "gamma_up")

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
    sweep.add_sweep(sweep_gamma_down, "gamma_down")

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
    sweep.add_sweep(sweep_gamma_phi, "gamma_phi")

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
    # sweep.add_sweep(sweep_gamma_cc, "gamma_phi_cc")

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
    # sweep.add_sweep(sweep_gamma_ng, "gamma_phi_ng")

    # def sweep_state(paramsweep: scq.ParameterSweep, paramindex_tuple, paramvals_tuple, **kwargs):
    #     """
    #     Get the logical zero state for each parameter
    #     """

    #     evals = paramsweep["evals"][paramindex_tuple]
    #     evecs = paramsweep["evecs"][paramindex_tuple]

    #     h_space = paramsweep.hilbertspace
    #     dims = [subsys.truncated_dim for subsys in h_space.subsys_list]
    #     paramsweep._update_hilbertspace(paramsweep, *paramvals_tuple)

    #     h_space.generate_lookup(dressed_esys=(evals, evecs))
    #     drs_idx = h_space.dressed_index((3, 0))
    #     state = evecs[drs_idx]

    #     qt.plot_fock_distribution(qt.ptrace(state, 0))
    #     plt.show()

    #     return 0
    # sweep.add_sweep(sweep_state, "test")

    # def sweep_n_bar(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
    #     """
    #     The overlap between dressed states
    #     """
    #     return

    # def test_sweep(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
    #     print(paramvals_tuple)
    #     return paramvals_tuple[-1]
    # sweep.add_sweep(test_sweep, "test_sweep")
    
    def sweep_min_detuning(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
        bare_evals0 = paramsweep["bare_evals"]["subsys":0][paramindex_tuple]
        bare_evals1 = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

        sys_freq = bare_evals0[1] - bare_evals0[0]

        anc_freq_0x = bare_evals1[1:3] - bare_evals1[0]
        # anc_freq_1x = bare_evals1[2:3] - bare_evals1[1]
        # anc_freq_2x = bare_evals1[3:4] - bare_evals1[2]

        return np.min(np.abs(anc_freq_0x - sys_freq))
    sweep.add_sweep(sweep_min_detuning, "min_detuning")

    def sweep_detuning(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
        bare_evals0 = paramsweep["bare_evals"]["subsys":0][paramindex_tuple]
        bare_evals1 = paramsweep["bare_evals"]["subsys":1][paramindex_tuple]

        sys_freq = bare_evals0[1] - bare_evals0[0]

        anc_freq_0x = bare_evals1[1:3] - bare_evals1[0]
        # anc_freq_1x = bare_evals1[2:3] - bare_evals1[1]
        # anc_freq_2x = bare_evals1[3:4] - bare_evals1[2]

        return np.min(np.abs(anc_freq_0x - sys_freq))
        

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



def sweep_for_params(
    sweep: scq.ParameterSweep, 
    ancilla: scq.Transmon | scq.Fluxonium, 
    para_dict,
):

    scq.settings.T1_DEFAULT_WARNING = False
    scq.settings.PROGRESSBAR_DISABLED = True

    def check_qubit_convergence(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs):
        bare_evecs = np.array(paramsweep["bare_evecs"]["subsys": 1][paramindex_tuple])
        return np.max(np.abs(bare_evecs[-1][-3:]))
    sweep.add_sweep(check_qubit_convergence, "convergence")

    if isinstance(ancilla, scq.Transmon):
        _sweep_tmon(
            sweep, ancilla, para_dict
        )
        
    elif isinstance(ancilla, scq.Fluxonium):
        _sweep_fl(
            sweep, ancilla, para_dict
        )