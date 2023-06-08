
import numpy as np

from typing import Dict, List, Tuple

# ##############################################################################
# fit data
def combine_peaks_dict(*dicts):
    """
    combine transition dictionaries
    
    the arguments should be multiple trans_dicts: 
    {
        (0, 1): [[para, freq], [para, freq], ...],
        (0, 2): [[para, freq], [para, freq], ...],
        ...
    }
    """
    combined_dict = {}
    for d in dicts:
        for key, val in d.items():
            if key in combined_dict:
                combined_dict[key].extend(val)
            else:
                combined_dict[key] = val

    return combined_dict

def process_selected_data(
    trans_freq_dict = {}
) -> Tuple[List, int]:
    """
    trans_freq_dict = {
        (0, 1): [[para, freq], [para, freq], ...],
        ...
    }

    Returns
    -------
    parameters used and maximal level used
    """
    paras = set()
    max_trans = 0
    for key, vals in trans_freq_dict.items():
        if vals == []:
            continue
        max_trans = np.max([max_trans, key[0], key[1]])
        for p in np.array(vals)[:, 0]:
            paras.add(p)
    
    return list(np.sort(list(paras))), max_trans

def MSE(
    sys_param: Dict[str, float], 
    sys_energy_func, extracted_peaks, 
    trust_transition=True, evals_count=None, max_initial_level=0,
    osc_energy=None, 
    print_actual_transition=False,
):
    """
    Calculate the mean square error between the extracted peaks and the numerically calculated
    spectrum.         

    Parameters
    ----------
    sys_param: Dict[str, float]
        The parameters that will be passed to the sys_energy_func by key word arguments.
    sys_energy_func: function
        The function that will be used to calculate the energy spectrum. The function should
        be in the format of
            `sys_energy_func(**sys_param, param_list, evals_count)`
        where `param_list` is a list of parameters that contains all of the sweep parameters
        stored in the extracted_peaks and `evals_count` is the maximal level that will be
        calculated. The function should return a 2D array of eigenenergies with the shape of
        (len(param_list), evals_count).
    extracted_peaks: Dict
        The extracted peaks in the format of
        {
            (0, 1): [[para, freq], [para, freq], ...],
            (0, 2): [[para, freq], [para, freq], ...],
            ...
        }
    trust_transition: bool
        Whether to trust the transition level given by the extracted_peaks.keys(). 
        If it is True, then the extracted peaks will be compared with the numerically 
        calculated spectrum corresponding to the given transition levels. If it is False,
        then the extracted peaks will be compared with the closest numerically calculated 
        spectrum.
    evals_count: int
        The maximal level that will be calculated. If it is None, then the maximal level
        will be determined by the extracted peaks.
    max_initial_level: int
        The maximal initial level that will be calculated. By default, it is 0.
    osc_energy: float
        The energy of the oscillator that coupled with the system. If it is given, then 
        sideband transitions will be included when not trusting transitions.
    print_actual_transition: bool
        When not trusting transitions, whether to print the actual transition level that
        corresponds to the minimal error. By default, it is False.

    """
    # find the parameter needed to calculate the energy
    # as well as the maximal level needed in the calculation
    params, max_level_needed = process_selected_data(extracted_peaks)

    # determine the maximal level needed in the calculation
    if evals_count is not None:
        max_level_needed = evals_count
    if max_level_needed <= max_initial_level:
        max_initial_level = max_level_needed - 1

    # calculate the energy spectrum numerically
    energies = sys_energy_func(**sys_param, param_list=params, evals_count=max_level_needed)

    # calculate the transitions
    all_transitions = np.zeros(energies.shape + (max_initial_level+1,))
    for starting_level in range(max_initial_level+1):
        all_transitions[..., starting_level] = energies - energies[:, starting_level:starting_level+1]

    # include sideband
    if osc_energy is not None:
        all_transitions = np.repeat(all_transitions[..., None], 2, axis=-1)
        all_transitions[..., 1] = all_transitions[..., 1] - osc_energy

    # calculate the error
    error = 0
    counts = 0
    for (lvl_1, lvl_2), peak_list in extracted_peaks.items():
        if peak_list == []:
            continue

        for p, freq_exp in peak_list:
            # p: parameter, freq_exp: frequency from experiment
            counts += 1

            freq_exp = np.abs(freq_exp)

            param_idx = params.index(p)  # index of the parameter in the numerical result

            if not trust_transition or lvl_1 == -1 or lvl_2 == -1:
                min_error = np.min((all_transitions[param_idx, ...] - freq_exp)**2)

                if print_actual_transition:
                    argmin = np.argmin((all_transitions[param_idx, ...] - freq_exp)**2)
                    transition_idx = np.unravel_index(argmin, all_transitions.shape[1:])
                    if osc_energy is None:
                        print_sideband = ""
                    elif transition_idx[-1] == 1:   # last index is 1
                        print_sideband = ", sideband"
                    else:
                        print_sideband = ""
                    print(f"({p, freq_exp}): {transition_idx[0]} <-> {transition_idx[1]} {print_sideband}")

                error += min_error
                
            else:
                freq_num = np.abs(energies[param_idx, lvl_1] - energies[param_idx, lvl_2])
                error += (freq_exp - freq_num)**2

    return (error / counts)
