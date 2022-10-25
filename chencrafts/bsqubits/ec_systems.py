import scqubits as scq
import numpy as np
import qutip as qt

from typing import Dict

def initialize_joint_system_tmon(
    para: Dict,
    sim_para: Dict,
    returns=["hilbert_space"],
    return_ndarray=False
):
    omega_s, EJ, EC, ng, g_sa, = [para[key] 
        for key in ["oemga_s", "EJ", "EC", "ng", "g_sa"]]
    sys_dim, anc_ncut, anc_dim = [sim_para[key] 
        for key in ["sys_dim", "anc_ncut", "anc_dim"]]
    
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

    # test ancilla dim is valid
    _, evecs = ancilla.eigensys(evals_count=anc_dim)
    assert (np.abs(evecs[-1][-3:]) < 1e-8).all()

    subsystem_list = [system, ancilla]
    h_space = scq.HilbertSpace(subsystem_list)

    h_space.add_interaction(
        g = g_sa,
        op1 = system.n_operator,
        op2 = ancilla.n_operator,
        add_hc = False,
        id_str = "sys-anc"
    )

    hamiltonian = h_space.hamiltonian()

    # Drive Operators
    n_anc = scq.identity_wrap(ancilla.n_operator(), ancilla, subsystem_list)

    return_obj = []
    for ret in returns:
        if ret == "hamiltonian":
            if return_ndarray:
                return_obj.append(hamiltonian.data.toarray())
            else:
                return_obj.append(hamiltonian)
        elif ret == "H_controls":
            if return_ndarray:
                return_obj.append([n_anc.data.toarray()])
            else:
                return_obj.append([n_anc])
        elif ret == "hilbert_space":
            return_obj.append(h_space)
        elif ret == "subsys":
            return_obj.append(subsystem_list)

    if len(return_obj) == 1:
        return return_obj[0]

    return return_obj
