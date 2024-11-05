import numpy as np
import qutip as qt

from typing import Tuple, Dict

def find_rotating_frame(
    evals: np.ndarray, 
    drive_op: qt.Qobj, 
    drive_freq: float,
    manual_constraints: Dict[Tuple[int, int], float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    We want to go to the rotating frame described by: 
    U = sum_i exp(-i * omega_i * t), which transforms the drive_op matrix elements as:
    U.dag * |i><j| * U = |i><j| * exp(i * (omega_i - omega_j) * t)
    
    The goal is to find the list of omega_i which makes the largest matrix element
    to be time-independent, meaning omega_i - omega_j + drive_freq = 0
    
    Parameters
    ----------
    evals: np.ndarray
        The eigenvalues of the static Hamiltonian.
    drive_op: qt.Qobj
        The drive operator in the eigenbasis of the static Hamiltonian.
    drive_freq: float
        The drive frequency.
    manual_constraints: Dict[Tuple[int, int], float], optional
        A dictionary of manually-specified constraints for the rotating frame.
        E.g. {(0, 1): 1.0, (1, 2): 2.0} indicates two following constraints:
        omega_0 - omega_1 + 1.0 = 0, omega_1 - omega_2 + 2.0 = 0
        
    Returns
    -------
    omega_vec: np.ndarray
        The list of omega_i.
    chosen_ij: np.ndarray
        The indices of the chosen matrix elements that should be time-independent
        (ignoring counter-rotating terms).
    """
    
    op_shape = drive_op.shape
    drive_op_sort_idx = np.argsort(np.abs(drive_op.full().ravel()))[::-1]
    drive_op_sort_2d_idx = np.array(np.unravel_index(drive_op_sort_idx, op_shape))

    # construct equations to solve: omega_i - omega_j + drive_freq = 0
    # or A * omega_vec = drive_freq_vec, where A is a matrix of 0, 1 and -1
    # go through each element in the drive_op (basically |i><j|), and try to cancel out the drive_freq
    chosen_ij = np.zeros((op_shape[0], 2), dtype=int)
    eq_set = np.zeros(op_shape, dtype=int)
    drive_freq_vec = -drive_freq * np.ones(op_shape[0])
    current_eq_num = 0
    
    # add a global shift to all omega_i: omega_0 = 0
    eq_set[0, 0] = 1
    drive_freq_vec[0] = 0
    current_eq_num += 1
    
    # add manual constraints
    if manual_constraints is not None:
        for (idx_i, idx_j), value in manual_constraints.items():
            eq_set[current_eq_num, idx_i] = 1
            eq_set[current_eq_num, idx_j] = -1
            chosen_ij[current_eq_num] = [idx_i, idx_j]
            drive_freq_vec[current_eq_num] = -value
            current_eq_num += 1
    
    
    for (idx_i, idx_j) in drive_op_sort_2d_idx.T:
        if evals[idx_i] >= evals[idx_j]:
            continue
        
        if current_eq_num == op_shape[0]:
            break
        
        eq_set[current_eq_num, idx_i] = 1
        eq_set[current_eq_num, idx_j] = -1
        
        # check linear independence
        if np.linalg.matrix_rank(eq_set[:current_eq_num+1]) == current_eq_num + 1:
            # linear independent, add the equation
            chosen_ij[current_eq_num] = [idx_i, idx_j]
            current_eq_num += 1
        else:
            # linear dependent, reset this equation
            eq_set[current_eq_num, idx_i] = 0
            eq_set[current_eq_num, idx_j] = 0
        
        if np.allclose(drive_op[idx_i, idx_j], 0):
            continue
        
    if current_eq_num < op_shape[0] - 1:
        print("Not enough linearly-independent equations to solve for all omega_i.")

    # solve for omega_vec
    omega_vec = np.linalg.solve(eq_set, drive_freq_vec)
    
    return omega_vec, chosen_ij

def H_in_rotating_frame(
    evals, drive_op, drive_freq, 
    manual_constraints: Dict[Tuple[int, int], float] = None,
    frame_omega_vec = None,
    ratio_threshold = 20,
) -> Tuple[qt.Qobj, qt.Qobj, np.ndarray]:
    """
    Transform the Hamiltonian and drive operator into the rotating frame.
    We assume the original drive term is drive_op * cos(drive_freq * t).
    
    Parameters
    ----------
    evals: np.ndarray
        The eigenvalues of the static Hamiltonian.
    drive_op: qt.Qobj
        The drive operator in the eigenbasis of the static Hamiltonian.
    drive_freq: float
        The drive frequency.
    manual_constraints: Dict[Tuple[int, int], float], optional
        A dictionary of manually-specified constraints for the rotating frame.
        E.g. {(0, 1): 1.0, (1, 2): 2.0} indicates two following constraints:
        omega_0 - omega_1 + 1.0 = 0, omega_1 - omega_2 + 2.0 = 0
    frame_omega_vec: np.ndarray, optional
        The list of frame transformation frequencies. It determines the frame 
        we want to go to, i.e. U = sum_i exp(-i * omega_i * t), which transforms 
        a matrix element by: U.dag * |i><j| * U = |i><j| * exp(i * (omega_i - 
        omega_j) * t)
    ratio_threshold: float, optional
        RWA is applicable if remaining frequency component/ matrix element > ratio_threshold.
        
    Returns
    -------
    h_rwa: qt.Qobj
        The transformed Hamiltonian in the rotating frame.
    drive_op_rwa: qt.Qobj
        The transformed drive operator in the rotating frame.
    frame_omega_vec: np.ndarray
        The list of frame transformation frequencies.
    """
    drive_freq = np.abs(drive_freq)

    H_shape = drive_op.shape
    drive_op_sort_idx = np.argsort(np.abs(drive_op.full().ravel()))[::-1]
    drive_op_sort_2d_idx = np.array(np.unravel_index(drive_op_sort_idx, H_shape))
    
    if frame_omega_vec is None:
        frame_omega_vec, _ = find_rotating_frame(evals, drive_op, drive_freq, manual_constraints)

    # check RWA applicability and get the drive term without fast rotating components
    drive_op_rwa = np.zeros_like(drive_op.full())
    for (idx_i, idx_j) in drive_op_sort_2d_idx.T:
        if idx_i == idx_j:
            drive_op_rwa[idx_i, idx_j] = drive_op[idx_i, idx_j]
            continue
        
        energy_diff = evals[idx_i] - evals[idx_j]
        if energy_diff > 0:
            # the positive energy difference matrix elements
            # are included by doing hermitian conjugate later
            continue
        
        # check if we have matrix element ~> omega_i - omega_j + drive_freq
        mat_elem = drive_op[idx_i, idx_j]
        remaining_freq_diff = np.abs(
            np.abs(frame_omega_vec[idx_i] - frame_omega_vec[idx_j]) 
            - drive_freq
        )
        remaining_freq_sum = (
            np.abs(frame_omega_vec[idx_i] - frame_omega_vec[idx_j]) 
            + drive_freq
        )
        
        if mat_elem == 0:
            ratio_diff = np.inf
            ratio_sum = np.inf
        else:
            ratio_diff = np.abs(remaining_freq_diff) / np.abs(mat_elem)
            ratio_sum = np.abs(remaining_freq_sum) / np.abs(mat_elem)
        
        if np.allclose(remaining_freq_diff, 0):
            # fully canceled out the time-dependent part of the drive, good.
            # It is divided by 2 because cosine(x) = (exp(ix) + exp(-ix))/2
            drive_op_rwa[idx_i, idx_j] = mat_elem / 2
            drive_op_rwa[idx_j, idx_i] = mat_elem.conjugate() / 2
            
            # check counter-rotating term
            if ratio_sum <= ratio_threshold:
                # counter-rotating term is not fast rotating, RWA is not valid
                # warn the user (but still include the term)
                print(
                    f"Counter-rotating term at |{idx_i}><{idx_j}| is not "
                    "fast rotating, thus the RWA is not applicable. "
                    "The matrix element is still included, though it's "
                    "potentially inaccurate. Here:"
                )
                print(f"\t energy difference: {energy_diff:.2e}")
                print(f"\t matrix element: {np.abs(mat_elem):.2e}, remaining freq: {remaining_freq_sum:.2f}, ratio: {ratio_sum:.2f} (< {ratio_threshold})")
                
        elif ratio_diff > ratio_threshold:
            # this rotating term is already fast rotating, RWA valid, discard
            # everything
            continue
        else:
            print(
                f"The rotating part of the drive operator |{idx_i}><{idx_j}| "
                "is time dependent and not fast enough to be discarded with "
                "RWA. The matrix element is discarded, though it's potentially "
                "inaccurate. Here:"
            )
            print(f"\t energy difference: {energy_diff:.2e}")
            print(f"\t matrix element: {np.abs(mat_elem):.2e}, remaining freq: {remaining_freq_diff:.2f}, ratio: {ratio_diff:.2f} (< {ratio_threshold})")

    drive_op_rwa = qt.Qobj(drive_op_rwa)
    
    h_rwa = qt.qdiags(evals - frame_omega_vec, 0) 
    
    return h_rwa, drive_op_rwa, frame_omega_vec