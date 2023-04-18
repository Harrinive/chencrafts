import numpy as np
from scipy.fft import fft, fftfreq

from scqubits.core.storage import WaveFunction
import scqubits as scq

from typing import List, Tuple


def wavefunc_FT(
    x_list: List | np.ndarray, 
    amp_x: List | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list = np.array(x_list)
    amp_x = np.array(amp_x)

    x0, x1 = x_list[0], x_list[-1]
    dx = x_list[1] - x_list[0]

    amp_p_dft = fft(amp_x)
    n_list = fftfreq(amp_x.size) * 2 * np.pi / dx

    # In order to get a discretisation of the continuous Fourier transform
    # we need to multiply amp_p_dft by a phase factor
    amp_p = amp_p_dft * dx * np.exp(-1j * n_list * x0) / (np.sqrt(2*np.pi))

    return n_list, amp_p

def label_convert(idx: Tuple | List | int, h_space: scq.HilbertSpace):

    dims = h_space.subsystem_dims

    if isinstance(idx, tuple | list):
        assert (np.array(idx) < np.array(dims)).all(), f"index is not valid for system dimension {dims}"

        drs_idx = 0
        for dim_idx, bare_idx in enumerate(idx):
            drs_idx += np.prod(dims[dim_idx+1:]) * bare_idx

        return int(drs_idx)
    
    elif isinstance(idx, int):
        assert (idx < np.prod(dims)).all(), f"index is not valid for system size {np.prod(dims)}"

        bare_idx_list = []
        for dim_idx in range(len(dims)):
            bare_idx_list.append(int(idx / np.prod(dims[dim_idx+1:])))
            idx = idx % int(np.prod(dims[dim_idx+1:]))

        return tuple(bare_idx_list)

    else:
        raise ValueError(f"Only support list/tuple/int as an index.")
