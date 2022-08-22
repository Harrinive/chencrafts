from statistics import median
from typing import Callable
import numpy as np
from scipy.integrate import odeint

# ##############################################################################
def sinusoidal(
    base_ang_freq,
    duration,
    drive_amp=None,
    tgt_mat_elem=1, 
):
    """
    Default is pi pulse, which means approximately: duration * drive_amp * tgt_mat_elem = pi. 
    Bloch-Siegert effect is considered

    The drive_amp will be automatically delected is not specified. 
    """
    if drive_amp is None:
        # drive_amp = np.sqrt((np.pi / duration)**2 - (2 * base_ang_freq)**2) / tgt_mat_elem
        drive_amp = np.pi / duration / np.abs(tgt_mat_elem)
    
    # Bloch–Siegert shift
    freq_shift = drive_amp**2 / base_ang_freq / 4
    drive_freq = base_ang_freq - freq_shift * 1

    return lambda t, *args: drive_amp * np.cos(drive_freq * t)

# ##############################################################################
def _gaussian_function(t, t_mid, sigma, amp=1):
    return amp * (np.exp(-(t - t_mid)**2 / 2 / sigma**2))

def gaussian(
    base_ang_freq, 
    sigma, 
    duration, 
    tgt_mat_elem=1, 
    base_amp=1, 
    pulse_start_time=0
):  
    """
    Default is pi pulse, change base_amp if needed
    """
    t_mid = pulse_start_time + duration/2

    mean_amp_scale = (odeint(
        lambda t, *args: (
            _gaussian_function(t, t_mid, sigma, 1) 
            - _gaussian_function(0, t_mid, sigma, 1)
        ),
        y0 = 0,
        t = [0, t_mid],
        tfirst = True,
    )[-1, 0] / t_mid) 
    pulse_amp = (base_amp / mean_amp_scale) * np.pi / duration / np.abs(tgt_mat_elem)

    # set envelope to be 0 at the beginning and the end
    env_bias = _gaussian_function(0, t_mid, sigma, pulse_amp)

    # Bloch–Siegert shift
    sine_drive_amp = np.pi / duration / np.abs(tgt_mat_elem)
    freq_shift = (sine_drive_amp)**2 / base_ang_freq / 4
    drive_freq = base_ang_freq - freq_shift

    return lambda t, *args: (_gaussian_function(
        t,
        t_mid,
        sigma,
        pulse_amp
    ) - env_bias) * np.cos(drive_freq * t)

# ##############################################################################
def _phase_from_init(base_ang_freq, freq_func, init_t, init_val, current_t):
    osc_cycles = (current_t - init_t) * base_ang_freq / 2 / np.pi
    integrate_steps = int(osc_cycles / 30) + 2

    current_phase = odeint(
        freq_func, 
        y0 = init_val,
        t = np.linspace(init_t, current_t, integrate_steps),
        tfirst=True,
    )[-1, 0]

    return current_phase

def drag_gaussian(
    base_ang_freq, 
    sigma, 
    duration, 
    tgt_mat_elem=1, 
    leaking_mat_elem=np.sqrt(2), 
    non_lin=0, 
    base_amp=1, 
    pulse_start_time=0
) -> Callable:
    """
    Default is pi pulse, change base_amp if needed
    """
    if np.abs(non_lin) < 1 / duration:
        raise ValueError("Non-linearity of the system should be specified and"
            "much larger than the pulse amplitude.")

    t_mid = pulse_start_time + duration/2

    mean_amp_scale = (odeint(
        lambda t, *args: (
            _gaussian_function(t, t_mid, sigma, 1) 
            - _gaussian_function(0, t_mid, sigma, 1)
        ),
        y0 = 0,
        t = [0, t_mid],
        tfirst = True,
    )[-1, 0] / t_mid) 
    pulse_amp = (base_amp / mean_amp_scale) * np.pi / duration / np.abs(tgt_mat_elem)

    mat_elem_diff = np.abs(leaking_mat_elem / tgt_mat_elem)

    # set envelope to be 0 at the beginning and the end
    env_bias = _gaussian_function(0, t_mid, sigma, pulse_amp)

    # drag pulse: detuning modification
    def drive_freq(t, *args):
        eps_pi_2 = (_gaussian_function(t, t_mid, sigma, pulse_amp) - env_bias)**2
        detuning = (
            (mat_elem_diff**2 - 4) / (4 * non_lin) * eps_pi_2
            - (mat_elem_diff**4 - 7*mat_elem_diff**2 + 12) / (16 * non_lin**3) * eps_pi_2**2
        )
        return base_ang_freq - detuning

    def pulse_func(t, t_n_phase=None, return_xyp=False, *args):
        """
        An external list [time, phase] can be input and will speed up the calculation
        """
        assert isinstance(t, float), "The input time should be a float"

        eps_pi = _gaussian_function(t, t_mid, sigma, pulse_amp) - env_bias
        eps_pi_dot = -_gaussian_function(t, t_mid, sigma, pulse_amp) * (t - t_mid) / sigma**2

        eps_x = (
            eps_pi
            + (mat_elem_diff**2 - 4) / (8 * non_lin**2) * eps_pi**3
            - (13*mat_elem_diff**4 - 76*mat_elem_diff**2 + 112) / (128*non_lin**4) * eps_pi**5
        )
        eps_y = (
            - eps_pi_dot / non_lin
            + 33*(mat_elem_diff**2 - 2) / (24*non_lin**3) * eps_pi**2 * eps_pi_dot
        )

        if t_n_phase is None:
            phase = _phase_from_init(base_ang_freq, drive_freq, 0, 0, t)
        elif len(t_n_phase) == 2:
            init_t, init_phase = t_n_phase
            phase = _phase_from_init(base_ang_freq, drive_freq, init_t, init_phase, t)
            t_n_phase[0] = t
            t_n_phase[1] = phase
        else:
            raise TypeError("The time and phase list is invalid:", t_n_phase)
        
        if not return_xyp:
            return eps_x * np.cos(phase) + eps_y * np.sin(phase)
        else:
            return eps_x, eps_y, phase

    return pulse_func