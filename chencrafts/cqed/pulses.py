from mimetypes import init
from statistics import median
from typing import Callable
import numpy as np
from scipy.integrate import odeint

# ##############################################################################
class PulseBase:
    def __init__(
        self,
        base_angular_freq: float,
        duration: float,
        rotation_angle: float,
        tgt_mat_elem: float, 
        init_time: float = 0,
    ) -> None:
        self.base_angular_freq = base_angular_freq
        self.duration = duration
        self.rotation_angle = rotation_angle
        self.tgt_mat_elem = tgt_mat_elem
        self.init_time = init_time

        self.drive_amp = rotation_angle / duration 
        self.env_amp = self.drive_amp / np.abs(tgt_mat_elem)
        self.drive_freq = self.base_angular_freq

        return

    def envelope(self, t):
        """Only support scalar t"""
        if not isinstance(t, float):
            raise TypeError("The input time should be a float")

        return self.env_amp

    def __call__(self, t) -> float:
        """Only support scalar t"""
        if not isinstance(t, float):
            raise TypeError("The input time should be a float")

        env = self.envelope(t)
        t_bias = t - self.init_time
        return env * np.cos(self.drive_freq * t_bias)

# ##############################################################################
class Sinusoidal(PulseBase):
    def __init__(
        self,
        base_angular_freq: float,
        duration: float,
        rotation_angle: float = np.pi,
        tgt_mat_elem: float = 1.0, 
        init_time: float = 0,
        with_freq_shift: bool = True,
    ) -> None:
        super().__init__(
            base_angular_freq, 
            duration, 
            rotation_angle,
            tgt_mat_elem, 
            init_time,
        )

        # modify the drive freq with the Bloch–Siegert shift
        if with_freq_shift:
            freq_shift = self.drive_amp**2 / self.base_angular_freq / 4
            self.drive_freq = self.base_angular_freq - freq_shift * 1

# ##############################################################################
def _gaussian_function(t, t_mid, sigma, amp=1):
    return amp * (np.exp(-(t - t_mid)**2 / 2 / sigma**2))

def _gaussian_mean_amp(duration, sigma):
    half_duration = duration / 2
    mean_amp_scale = odeint(
        lambda t, *args: (
            _gaussian_function(t, half_duration, sigma, 1) 
            - _gaussian_function(0, half_duration, sigma, 1)
        ),
        y0 = 0,
        t = [0, half_duration],
        tfirst = True,
    )[-1, 0] / half_duration

    return mean_amp_scale

class Gaussian(PulseBase):
    def __init__(
        self, 
        base_angular_freq: float, 
        duration: float, 
        sigma: float, 
        rotation_angle: float = np.pi, 
        tgt_mat_elem: float = 1.0,
        init_time: float = 0,
    ) -> None:
        super().__init__(
            base_angular_freq, 
            duration, 
            rotation_angle, 
            tgt_mat_elem,
            init_time,
        )

        self.sigma = sigma
        self.t_mid = self.init_time + self.duration/2

        # evaluate the effective pulse amplitude
        mean_amp_scale = _gaussian_mean_amp(duration, sigma)
        self.drive_amp = self.rotation_angle / mean_amp_scale / self.duration
        self.env_amp = self.drive_amp / np.abs(tgt_mat_elem)

        # set envelope to be 0 at the beginning and the end
        self.env_bias = _gaussian_function(0, self.duration/2, sigma, self.env_amp)

        # Bloch–Siegert shift
        sine_drive_amp = self.rotation_angle / self.duration
        freq_shift = (sine_drive_amp)**2 / self.base_angular_freq / 4
        self.drive_freq = self.base_angular_freq - freq_shift

    def envelope(self, t):
        """Only support scalar t"""
        if not isinstance(t, float):
            raise TypeError("The input time should be a float")

        return _gaussian_function(
            t,
            self.t_mid,
            self.sigma,
            self.env_amp
        ) - self.env_bias

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

# class DRAG(PulseBase):
#     def __init__(
#         self, 
#         base_angular_freq: float, 
#         duration: float, 
#         env_func: Callable,
#         d_env_func: Callable,
#         order: int = 3,
#         non_lin: float = 0, 
#         rotation_angle: float = np.pi, 
#         tgt_mat_elem: float = 1, 
#         leaking_mat_elem: float = np.sqrt(2), 
#         init_time: float = 0,
#     ) -> None:
#         if np.abs(non_lin) < 1 / duration:
#             raise ValueError("Non-linearity of the system should be specified and"
#                 "much larger than the pulse amplitude.")

#         super().__init__(
#             base_angular_freq, 
#             duration, 
#             rotation_angle, 
#             tgt_mat_elem, 
#             init_time
#         )

#         self.non_lin = non_lin
#         self.leaking_mat_elem = leaking_mat_elem
#         self.leaking_elem_ratio = np.abs(leaking_mat_elem / tgt_mat_elem)

class DRAGGaussian(PulseBase):
    def __init__(
        self, 
        base_angular_freq: float, 
        duration: float, 
        sigma: float, 
        non_lin: float = 0, 
        rotation_angle: float = np.pi, 
        tgt_mat_elem: float = 1, 
        leaking_mat_elem: float = np.sqrt(2), 
        init_time: float = 0
    ) -> None:
        if np.abs(non_lin) < 1 / duration:
            raise ValueError("Non-linearity of the system should be specified and"
                "much larger than the pulse amplitude.")

        super().__init__(
            base_angular_freq, 
            duration, 
            rotation_angle, 
            tgt_mat_elem, 
            init_time,
        )

        self.sigma = sigma
        self.non_lin = non_lin
        self.leaking_mat_elem = leaking_mat_elem
        self.leaking_elem_ratio = np.abs(leaking_mat_elem / tgt_mat_elem)
        self.t_mid = self.init_time + self.duration / 2

        # evaluate the effective pulse amplitude
        mean_amp_scale = _gaussian_mean_amp(duration, sigma)
        self.drive_amp = self.rotation_angle / mean_amp_scale / self.duration

        # set envelope to be 0 at the beginning and the end
        self.drive_env_bias = _gaussian_function(0, self.duration/2, sigma, self.drive_amp)

        self.reset()

    def reset(self):
        self.t_n_phase = [self.init_time, 0]

    def drive_freq_func(self, t, *args):
        """Only support scalar t"""
        if not isinstance(t, float):
            raise TypeError("The input time should be a float")

        eps_pi_2 = (_gaussian_function(t, self.t_mid, self.sigma, self.drive_amp) - self.drive_env_bias)**2
        detuning = (
            (self.leaking_elem_ratio**2 - 4) / (4 * self.non_lin) * eps_pi_2
            # - (self.leaking_elem_ratio**4 - 7*self.leaking_elem_ratio**2 + 12) / (16 * self.non_lin**3) * eps_pi_2**2
        )
        return self.base_angular_freq - detuning

    def phase(self, t):
        """Only support scalar t"""
        if not isinstance(t, float):
            raise TypeError("The input time should be a float")

        init_t, init_phase = self.t_n_phase
        phase = _phase_from_init(self.base_angular_freq, self.drive_freq_func, init_t, init_phase, t)
        self.t_n_phase[0] = t
        self.t_n_phase[1] = phase

        return phase

    def envelope(self, t):
        """Only support scalar t"""
        if not isinstance(t, float):
            raise TypeError("The input time should be a float")

        eps_pi = _gaussian_function(t, self.t_mid, self.sigma, self.drive_amp) - self.drive_env_bias
        eps_pi_dot = -_gaussian_function(t, self.t_mid, self.sigma, self.drive_amp) * (t - self.t_mid) / self.sigma**2

        eps_x = (
            eps_pi
            # + (self.leaking_elem_ratio**2 - 4) / (8 * self.non_lin**2) * eps_pi**3
            # - (13*self.leaking_elem_ratio**4 - 76*self.leaking_elem_ratio**2 + 112) / (128*self.non_lin**4) * eps_pi**5
        )
        eps_y = (
            - eps_pi_dot / self.non_lin
            # + 33*(self.leaking_elem_ratio**2 - 2) / (24*self.non_lin**3) * eps_pi**2 * eps_pi_dot
        )

        return eps_x, eps_y

    
    def __call__(self, t, *args, **kwargs):
        """Only support scalar t"""
        if not isinstance(t, float):
            raise TypeError("The input time should be a float")

        phase = self.phase(t)
        eps_x, eps_y = self.envelope(t)

        return (eps_x * np.cos(phase) + eps_y * np.sin(phase)) / np.abs(self.tgt_mat_elem)
