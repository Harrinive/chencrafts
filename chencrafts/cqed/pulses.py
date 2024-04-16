import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from warnings import warn
from typing import Tuple, List

# ##############################################################################
class PulseBase:
    """
    """
    def __init__(
        self,
        base_angular_freq: float,
        duration: float,
        rotation_angle: float,
        tgt_mat_elem: float, 
        init_time: float = 0,
        init_phase: float = 0,
    ) -> None:
        """
        Parameters
        ----------
        base_angular_freq : float
            The angular frequency of the desired transitions of the undriven Hamiltonian.
        duration : float
            The duration of the pulse.
        rotation_angle : float
            The desired rotation angle that the pulse want to achieve. Leave it alone if you
            want to directly specify the drive amplitude later by setting pulse.drive_amp 
            = <value>.
        tgt_mat_elem : float
            The matrix element of the drive operator for the transition.
        init_time : float, optional
            The initial time of the pulse, by default 0.

        """
        self.base_angular_freq = base_angular_freq
        self.duration = duration
        self._rotation_angle = rotation_angle

        if np.abs(tgt_mat_elem) < 1e-10:
            warn("The matrix element of the drive operator is too small, "
                "which may cause numerical instability.")

        self.tgt_mat_elem = tgt_mat_elem
        self.init_time = init_time

        self._rabi_amp = rotation_angle / duration 
        self._drive_amp = self._rabi_amp / np.abs(tgt_mat_elem)
        self.drive_freq = self.base_angular_freq

        self.custom_envelope_I = None
        self.custom_envelope_Q = None

        self.init_phase = init_phase

        self._exp_only = False

    @property
    def drive_amp(self):
        return self._drive_amp
    
    @drive_amp.setter
    def drive_amp(self, new_drive_amp):
        self._drive_amp = new_drive_amp
        self._rabi_amp = self._drive_amp * np.abs(self.tgt_mat_elem)

    @property
    def rabi_amp(self):
        return self._rabi_amp
    
    @rabi_amp.setter
    def rabi_amp(self, new_rabi_amp):
        self._rabi_amp = new_rabi_amp
        self._drive_amp = self._rabi_amp / np.abs(self.tgt_mat_elem)

    def envelope_I(self, t):
        """Only support scalar t"""
        self._check_input_t(t)

        if self.custom_envelope_I is not None:
            return self.custom_envelope_I(t)

        return self._drive_amp
    
    def envelope_Q(self, t):
        """Only support scalar t"""
        self._check_input_t(t)

        if self.custom_envelope_Q is not None:
            return self.custom_envelope_Q(t)

        return 0

    def phase(self, t):
        """Only support scalar t"""
        self._check_input_t(t)

        t_bias = t - self.init_time
        phase = self.drive_freq * t_bias + self.init_phase
        return phase
    
    @property
    def exp_only(self):
        return self._exp_only
    
    @exp_only.setter
    def exp_only(self, value: bool):
        warn("When called, this pulse object will return a complex exponential function."
              "pulse(t) + pulse(t).conj() will give the full pulse.\n")
        self._exp_only = value

    def __call__(self, t, *args) -> float:
        """
        Only support scalar t
        args are here for qutip solver
        """
        self._check_input_t(t)

        env_I = self.envelope_I(t)
        env_Q = self.envelope_Q(t)
        phase = self.phase(t)

        if self.exp_only:
            return env_I * np.exp(1j * phase) / 2 + env_Q * np.exp(1j * phase) / 2j
        return env_I * np.cos(phase) + env_Q * np.sin(phase)

    def _check_input_t(self, t):
        if not isinstance(t, float | int):
            raise TypeError("The input time should be a 0d number")

    def plot(self, t_list = None, env_only = False, ax = None):
        if t_list is None:
            t_list = np.linspace(self.init_time, self.init_time + self.duration, 100)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(t_list, [self.envelope_I(t) for t in t_list], label = "I")
        ax.plot(t_list, [self.envelope_Q(t) for t in t_list], label = "Q")

        if not env_only:
            ax.plot(t_list, [self(t) for t in t_list], label="full")

        ax.legend()

        return ax 

    def discretized_IQ(
        self, 
        time_steps: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Discretize the pulse into a list of I and Q values.

        Parameters
        ----------
        time_steps : int, optional
            The number of time steps to discretize the pulse, by default to be [0, 1, ..., duration]. 

        Returns
        -------
        time_list, I_list, Q_list : np.ndarray
            They are all 1d arrays of the same length.
        """
        if time_steps is None:
            time_steps = int(self.duration)
        t_list = np.linspace(self.init_time, self.init_time + self.duration, time_steps + 1)

        # sample the middle point of each time interval
        t_mid_list = t_list[:-1] + (t_list[1] - t_list[0]) / 2
        I_list = np.array([self.envelope_I(t) for t in t_mid_list])
        Q_list = np.array([self.envelope_Q(t) for t in t_mid_list])
        
        return t_list[:-1], I_list, Q_list


# ##############################################################################
class GeneralPulse(PulseBase):
    """
    A class for generating a general pulse whose envelope can be customized. By 
    default, it'll use a square envelope. 

    To define a pulse object:
    -------------------------
        - A square pulse will be generated by specifying the rotation_angle, duration and 
        the drive operator's matrix element. the drive amplitude will be automatically determined by 
            `rotation_angle / duration / np.abs(tgt_mat_elem)`
        - You can also specify the drive amplitude by setting either `pulse.drive_amp` or `pulse.rabi_amp`.
        `rabi_amp` is the overall coefficient of driven Hamiltonian (including the matrix element),
        while `drive_amp` does not include the matrix element. They are correlated and adjusted 
        simutaneously.
        - Users can also specify an arbitrary envelope function by setting 
        `pulse.custom_envelope_I` and `pulse.custom_envelope_Q`, which should be a function of `t`.
        - After defining the pulse, modifying any of the parameters will make the rotation
        angle invalid. It's better to redefine the pulse.

    To use the pulse object:
    ------------------------
        - `pulse(t)` to get the pulse value at time `t`
        - `pulse.envelope_I(t)` and `pulse.envelope_Q(t)` to get the envelope for two quadratures
        - `pulse.discretized_IQ()` to get the discretized I and Q values, which is useful for
        feeding into the AWG.
    """
    def __init__(
        self,
        base_angular_freq: float,
        duration: float = 1.0,
        rotation_angle: float = np.pi,
        tgt_mat_elem: float = 1.0, 
        init_time: float = 0.0,
        init_phase: float = 0.0,
        with_freq_shift: bool = True,
    ) -> None:
        """
        A class for generating a general pulse whose envelope can be customized. By 
        default, it'll use a square envelope. 

        To define a pulse object:
            - A square pulse will be generated by specifying the rotation_angle, duration and 
            the drive operator's matrix element. the drive amplitude will be automatically determined by 
                `rotation_angle / duration / np.abs(tgt_mat_elem)`
            - You can also specify the drive amplitude by setting either `pulse.drive_amp` or `pulse.rabi_amp`.
            `rabi_amp` is the overall coefficient of driven Hamiltonian (including the matrix element),
            while `drive_amp` does not include the matrix element. They are correlated and adjusted 
            simutaneously.
            - Users can also specify an arbitrary envelope function by setting 
            `pulse.custom_envelope_I` and `pulse.custom_envelope_Q`, which should be a function of `t`.
            - After defining the pulse, modifying any of the parameters will make the rotation
            angle invalid. It's better to redefine the pulse.

        To use the pulse object:
            - `pulse(t)` to get the pulse value at time `t`
            - `pulse.envelope_I(t)` and `pulse.envelope_Q(t)` to get the envelope for two quadratures
            - `pulse.discretized_IQ()` to get the discretized I and Q values, which is useful for
            feeding into the AWG.
                
        Parameters
        ----------
        base_angular_freq : float
            The angular frequency of the desired transitions of the undriven Hamiltonian.
        duration : float
            The duration of the pulse.
        rotation_angle : float, optional
            The desired rotation angle that the pulse want to achieve, this will automatically
            determine the drive amplitude for the user. By default np.pi.
            Leave it alone if you want to directly specify the drive amplitude later by 
            setting pulse.drive_amp = <value>.
        tgt_mat_elem : float, optional
            The matrix element of the drive operator for the transition. By default 1.0.
        init_time : float, optional
            The initial time of the pulse, by default 0.
        init_phase : float, optional    
            The initial phase of the pulse, by default 0.
        with_freq_shift : bool, optional
            Whether to include the Bloch-Siegert shift, by default True.
        """
        super().__init__(
            base_angular_freq, 
            duration, 
            rotation_angle,
            tgt_mat_elem, 
            init_time,
            init_phase,
        )

        # modify the drive freq with the Bloch–Siegert shift
        self.with_freq_shift = with_freq_shift
        self.drive_freq = self.base_angular_freq - self._bloch_siegert_shift()

    def _bloch_siegert_shift(self):
        if self.with_freq_shift:
            freq_shift = self._rabi_amp**2 / self.base_angular_freq / 4
            return freq_shift
        else:
            return 0
    
    @PulseBase.rabi_amp.setter
    def rabi_amp(self, new_rabi_amp):
        super(GeneralPulse, GeneralPulse).rabi_amp.__set__(self, new_rabi_amp)
        self.drive_freq = self.base_angular_freq - self._bloch_siegert_shift()

    @PulseBase.drive_amp.setter
    def drive_amp(self, new_drive_amp):
        super(GeneralPulse, GeneralPulse).drive_amp.__set__(self, new_drive_amp)
        self.drive_freq = self.base_angular_freq - self._bloch_siegert_shift()


# ##############################################################################
def _gaussian_function(t, t_mid, sigma, amp=1):
    return amp * (np.exp(-(t - t_mid)**2 / 2 / sigma**2))

def _gaussian_mean_amp(duration, sigma):
    """Calculate the mean amplitude of a gaussian function in the time interval [0, duration]"""
    half_duration = duration / 2
    bias = _gaussian_function(0, half_duration, sigma, 1)
    mean_amp_scale = odeint(
        lambda t, *args: _gaussian_function(t, half_duration, sigma, 1) - bias,
        y0 = 0,
        t = [0, half_duration],
        tfirst = True,
    )[-1, 0] / half_duration

    return mean_amp_scale

class Gaussian(PulseBase):
    """
    Pulse with Gaussian envelope. 

    To define a pulse object:
    -------------------------
        - A gaussian pulse will be generated by specifying the rotation_angle, duration, 
        sigma and 
        the drive operator's matrix element. the drive amplitude will be automatically
        determined to achieve the desired rotation angle.
        - You can also specify the drive amplitude by setting either `pulse.drive_amp` or `pulse.rabi_amp`.
        `rabi_amp` is the overall coefficient of driven Hamiltonian (including the matrix element),
        while `drive_amp` does not include the matrix element. They are correlated and adjusted 
        simutaneously.
        - After defining the pulse, modifying any of the parameters will make the rotation
        angle invalid. It's better to redefine the pulse.

    To use the pulse object:
    ------------------------
        - `pulse(t)` to get the pulse value at time `t`
        - `pulse.envelope_I(t)` and `pulse.envelope_Q(t)` to get the envelope for two quadratures
        - `pulse.discretized_IQ()` to get the discretized I and Q values, which is useful for
        feeding into the AWG.
    """
    def __init__(
        self, 
        base_angular_freq: float, 
        duration: float, 
        sigma: float, 
        rotation_angle: float = np.pi, 
        tgt_mat_elem: float = 1.0,
        init_time: float = 0,
        init_phase: float = 0,
        with_freq_shift: bool = False,
    ) -> None:
        """
        Pulse with Gaussian envelope. 

        To define a pulse object:
            - A gaussian pulse will be generated by specifying the rotation_angle, duration, 
            sigma and 
            the drive operator's matrix element. the drive amplitude will be automatically
            determined to achieve the desired rotation angle.
            - You can also specify the drive amplitude by setting either `pulse.drive_amp` or `pulse.rabi_amp`.
            `rabi_amp` is the overall coefficient of driven Hamiltonian (including the matrix element),
            while `drive_amp` does not include the matrix element. They are correlated and adjusted 
            simutaneously.
            - After defining the pulse, modifying any of the parameters will make the rotation
            angle invalid. It's better to redefine the pulse.

        To use the pulse object:
            - `pulse(t)` to get the pulse value at time `t`
            - `pulse.envelope_I(t)` and `pulse.envelope_Q(t)` to get the envelope for two quadratures
            - `pulse.discretized_IQ()` to get the discretized I and Q values, which is useful for
            feeding into the AWG.
        
        Parameters
        ----------
        base_angular_freq : float
            The angular frequency of the desired transitions of the undriven Hamiltonian.
        duration : float
            The duration of the pulse.
        sigma : float
            The standard deviation of the Gaussian envelope.
        rotation_angle : float, optional
            The desired rotation angle that the pulse want to achieve, this will automatically
            determine the drive amplitude for the user. By default np.pi.
            Leave it alone if you want to directly specify the drive amplitude later by 
            setting pulse.drive_amp = <value>.
        tgt_mat_elem : float, optional
            The matrix element of the drive operator for the transition. By default 1.0.
        init_time : float, optional
            The initial time of the pulse, by default 0.    
        init_phase : float, optional
            The initial phase of the pulse, by default 0.
        with_freq_shift : bool, optional    
            Whether to include the Bloch-Siegert shift, by default True. Currently it's 
            kind of inaccurate and don't use it!
        """
        super().__init__(
            base_angular_freq, 
            duration, 
            rotation_angle, 
            tgt_mat_elem,
            init_time,
            init_phase,
        )

        self.sigma = sigma
        self.t_mid = self.init_time + self.duration/2

        # evaluate the effective pulse amplitude
        # mean_amp_scale is the average amplitude of the gaussian in the time interval
        mean_amp_scale = _gaussian_mean_amp(duration, sigma)
        self._rabi_amp = self._rotation_angle / mean_amp_scale / self.duration
        self._drive_amp = self._rabi_amp / np.abs(tgt_mat_elem)

        # set envelope to be 0 at the beginning and the end
        self._drive_amp_bias = _gaussian_function(0, self.duration/2, sigma, self._drive_amp)

        # Bloch–Siegert shift
        self.with_freq_shift = with_freq_shift
        self.drive_freq = self.base_angular_freq - self._bloch_siegert_shift()

    def _bloch_siegert_shift(self):
        if self.with_freq_shift:
            sine_rabi_amp = self._rotation_angle / self.duration
            freq_shift = (sine_rabi_amp)**2 / self.base_angular_freq / 4
            return freq_shift
        else:
            return 0
        
    @PulseBase.rabi_amp.setter
    def rabi_amp(self, new_rabi_amp):
        super(Gaussian, Gaussian).rabi_amp.__set__(self, new_rabi_amp)
        self.drive_freq = self.base_angular_freq - self._bloch_siegert_shift()
        self._drive_amp_bias = _gaussian_function(0, self.duration/2, self.sigma, self._drive_amp)

    @PulseBase.drive_amp.setter
    def drive_amp(self, new_drive_amp):
        super(Gaussian, Gaussian).drive_amp.__set__(self, new_drive_amp)
        self.drive_freq = self.base_angular_freq - self._bloch_siegert_shift()
        self._drive_amp_bias = _gaussian_function(0, self.duration/2, self.sigma, self._drive_amp)

    def envelope_I(self, t):
        """Only support scalar t"""
        self._check_input_t(t)

        return _gaussian_function(
            t,
            self.t_mid,
            self.sigma,
            self._drive_amp
        ) - self._drive_amp_bias
        

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
    """
    A class for generating a DRAG pulse with gaussian envelope. By default, it'll use 

    To define a pulse object:
    -------------------------
        - A DRAG pulse will be generated by specifying the rotation_angle, duration, 
        sigma, non-linearity of the undriven system and 
        the drive operator's matrix element. the drive amplitude will be automatically 
        determined to achieve the desired rotation angle.
        - You can also specify the drive amplitude by setting either `pulse.drive_amp` or `pulse.rabi_amp`.
        `rabi_amp` is the overall coefficient of driven Hamiltonian (including the matrix element),
        while `drive_amp` does not include the matrix element. They are correlated and adjusted 
        simutaneously.
        - After defining the pulse, modifying any of the parameters will make the rotation
        angle invalid. It's better to redefine the pulse.

    To use the pulse object:
    ------------------------
        - `pulse(t)` to get the pulse value at time `t`
        - `pulse.envelope_I(t)` and `pulse.envelope_Q(t)` to get the envelope for two quadratures
        - `pulse.detuning(t)` to get the detuning of the drive frequency from the system frequency
        - `pulse.discretized_IQ()` to get the discretized I and Q values, which is useful for
        feeding into the AWG.
    """
    def __init__(
        self, 
        base_angular_freq: float, 
        duration: float, 
        sigma: float, 
        non_lin: float = np.inf, 
        order: int = 5,
        rotation_angle: float = np.pi, 
        tgt_mat_elem: float = 1, 
        leaking_mat_elem: float = np.sqrt(2), 
        init_time: float = 0,
        init_phase: float = 0,
        dynamic_drive_freq: bool = False,
    ) -> None:
        """
        A class for generating a DRAG pulse with gaussian envelope. By default, it'll use 

        To define a pulse object:
            - A DRAG pulse will be generated by specifying the rotation_angle, duration, 
            sigma, non-linearity of the undriven system and 
            the drive operator's matrix element. the drive amplitude will be automatically 
            determined to achieve the desired rotation angle.
            - You can also specify the drive amplitude by setting either `pulse.drive_amp` or `pulse.rabi_amp`.
            `rabi_amp` is the overall coefficient of driven Hamiltonian (including the matrix element),
            while `drive_amp` does not include the matrix element. They are correlated and adjusted 
            simutaneously.
            - After defining the pulse, modifying any of the parameters will make the rotation
            angle invalid. It's better to redefine the pulse.

        To use the pulse object:
            - `pulse(t)` to get the pulse value at time `t`
            - `pulse.envelope_I(t)` and `pulse.envelope_Q(t)` to get the envelope for two quadratures
            - `pulse.detuning(t)` to get the detuning of the drive frequency from the system frequency
            - `pulse.discretized_IQ()` to get the discretized I and Q values, which is useful for
            feeding into the AWG.

        Parameters
        ----------
        base_angular_freq : float
            The angular frequency of the desired transitions of the undriven Hamiltonian.
        duration : float
            The duration of the pulse.
        sigma : float   
            The standard deviation of the Gaussian envelope.
        non_lin : float, optional
            The non-linearity of the undriven hamiltonian. By default np.inf.
        order : int, optional
            The order of the DRAG correction. Can be 2 or 5. By default 5.
        rotation_angle : float, optional
            The desired rotation angle that the pulse want to achieve. By default np.pi.
        tgt_mat_elem : float, optional
            The matrix element of the drive operator for the transition. By default 1.0.
        leaking_mat_elem : float, optional
            The matrix element of the drive operator for the leaking transition. By default np.sqrt(2).
        init_time : float, optional
            The initial time of the pulse, by default 0.
        init_phase : float, optional
            The initial phase of the pulse, by default 0.
        dynamic_drive_freq : bool, optional
            Whether to include the time-dependent drive frequency to the pulse. By default False.
            If True, the phase of the pulse will be calculated by integrating the drive frequency
            from the initial time. In this case, whenever the pulse is finished, users 
            should call `pulse.reset()` to reset the integration. 
            If False, the drive frequency will be time independent.
            Users should put `pulse.detuning(t)` into the Hamiltonian to shift the 
            system frequency.
        """
        
        if order != 2 and order != 5:
            raise ValueError(f"Currently the code only support order = 2 or 5!")
        else:
            self.order = order

        if np.abs(non_lin) < 1 / duration:
            raise ValueError("Non-linearity of the system should be specified and"
                "much larger than the pulse amplitude.")

        super().__init__(
            base_angular_freq, 
            duration, 
            rotation_angle, 
            tgt_mat_elem, 
            init_time,
            init_phase,
        )

        self.sigma = sigma
        self.non_lin = non_lin
        self.leaking_elem_ratio = np.abs(leaking_mat_elem / tgt_mat_elem)
        self.t_mid = self.init_time + self.duration / 2

        # evaluate the effective pulse amplitude
        mean_amp_scale = _gaussian_mean_amp(duration, sigma)
        self._rabi_amp = self._rotation_angle / mean_amp_scale / self.duration
        self._drive_amp = self._rabi_amp / np.abs(tgt_mat_elem)

        # set envelope to be 0 at the beginning and the end
        self._env_bias = _gaussian_function(0, self.duration/2, sigma, self._drive_amp)

        # dynamic drive frequency
        self.dynamic_drive_freq = dynamic_drive_freq

        self.reset()

    @PulseBase.drive_amp.setter
    def drive_amp(self, new_rabi_amp):
        super(DRAGGaussian, DRAGGaussian).drive_amp.__set__(self, new_rabi_amp)
        self._env_bias = _gaussian_function(0, self.duration/2, self.sigma, self._drive_amp)

    @PulseBase.rabi_amp.setter
    def rabi_amp(self, new_rabi_amp):
        super(DRAGGaussian, DRAGGaussian).rabi_amp.__set__(self, new_rabi_amp)
        self._env_bias = _gaussian_function(0, self.duration/2, self.sigma, self._drive_amp)

    def reset(self):
        self.t_n_phase = [self.init_time, self.init_phase]

    def detuning(self, t):
        """
        system frequency - drive frequency
        Only support scalar t
        """
        if not isinstance(t, float):
            raise TypeError("The input time should be a float")

        eps_pi_2 = (_gaussian_function(t, self.t_mid, self.sigma, self._drive_amp) - self._env_bias)**2
        detuning = (self.leaking_elem_ratio**2 - 4) / (4 * self.non_lin) * eps_pi_2
        if self.order == 5:
            detuning -= (self.leaking_elem_ratio**4 - 7*self.leaking_elem_ratio**2 + 12) \
                / (16 * self.non_lin**3) * eps_pi_2**2

        return detuning
    
    def _drive_freq(self, t, *args):
        """only support scalar t"""
        return self.base_angular_freq - self.detuning(t)

    def phase(self, t):
        """Only support scalar t"""
        self._check_input_t(t)

        if self.dynamic_drive_freq:
            return super().phase(t)

        init_t, init_phase = self.t_n_phase
        phase = _phase_from_init(self.base_angular_freq, self._drive_freq, init_t, init_phase, t)
        self.t_n_phase[0] = t
        self.t_n_phase[1] = phase

        return phase

    def envelope_I(self, t):
        """Only support scalar t"""
        self._check_input_t(t)

        eps_pi = _gaussian_function(t, self.t_mid, self.sigma, self._drive_amp) - self._env_bias

        eps_x = eps_pi
        if self.order == 5:
            eps_x += (
                + (self.leaking_elem_ratio**2 - 4) / (8 * self.non_lin**2) * eps_pi**3
                - (13*self.leaking_elem_ratio**4 - 76*self.leaking_elem_ratio**2 + 112) / (128*self.non_lin**4) * eps_pi**5
            )

        return eps_x
    
    def envelope_Q(self, t):
        """Only support scalar t"""
        self._check_input_t(t)

        eps_pi = _gaussian_function(t, self.t_mid, self.sigma, self._drive_amp) - self._env_bias
        eps_pi_dot = -_gaussian_function(t, self.t_mid, self.sigma, self._drive_amp) * (t - self.t_mid) / self.sigma**2

        eps_y = - eps_pi_dot / self.non_lin
        if self.order == 5:
            eps_y += 33*(self.leaking_elem_ratio**2 - 2) / (24*self.non_lin**3) * eps_pi**2 * eps_pi_dot

        return eps_y
    

# ##############################################################################
class Interpolated(PulseBase):
    """
    A class for generating a pulse with envelope defined by a series of data points for 
    I and Q quadratures. The envelope will be interpolated from the data points.

    To define a pulse object:
    -------------------------
        - A pulse will be generated by specifying data points for the two quadratures.

    To use the pulse object:
    ------------------------
        - `pulse(t)` to get the pulse value at time `t`
        - `pulse.envelope_I(t)` and `pulse.envelope_Q(t)` to get the envelope for two quadratures
        - `pulse.discretized_IQ()` to get the discretized I and Q values, which is useful for
        feeding into the AWG.
    """
    def __init__(
        self, 
        base_angular_freq: float, 
        duration: float, 
        # rotation_angle: float, 
        # tgt_mat_elem: float, 
        init_time: float = 0,
        init_phase: float = 0,
        I_data: np.ndarray | List = np.array([1, 1]),
        Q_data: np.ndarray | List = np.array([0, 0]),
        interpolation_mode: str = "linear",
    ) -> None:
        """
        A class for generating a pulse with envelope defined by a series of data points for 
        I and Q quadratures. The envelope will be interpolated from the data points.

        To define a pulse object:
            - A pulse will be generated by specifying data points for the two quadratures.

        To use the pulse object:
            - `pulse(t)` to get the pulse value at time `t`
            - `pulse.envelope_I(t)` and `pulse.envelope_Q(t)` to get the envelope for two quadratures
            - `pulse.discretized_IQ()` to get the discretized I and Q values, which is useful for
            feeding into the AWG.

        Parameters
        ----------
        base_angular_freq : float
            The angular frequency of the desired transitions of the undriven Hamiltonian.
        duration : float
            The duration of the pulse.
        init_time : float, optional
            The initial time of the pulse, by default 0.
        init_phase : float, optional
            The initial phase of the pulse, by default 0.
        I_data : np.ndarray, optional
            The data points for the I quadrature, should include the initial and final points.
            By default it's np.array([1, 1]), indicating a square pulse with amplitude 1.
        Q_data : np.ndarray, optional
            The data points for the Q quadrature, should include the initial and final points.
            By default it's np.array([0, 0]), indicating no Q quadrature.
        interpolation_mode : str
            The mode of interpolation. Can be "linear", "nearest", "zero", "slinear",
            See scipy.interpolate.interp1d for details. 
        """
        super().__init__(
            base_angular_freq, 
            duration, 
            np.pi,  # useless
            1,      # useless
            init_time,
            init_phase,
        )
        
        self.interpolation_mode = interpolation_mode

        self.I_t_list = np.linspace(self.init_time, self.init_time + self.duration, len(I_data))
        self.Q_t_list = np.linspace(self.init_time, self.init_time + self.duration, len(Q_data))
        self.I_data = np.array(I_data)
        self.Q_data = np.array(Q_data)
        self.custom_envelope_I = self._points_to_func(self.I_t_list, self.I_data)
        self.custom_envelope_Q = self._points_to_func(self.Q_t_list, self.Q_data)

    def _points_to_func(self, t_list, data):
        return interp1d(
            t_list,
            data,
            kind = self.interpolation_mode,
            fill_value = "extrapolate",
        )
    
    def plot(self, t_list = None, env_only = False, ax=None):
        ax = super().plot(t_list, env_only, ax)

        ax.scatter(self.I_t_list, self.I_data)
        ax.scatter(self.Q_t_list, self.Q_data)

        return ax