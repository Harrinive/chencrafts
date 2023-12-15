from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import rcParams
from chencrafts.toolbox import bar_plot_compare

from typing import Callable, List, Dict

class ErrorChannel:
    def __init__(
        self, 
        name: str, 
        expression: Callable,
    ):
        self.name = name
        self.expression = expression

    def __call__(self, *args, **kwargs):
        return self.expression(*args, **kwargs)

# ##############################################################################
class ErrorRate:
    def __init__(
        self, 
    ):
        self.error_channels: dict = {}
        self.channel_enable_info: dict = {}

    def __call__(
        self,
        return_dict: bool = False,
        *args,
        **kwargs,
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """returns the total enabled error rate"""
        total_error = 0
        if return_dict:
            error_dict = {}

        for name, error_channel in self.error_channels.items():
            if not self.channel_enable_info[name]:
                continue
            error_rate = error_channel(*args, **kwargs)
            
            total_error += error_rate
            if return_dict:
                error_dict[name] = error_rate

        if return_dict:
            error_dict["total"] = total_error
            return error_dict
        else:
            return total_error

    def __getitem__(
        self,
        error_name: str | List[str],
    ) -> ErrorChannel | ErrorRate:
        """calculate the error rate from a single channel"""
        if isinstance(error_name, str):
            return self.error_channels[error_name]
        elif isinstance(error_name, list) or isinstance(error_name, tuple):
            new_rate = ErrorRate()
            for name in error_name:
                new_rate.add_existed_channel(self[name])
            return new_rate
    
    @property
    def num(self):
        return len(self.error_channels)

    @property
    def enable_num(self):
        return int(np.sum(list(self.channel_enable_info.values())))

    @property
    def error_names(self):
        return list(self.error_channels.keys())

    def add_channel(
        self,
        name: str, 
        expression: Callable,
    ):
        self.error_channels[name] = ErrorChannel(
            name,
            expression,
        )

        # update info
        self.channel_enable_info[name] = True

    def add_existed_channel(
        self,
        channel: ErrorChannel,
    ):  
        name = channel.name

        self.error_channels[name] = channel
        
        # update info
        self.channel_enable_info[name] = True

    def remove_channel(
        self,
        name
    ):
        try:
            del self.error_channels[name]
            del self.channel_enable_info[name]

        except KeyError:
            print(f"No error named {name}")

    def merge_channel(
        self,
        other_error_rate: ErrorRate,
    ):
        for err in other_error_rate.error_channels.values():
            self.add_existed_channel(err)

    def disable_channel(
        self,
        name: str
    ):
        if not self.channel_enable_info[name]:
            print(f"This channel [{name}] is already disabled")
        else:
            self.channel_enable_info[name] = False

    def enable_channel(
        self,
        name: str
    ):
        if self.channel_enable_info[name]:
            print(f"This channel [{name}] is already enabled")
        else:
            self.channel_enable_info[name] = True

    def pie_chart(
        self,
        ax = None,
        figsize = (6, 3),
        dpi = None,
        start_angle = 60,
        **kwargs,
    ):

        # error calculation
        error_dict: Dict = self(**kwargs, return_dict=True)
        error_dict.pop("total")
        error_rate_list = list(error_dict.values())
        total_error_rate = np.sum(error_rate_list)

        # color map
        cm_pie = colormaps["rainbow"]
        cm_pie_norm = plt.Normalize(0, self.enable_num)
        cmap_pie = lambda x: cm_pie(cm_pie_norm(x))

        # figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"), dpi=dpi)
            need_show = True

        wedges, texts = ax.pie(
            error_rate_list, 
            wedgeprops=dict(width=0.5), 
            startangle=start_angle, 
            colors=[cmap_pie(i) for i in range(self.enable_num)]
        )

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            err = error_rate_list[i]
            err_percent = err / total_error_rate * 100
            if err < 3e-8:
                continue
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(
                f"{self.error_names[i]}: {err_percent:.0f}%", 
                xy=(x, y), 
                xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, 
                **kw
            )

        if need_show:
            plt.tight_layout()
            plt.show()

    def bar_plot(
        self,
        ax = None,
        para_dicts = None,
        figsize = (6, 3), 
        dpi = None,
        labels = None,
        **kwargs
    ):
        # check instance type
        if isinstance(para_dicts, dict):
            para_dicts = [para_dicts]
            compare_num = 1
        elif isinstance(para_dicts, list):
            if isinstance(para_dicts[0], dict):
                compare_num = len(para_dicts)
            else:
                raise TypeError("Should input a single dictionary or a list of "
                    "dictionaries")
        elif para_dicts == None:
            para_dicts = [kwargs]
            compare_num = 1
        else:
            raise TypeError("Should input a single dictionary or a list of "
                "dictionaries")

        # calculate errors and obtain a label to each set of data
        errors = np.zeros((compare_num, self.enable_num))
        for i in range(compare_num):
            error_dict = self(**para_dicts[i], return_dict=True)
            del error_dict["total"]
            for j, err in enumerate(error_dict.values()):
                errors[i, j] = err

        label_list = []
        for i in range(compare_num):
            if labels is not None:
                total = np.sum(errors[i, :])
                label_list.append(labels[i] + f": {total:.2e}")
            else:
                total = np.sum(errors[i, :])
                label_list.append(f"E{i:d}: {total:.2e}")
        plot_dict = dict(zip(label_list, errors))

        
        # plot 
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            need_show = True

        bar_plot_compare(
            plot_dict,
            x_ticks = self.error_names,
            ax = ax,
            figsize = figsize, 
            dpi = dpi,
            x_tick_rotation = 45, 
        )
        ax.set_ylabel(r"Error Rate / GHz")

        if need_show:
            plt.tight_layout()
            plt.show()

# ##############################################################################
def qubit_g_to_e_prob(Gamma_up, Gamma_down, time):
    return Gamma_up / (Gamma_up + Gamma_down) * (1 - np.exp(-(Gamma_up + Gamma_down) * time))
def qubit_e_to_g_prob(Gamma_up, Gamma_down, time):
    return Gamma_down / (Gamma_up + Gamma_down) * (1 - np.exp(-(Gamma_up + Gamma_down) * time))

def manual_constr(g_sa, min_detuning, detuning_lower_bound, constr_amp, *args, **kwargs):
    detuning_undersize = 2 * np.pi * (g_sa * detuning_lower_bound - min_detuning)
    return constr_amp * detuning_undersize * np.heaviside(detuning_undersize, 0)

manual_constr = ErrorChannel(
    "manual_constr",
    manual_constr
)

# ------------------------------------------------------------------------------
basic_channels = ErrorRate()

basic_channels.add_channel(
    "multiple_photon_loss",
    lambda gamma_down, T_M, *args, **kwargs: 
        - np.log((1 + gamma_down * T_M) * np.exp(-gamma_down * T_M)) / T_M
)
basic_channels.add_channel(
    "photon_gain", 
    lambda gamma_up, *args, **kwargs: 
        gamma_up
)
basic_channels.add_channel(
    "anc_prepare", 
    lambda T_W, Gamma_up, Gamma_down, T_M, tau_FD, *args, **kwargs: 
        Gamma_up * (T_W + tau_FD) / T_M,
        # qubit_g_to_e_prob(Gamma_up, Gamma_down, T_W + tau_FD) / T_M
        # not correctly reveal the physics: the commented expression is the 
        # probability of the qubit being at the excited state.
        # While it's not the probability that the qubit is excited (maybe relaxed again).
        # this probability is more important as each excitation dephases the 
        # cavity
)
basic_channels.add_channel(
    "anc_relax_map", 
    lambda Gamma_down, chi_sa, T_M, *args, **kwargs: 
        np.pi * Gamma_down / (np.abs(chi_sa) * T_M)
)
basic_channels.add_channel(
    "anc_dephase_map", 
    lambda Gamma_phi, chi_sa, T_M, *args, **kwargs: 
        np.pi * Gamma_phi / (np.abs(chi_sa) * T_M)
)
basic_channels.add_channel(
    "cav_relax_map", 
    lambda gamma_down, chi_sa, T_M, *args, **kwargs: 
        np.pi * gamma_down / (np.abs(chi_sa) * T_M)
)
basic_channels.add_channel(
    "anc_relax_ro", 
    lambda gamma_down, tau_m, tau_FD, Gamma_down_ro, Gamma_down, *args, **kwargs: 
        gamma_down * Gamma_down_ro * tau_m + gamma_down * Gamma_down * tau_FD
)
basic_channels.add_channel(
    "anc_excite_ro", 
    lambda tau_m, Gamma_up_ro, T_M, *args, **kwargs: 
        Gamma_up_ro * tau_m / T_M
)
basic_channels.add_channel(
    "Kerr_dephase", 
    lambda gamma_down, K_s, T_M, n_bar_s, *args, **kwargs: 
        n_bar_s * gamma_down * K_s**2 * T_M**2 / 3
)
basic_channels.add_channel(
    "ro_infidelity", 
    lambda gamma_down, M_eg, M_ge, T_M, *args, **kwargs: 
        gamma_down * M_eg + M_ge / T_M
)
basic_channels.add_channel(
    "pi_pulse_error", 
    lambda gamma_down, n_bar_s, tau_p_eff, chi_sa, T_M, *args, **kwargs: 
        (2 + 4 * gamma_down * T_M) * (n_bar_s * chi_sa**2 * tau_p_eff**2 / T_M)
)
basic_channels.add_channel(
    "high_order_int", 
    lambda n_bar_s, chi_sa, T_M, chi_prime, *args, **kwargs: 
        n_bar_s * chi_prime**2 * np.pi**2 / (2 * chi_sa**2 * T_M),
)

# ------------------------------------------------------------------------------
flxn_hf_flx_channels = ErrorRate()
flxn_hf_flx_channels.merge_channel(basic_channels)
# ancilla initialization:
flxn_hf_flx_channels.remove_channel("anc_prepare")
flxn_hf_flx_channels.add_channel(
    "rot_after_excitation",
    lambda chi_sa, T_M, Gamma_up, n_bar_s, reset_no, *args, **kwargs: 
        np.min([n_bar_s * chi_sa**2 * T_M**2 / 12 / reset_no**2, np.ones_like(chi_sa)], axis=0) * Gamma_up
)
flxn_hf_flx_channels.add_channel(
    "anc_relax_reset_ro", 
    lambda Gamma_up, Gamma_down_ro, Gamma_down, tau_m, tau_FD, reset_no, T_M, *args, **kwargs: 
        qubit_g_to_e_prob(Gamma_up, Gamma_down, T_M / reset_no - tau_m - tau_FD) \
        * (Gamma_down_ro * tau_m + Gamma_down * tau_FD) \
        * reset_no / T_M
)
flxn_hf_flx_channels.add_channel(
    "anc_excite_reset_ro", 
    lambda Gamma_up, Gamma_up_ro, Gamma_down, tau_m, tau_FD, reset_no, T_M, *args, **kwargs: 
        (1 - qubit_g_to_e_prob(Gamma_up, Gamma_down, T_M / reset_no - tau_m - tau_FD)) \
        * (Gamma_up_ro * tau_m + Gamma_up * tau_FD) \
        * reset_no / T_M
)
flxn_hf_flx_channels.add_channel(
    "anc_reset_pulse_error", 
    lambda n_bar_s, tau_p_eff, chi_sa, T_M, Gamma_up, Gamma_down, reset_no, tau_m, tau_FD, *args, **kwargs: 
        qubit_g_to_e_prob(Gamma_up, Gamma_down, T_M / reset_no - tau_m - tau_FD) \
        * 4 * n_bar_s * chi_sa**2 * tau_p_eff**2 \
        * reset_no / T_M
)