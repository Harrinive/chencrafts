from __future__ import annotations

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import rcParams

from typing import Callable

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
        self.error_channels: dict = OrderedDict({})
        self.channel_enable_info: dict = OrderedDict({})

    def __call__(
        self,
        return_dict: bool = False,
        *args,
        **kwargs,
    ):
        """returns the total enabled error rate"""
        if return_dict:
            error_dict = OrderedDict({})
        else:
            total_error = 0

        for name, error_channel in self.error_channels.items():
            if not self.channel_enable_info[name]:
                continue
            error_rate = error_channel(*args, **kwargs)
            
            if return_dict:
                error_dict[name] = error_rate
            else:
                total_error += error_rate

        if return_dict:
            return error_dict
        else:
            return total_error

    def __getitem__(
        self,
        error_name
    ):
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
        error_dict = self(**kwargs, return_dict=True)
        error_rate_list = list(error_dict.values())
        total_error_rate = np.sum(error_rate_list)

        # color map
        cm_pie = colormaps["rainbow"]
        cm_pie_norm = plt.Normalize(0, self.enable_num)
        cmap_pie = lambda x: cm_pie(cm_pie_norm(x))

        # figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"), dpi=dpi)

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

        if ax is None:
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
            if isinstance(para_dicts[0], list):
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

        # calculate errors
        errors = np.zeros((compare_num, self.enable_num))

        for i in range(compare_num):
            error_dict = self(**para_dicts[i], return_dict=True)
            for j, err in enumerate(error_dict.values()):
                errors[i, j] = err

        # plot 
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        plot_width = 1 / (compare_num + 1)
        plot_x = np.linspace(0, self.enable_num-1, self.enable_num) + 0.5 * plot_width
        
        for i in range(compare_num):
            if labels is not None:
                total = np.sum(errors[i, :])
                lable_to_plot = labels[i] + f": {total:.2e}"
            else:
                total = np.sum(errors[i, :])
                lable_to_plot = f"E{i:d}: {total:.2e}"
                
            ax.bar(
                x = plot_x + i * plot_width, 
                height = errors[i],
                width = plot_width,
                align = "edge",
                label = lable_to_plot
            )
            
        ax.set_xticks(plot_x + plot_width * compare_num / 2)
        ax.set_xticklabels(
            self.error_names, 
            rotation=45, 
            rotation_mode="anchor", 
            horizontalalignment="right", 
            verticalalignment="top", 
            fontsize=rcParams["axes.labelsize"]
        )
        ax.set_ylabel(r"Error Rate / GHz")

        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

# ##############################################################################
default_channels = ErrorRate()

default_channels.add_channel(
    "multiple_photon_loss",
    lambda n_bar, T_M, kappa_s, *args, **kwargs: 
        - np.log((1 + kappa_s * n_bar * T_M) * np.exp(-n_bar * kappa_s * T_M)) / T_M
)
default_channels.add_channel(
    "photon_gain", 
    lambda n_bar, kappa_s, n_th, *args, **kwargs: 
        kappa_s * n_bar * n_th
)
default_channels.add_channel(
    "anc_prepare", 
    lambda tau_FD, sigma, T_W, Gamma_up, Gamma_down, T_M, *args, **kwargs: 
        Gamma_up / (Gamma_up + Gamma_down) 
        * (1 - np.exp(-(Gamma_up + Gamma_down) * (T_W + tau_FD + 8 * sigma))) / T_M
)
default_channels.add_channel(
    "anc_relax_map", 
    lambda Gamma_down, chi_sa, T_M, *args, **kwargs: 
        np.pi * Gamma_down / (np.abs(chi_sa) * T_M)
)
default_channels.add_channel(
    "anc_dephase_map", 
    lambda Gamma_phi, chi_sa, T_M, *args, **kwargs: 
        np.pi * Gamma_phi / (np.abs(chi_sa) * T_M)
)
default_channels.add_channel(
    "anc_relax_ro", 
    lambda n_bar, tau_m, tau_FD, Gamma_down_ro, kappa_s, *args, **kwargs: 
        n_bar * kappa_s * Gamma_down_ro * (tau_m + tau_FD)
)
default_channels.add_channel(
    "anc_excite_ro", 
    lambda tau_m, Gamma_up_ro, T_M, *args, **kwargs: 
        Gamma_up_ro * tau_m / T_M
)
default_channels.add_channel(
    "Kerr_dephase", 
    lambda n_bar, K_s, T_M, kappa_s, *args, **kwargs: 
        kappa_s * n_bar * K_s**2 * T_M**2 / 6
)
default_channels.add_channel(
    "ro_infidelity", 
    lambda n_bar, M_eg, M_ge, T_M, kappa_s, *args, **kwargs: 
        n_bar * kappa_s * M_eg + M_ge / T_M
)
default_channels.add_channel(
    "high_order_int", 
    lambda n_bar, chi_sa, T_M, chi_prime, *args, **kwargs: 
        n_bar * chi_prime**2 * np.pi**2 / (2 * chi_sa**2 * T_M),
)
default_channels.add_channel(
    "pi_pulse_error", 
    lambda n_bar, sigma, chi_sa, T_M, *args, **kwargs: 
        n_bar * chi_sa**2 * (4 * sigma)**2 / (2 * T_M)
)

def manual_constr(g_sa, min_detuning, detuning_lower_bound, constr_amp, *args, **kwargs):
    detuning_undersize = 2 * np.pi * (g_sa * detuning_lower_bound - min_detuning)
    return constr_amp * detuning_undersize * np.heaviside(detuning_undersize, 0)

manual_constr = ErrorChannel(
    "manual_constr",
    manual_constr
)

# ##############################################################################
class ErrorRateTmon(ErrorRate):
    def __init__(self):
        super().__init__()
        self.merge_channel(default_channels)
