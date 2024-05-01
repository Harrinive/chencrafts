from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib import rcParams
import matplotlib as mpl
from cycler import cycler
from itertools import cycle
import numpy as np

from typing import Dict, List, Tuple

from chencrafts.toolbox.data_processing import nd_interpolation

# color cyclers
color_palettes = dict(
    PGL = [
        "#0c2e6d", "#b63566", "#91adc2", "#e9c2c3", "#AEB358"],
    green_to_red = [
        "#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6", 
        "#ee9b00", "#ca6702", "#bb3e03", "#9b2226"],
    sunset = [
        "#F8B195", "#F67280", "#C06C84", "#6C5B7B", "#355C7D"],
    hotel_70s = [
        "#448a9a", "#fb9ab6", "#e1cdd1", "#e1b10f", "#705b4c"],
    blue_to_red = [
        "#e63946", "#a8dadc", "#457b9d", "#a7bb40", "#3d1645"],
    colorblind_1 = [    # from https://arxiv.org/abs/2107.02270
        "#3f90da", "#ffa90e", "#bd1f01", "#832db6", "#94a4a2", "#a96b59", 
        "#e76300", "#b9ac70", "#717581", "#92dadd",]
)
color_cyclers = dict([
    (key, cycler(color = color_palettes[key])) for key in color_palettes
])
color_iters = dict([
    (key, cycle(color_palettes[key])) for key in color_palettes
])
def set_color_cycler(cycler_name):
    mpl.rcParams["axes.prop_cycle"] = color_cyclers[cycler_name]

def remove_repeated_legend(ax=None):
    """remove repeated legend"""
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

def filter(c, filter_name):
    if filter_name in ["translucent", "trans"]:
        r, g, b, a = c
        return [r, g, b, a * 0.2]
    elif filter_name in ["emphsize", "emph"]:
        r, g, b, a = c
        factor = 3
        return [r ** factor, g ** factor, b ** factor, a]

# class Cmap():
#     def __init__(
#         self, 
#         upper: float, 
#         lower: float = 0, 
#         cmap_name="rainbow"
#     ):
#         self.upper = upper
#         self.lower = lower
#         self.cmap_name = cmap_name

#         self.cmap = colormaps[self.cmap_name]
#         self.norm = plt.Normalize(self.lower, self.upper)
#         self.mappable = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    
#     def __call__(self, val):
#         # return self.mappable.cmap(val)
#         return self.cmap(self.norm(val))
# useless now, can be simply replaced by plt.cm.get_cmap(cmap_name)


class PiecewiseLinearNorm:
    def __init__(
        self, 
        value_list: List[float] | np.ndarray,
        color_list: List[float] | np.ndarray,
        clip: bool = False, 
    ):
        assert len(value_list) == len(color_list), "value_list and color_list must have the same length."
        
        # Sorting the lists based on the value_list
        sorted_indices = np.argsort(value_list)
        self.value_list = np.array(value_list)[sorted_indices]
        self.color_list = np.array(color_list)[sorted_indices]
        self.clip = clip

    def __call__(self, value: float) -> float:
        if self.clip:
            if value < self.value_list[0]:
                return self.color_list[0]
            elif value > self.value_list[-1]:
                return self.color_list[-1]
        
        return np.interp(value, self.value_list, self.color_list)

    def inverse(self, color: float) -> float:
        return np.interp(color, self.color_list, self.value_list)


def bar_plot_compare(
    var_list_dict: Dict[str, np.ndarray],
    x_ticks: List = None,
    ax = None,
    figsize = None, 
    dpi = None,
    x_tick_rotation = 45, 
):
    """
    The var_list_dict should be {labels: a series of value to compare}. 

    Note: such function can be realized in pandas by DataFrame.plot(kind="bar")
    """
    # plot 
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    x_len = len(x_ticks)
    for key, val in var_list_dict.items():
        assert len(x_ticks) == len(val), (f"x_lables should have the same length with"
        f"the data to be plotted, exception occurs for {key}")

    compare_num = len(var_list_dict)
    plot_width = 1 / (compare_num + 1)
    plot_x = np.linspace(0, x_len-1, x_len) + 0.5 * plot_width
    
    for i, (key, val) in enumerate(var_list_dict.items()):
            
        ax.bar(
            x = plot_x + i * plot_width, 
            height = val,
            width = plot_width,
            align = "edge",
            label = key
        )
            
        ax.set_xticks(plot_x + plot_width * compare_num / 2)
        ax.set_xticklabels(
            x_ticks, 
            rotation=x_tick_rotation, 
            rotation_mode="anchor", 
            horizontalalignment="right", 
            verticalalignment="top", 
            fontsize=rcParams["axes.labelsize"]
        )

        ax.legend()

def plot_dictionary_2d(
    dict: Dict[str, np.ndarray], 
    xy_mesh: List[np.ndarray],
    xy_label: List[str] = ["", ""], 
    single_figsize = (3, 2.5), 
    cols = 3, 
    place_a_point: Tuple[float, float] = tuple(),     # plot a point in the figure
    show_value = False,                   # plot the number number near the destination of the trajectory  
    slc = slice(None),                            # slice the value stored in the dictionary before any processing
    slc_2d = slice(None),  # for zooming in the plots
    contour_levels = 0,
    cmap = "viridis",
    vmin = None,
    vmax = None,
    dpi = 150,
    save_filename = None,
):
    """
    this function plot meshes from a dictionary

    place_a_point should be (x, y)

    """

    rows = np.ceil(len(dict) / cols).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*single_figsize[0], rows*single_figsize[1]), dpi=dpi)

    X_mesh, Y_mesh = xy_mesh
    X_mesh, Y_mesh = X_mesh[slc][slc_2d], Y_mesh[slc][slc_2d]
    x_name, y_name = xy_label

    ax_row, ax_col = 0, 0
    for key, full_value in dict.items():
        if rows == 1:
            ax: Axes = axs[ax_col]
        else:
            ax: Axes = axs[ax_row, ax_col]
        value = full_value[slc][slc_2d]

        # base value
        try:
            cax = ax.pcolormesh(X_mesh, Y_mesh, value, vmin=vmin, vmax=vmax, cmap=cmap)
        except (ValueError, IndexError):
            print("Error, Value to be plotted has the shape", value.shape, ", key: ", key)
        # except TypeError:
        #     print("TypeError, key: ", key, "value: ", value, "X, Y mesh", X_mesh, Y_mesh)
        fig.colorbar(cax, ax=ax)

        # contour
        if contour_levels > 0 and np.std(value) > 1e-14:
            try:
                CS = ax.contour(X_mesh, Y_mesh, value, cmap="hsv", levels=contour_levels)
                ax.clabel(CS, inline=True, fontsize=7)
                # fig.colorbar(cax_cont, ax=ax)
            except IndexError as err: # usually when no contour is found\
                print(f"In {key}, except IndexError: {err}")
                pass

        # trajectory
        if place_a_point != ():
            px, py = place_a_point
            ax.scatter(px, py, c="white", s=8)
            if show_value:
                interp = nd_interpolation(
                    [X_mesh, Y_mesh],
                    value
                )
                val = interp(px, py)
                if np.abs(val) >= 1e-2 and np.abs(val) < 1e2: 
                    text = f"  {val:.3f}"
                else:
                    text = f"  {val:.1e}"
                ax.text(px, py, text, ha="left", va="center", c="white", fontsize=7)

        # labels 
        ax.set_title(key)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid()

        ax_col += 1
        if ax_col % cols == 0:
            ax_col = 0
            ax_row += 1

    plt.tight_layout()

    if save_filename is not None:
        plt.savefig(save_filename)

    plt.show()
