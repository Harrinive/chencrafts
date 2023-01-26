from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib import rcParams

import numpy as np

from typing import Dict, List, Tuple

from chencrafts.toolbox.data_processing import nd_interpolation

def filter(c, filter_name):
    if filter_name in ["translucent", "trans"]:
        r, g, b, a = c
        return [r, g, b, a * 0.2]
    elif filter_name in ["emphsize", "emph"]:
        r, g, b, a = c
        factor = 3
        return [r ** factor, g ** factor, b ** factor, a]

class Cmap():
    def __init__(
        self, 
        upper: float, 
        lower: float = 0, 
        cmap_name="rainbow"
    ):
        self.upper = upper
        self.lower = lower
        self.cmap_name = cmap_name

        self.cmap = colormaps[self.cmap_name]
        self.norm = plt.Normalize(self.lower, self.upper)
        self.mappable = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    
    def __call__(self, val):
        # return self.mappable.cmap(val)
        return self.cmap(self.norm(val))

def bar_plot_compare(
    var_list_dict: Dict[str, np.ndarray],
    x_ticks: List = None,
    ax = None,
    figsize = None, 
    dpi = None,
    x_tick_rotation = 45, 
):
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
    xy_label: List[str], 
    single_figsize=(3, 2.5), 
    cols=3, 
    place_a_point: Tuple[float, float]=(),     # plot a point in the figure
    show_value=False,                   # plot the number number near the destination of the trajectory  
    slc=slice(None),                            # slice the value stored in the dictionary before any processing
    slc_2d=slice(None),  # for zooming in the plots
    contour_levels=0,
    vmin=None,
    vmax=None,
    dpi=150,
    save_filename=None,
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
        ax: Axes = axs[ax_row, ax_col]
        value = full_value[slc][slc_2d]

        # base value
        try:
            cax = ax.pcolormesh(X_mesh, Y_mesh, value, vmin=vmin, vmax=vmax)
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

        ax_col += 1
        if ax_col % cols == 0:
            ax_col = 0
            ax_row += 1

    plt.tight_layout()

    if save_filename is not None:
        plt.savefig(save_filename)

    plt.show()
