from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes

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

class IntCmap():
    def __init__(self, total, cmap_name="rainbow"):
        self.total = total
        self.cmap_name = cmap_name

        self.cmap = colormaps[self.cmap_name]
        self.norm = plt.Normalize(0, self.total)
    
    def __call__(self, idx):
        return self.cmap(self.norm(idx))

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

        # contour
        if contour_levels > 0 and np.std(value) > 1e-14:
            try:
                cax_cont = ax.contour(X_mesh, Y_mesh, value, cmap="hsv", levels=contour_levels)
                fig.colorbar(cax_cont, ax=ax)
            except IndexError as err: # usually when no contour is found\
                print(f"In {key}, except IndexError: {err}")
                pass
        else:
            fig.colorbar(cax, ax=ax)

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