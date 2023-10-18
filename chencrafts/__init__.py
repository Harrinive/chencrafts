from chencrafts.version import version as __version__

import numpy as np

import scqubits as scq

# set matplotlib 
import matplotlib as mpl
import matplotlib.font_manager as mpl_font
import matplotlib_inline.backend_inline
from chencrafts.toolbox.plot import color_cyclers as _color_cyclers
mpl.rcParams = mpl.rcParamsDefault.copy()
# figure format
matplotlib_inline.backend_inline.set_matplotlib_formats("png")
# color cycle
mpl.rcParams["axes.prop_cycle"] = _color_cyclers["PGL"]
mpl.rcParams['text.usetex'] = False
# disable font warning message
font_selected = None
try:
    font_names = mpl_font.get_font_names()
    for font in ["IBM Plex Sans", "Roboto", "Arial", "Helvetica"]:
        if font in font_names:
            font_selected = font
            break
    if not font_selected:
        font_selected = "sans-serif"
except AttributeError:
    font_selected = "sans-serif"
mpl.rcParams["font.family"] = font_selected

# set numpy print options
np.set_printoptions(precision=6, linewidth=130)
