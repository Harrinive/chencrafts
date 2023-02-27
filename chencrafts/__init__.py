from chencrafts.version import version as __version__

import numpy as np

import matplotlib as mpl
import matplotlib_inline.backend_inline

import scqubits as scq

from cycler import cycler

# set matplotlib 
mpl.rcParams = mpl.rcParamsDefault.copy()
matplotlib_inline.backend_inline.set_matplotlib_formats("png")
PGL_cycler = cycler(color = [
    "#0c2e6d", "#b63566", "#91adc2", "#e9c2c3", "#AEB358"
])
green_to_red_cycler = cycler(color = [
    "#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6", 
    "#ee9b00", "#ca6702", "#bb3e03", "#ae2012", "#9b2226"
])
sunset_cycler = cycler(color = [
    "#F8B195", "#F67280", "#C06C84", "#6C5B7B", "#355C7D"
])
hotel_70s_cycler = cycler(color = [
    "#448a9a", "#fb9ab6", "#e1cdd1", "#e1b10f", "#705b4c"
])
red_green_blue_purple_cycler = cycler(color = [
    "#e63946", "#a8dadc", "#457b9d", "#a7bb40", "#3d1645"
])
mpl.rcParams["axes.prop_cycle"] = PGL_cycler
mpl.rcParams['text.usetex'] = True


# set numpy 
np.set_printoptions(precision=5)

