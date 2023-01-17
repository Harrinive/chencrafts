from chencrafts.version import version as __version__

import numpy as np

import matplotlib as mpl
import matplotlib_inline.backend_inline

import scqubits as scq

# set matplotlib 
mpl.rcParams = mpl.rcParamsDefault.copy()
matplotlib_inline.backend_inline.set_matplotlib_formats("png")

# set numpy 
np.set_printoptions(precision=5)

