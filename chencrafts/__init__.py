from chencrafts.version import version as __version__

import numpy as np

import matplotlib as mpl
import matplotlib_inline.backend_inline

import scqubits as scq

# set matplotlib 
mpl.rcParams = mpl.rcParamsDefault.copy()
matplotlib_inline.backend_inline.set_matplotlib_formats("png")

# set scqubits
scq.set_units('GHz')
scq.settings.T1_DEFAULT_WARNING = False
scq.settings.PROGRESSBAR_DISABLED = True
scq.settings.FUZZY_SLICING = True

# set numpy 
np.set_printoptions(precision=5)

