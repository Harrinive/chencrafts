import scqubits as scq

from chencrafts.bsqubits.ec_systems import (
    CavityTmonSys,
    CavityFlxnSys,
)
# from chencrafts.bsqubits.pulse_utils import *
# from chencrafts.bsqubits.qoc_init import *
from chencrafts.bsqubits.basis_n_states import (
    coherent,
    cat,
    projector_w_basis,
    oprt_in_basis,
)
from chencrafts.bsqubits.derive_var import (
    DerivedVariableTmon, 
    DerivedVariableFlxn, 
)
from chencrafts.bsqubits.error_rates import (
    ErrorChannel, 
    ErrorRate, 
    basic_channels,
    flxn_hf_flx_channels,
)

from chencrafts.bsqubits.systems import (
    transmon_resonator,
) 

# set scqubits
scq.set_units('GHz')
scq.settings.T1_DEFAULT_WARNING = False
scq.settings.PROGRESSBAR_DISABLED = True
scq.settings.FUZZY_SLICING = True