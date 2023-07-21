import scqubits as scq

from chencrafts.bsqubits.error_rates import (
    ErrorChannel, 
    ErrorRate, 
    basic_channels,
    flxn_hf_flx_channels,
)

from chencrafts.bsqubits.systems import (
    ResonatorTransmon,
    ResonatorFluxonium,
    FluxoniumResonatorFluxonium,
) 

from chencrafts.bsqubits.batched_custom_sweeps import (
    batched_sweep_general,
    batched_sweep_bare_decoherence,
    batched_sweep_purcell_cats,
    batched_sweep_purcell_fock,
    batched_sweep_readout,
    batched_sweep_total_decoherence,
    batched_sweep_pulse,
    batched_sweep_cat_code,
)

from chencrafts.bsqubits.sims_n_sweeps import (
    cavity_ancilla_me_ingredients,
    idling_propagator,
)
import chencrafts.bsqubits.ideal_qec as ideal_qec

# scqubits settings
scq.set_units('GHz')
scq.settings.T1_DEFAULT_WARNING = False
scq.settings.PROGRESSBAR_DISABLED = True
scq.settings.FUZZY_SLICING = True
scq.settings.OVERLAP_THRESHOLD = 0.853